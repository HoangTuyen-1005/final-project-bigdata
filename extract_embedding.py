"""Run-scoped, bounded-memory, sequential E5 / CLIP embedding inference.

Reads Spark Silver Parquet in Arrow batches; writes keyed Parquet. A failed image
produces NO image row (never a zero vector). Finalize uses inner joins.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import os
import uuid
import ipaddress
import logging
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode

import numpy as np
import requests
import torch
from PIL import Image, UnidentifiedImageError

from config import ModelConfig
from project_state import load_state, save_state

LOGGER = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = 60_000_000


class SafeImageDownloader:
    """Limits bad images, redirects and obvious private-network URLs.

    Not a full network sandbox against DNS rebinding; use egress ACLs for untrusted input.
    """

    def __init__(self, timeout: float = 10.0, max_bytes: int = 20_000_000,
                 max_redirects: int = 3, max_pixels: int = 60_000_000,
                 target_size: tuple[int, int] = (768, 768),
                 cache_dir: str | Path = "data/cache/images") -> None:
        self.timeout, self.max_bytes = timeout, max_bytes
        self.max_redirects, self.max_pixels = max_redirects, max_pixels
        self.target_size = target_size
        self.cache_dir = Path(cache_dir)
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.trust_env = False  # don't route requests through opaque env proxies
            session.headers.update({"User-Agent": "AirbnbResearchImageEncoder/1.0"})
            self._local.session = session
        return self._local.session

    @staticmethod
    def _check_public_https(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Only public HTTPS image URLs without credentials are supported")
        if parsed.port not in (None, 443):
            raise ValueError("Nonstandard HTTPS port is blocked")
        hostname = parsed.hostname
        if hostname.lower() == "localhost" or hostname.endswith(".local"):
            raise ValueError("Local hostname blocked")
        results = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        if not results:
            raise ValueError("Hostname did not resolve")
        for info in results:
            address = ipaddress.ip_address(info[4][0])
            if not address.is_global:
                raise ValueError("Private/reserved destination blocked")

    @staticmethod
    def _cdn_url(url: str) -> str:
        """Request a smaller image from the verified Airbnb image CDN."""
        parsed = urlparse(url)

        if (
            parsed.hostname != "a0.muscache.com"
            or not parsed.path.startswith("/pictures/")
        ):
            return url

        query = [
            (key, value)
            for key, value in parse_qsl(
                parsed.query, keep_blank_values=True
            )
            if key != "im_w"
        ]
        query.append(("im_w", "720"))

        return parsed._replace(
            path="/im" + parsed.path,
            query=urlencode(query)
        ).geturl()

    def _cache_path(self, url: str) -> Path:
        name = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".jpg"
        return self.cache_dir / name

    def _load_cached(self, url: str) -> Image.Image | None:
        path = self._cache_path(url)
        if not path.is_file():
            return None

        try:
            with Image.open(path) as img:
                img.load()
                if img.width < 32 or img.height < 32:
                    raise ValueError("Cached image too small")
                if img.width * img.height > self.max_pixels:
                    raise ValueError("Cached image too large")
                return img.convert("RGB")

        except (OSError, ValueError, UnidentifiedImageError,
                Image.DecompressionBombError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def _save_cached(self, url: str, image: Image.Image) -> None:
        path = self._cache_path(url)
        temp = path.with_name(
            path.stem + "." + uuid.uuid4().hex + ".tmp"
        )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(temp, format="JPEG", quality=90)
            os.replace(temp, path)

        except OSError:
            LOGGER.debug("Could not write image cache")

        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def download(self, url: str) -> Image.Image | None:
        if not isinstance(url, str) or not url:
            return None
        try:
            url = self._cdn_url(url)
            cached = self._load_cached(url)
            if cached is not None:
                return cached
            cache_url = url
            session = self._session()
            for redirect in range(self.max_redirects + 1):
                self._check_public_https(url)
                with session.get(url, timeout=self.timeout, stream=True,
                                 allow_redirects=False) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        loc = response.headers.get("Location")
                        if not loc or redirect == self.max_redirects:
                            return None
                        url = urljoin(url, loc)
                        continue
                    if response.status_code != 200:
                        return None
                    content_type = response.headers.get("Content-Type", "").lower()
                    if content_type and not content_type.startswith("image/"):
                        return None
                    length = response.headers.get("Content-Length")
                    if length and int(length) > self.max_bytes:
                        return None
                    chunks, total = [], 0
                    for chunk in response.iter_content(chunk_size=65536):
                        total += len(chunk)
                        if total > self.max_bytes:
                            return None
                        chunks.append(chunk)
                    with Image.open(BytesIO(b"".join(chunks))) as decoded:
                        if decoded.width < 32 or decoded.height < 32:
                            return None
                        if decoded.width * decoded.height > self.max_pixels:
                            return None
                        converted = decoded.convert("RGB")
                        converted.thumbnail(self.target_size, Image.Resampling.LANCZOS)
                        self._save_cached(cache_url, converted)
                        cached = self._load_cached(cache_url)
                        if cached is not None:
                            converted.close()
                            return cached
                        return converted
        except (requests.RequestException, OSError, ValueError, OverflowError,
                UnidentifiedImageError, Image.DecompressionBombError) as error:
            LOGGER.debug("Image unavailable: %s", type(error).__name__)
        return None


def _writer(file_path: Path, dim: int):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("listing_key", pa.string()), ("run_id", pa.string()),
        ("embedding", pa.list_(pa.float32())),
    ])
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_suffix(".tmp.parquet")
    return pq.ParquetWriter(str(tmp_path), schema, compression="zstd"), schema, tmp_path


def _write_embeddings(writer, schema, keys: list[str], run_id: str,
                      vectors: np.ndarray) -> None:
    import pyarrow as pa
    if len(keys) != len(vectors):
        raise ValueError("Keys and embedding rows differ")
    if not keys:
        return
    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite embedding returned by encoder")
    writer.write_table(pa.Table.from_arrays([
        pa.array(keys, type=pa.string()),
        pa.array([run_id] * len(keys), type=pa.string()),
        pa.array(vectors.tolist(), type=pa.list_(pa.float32())),
    ], schema=schema))


def _batch_iter(path: Path, columns: list[str], batch_size: int = 128):
    import pyarrow.dataset as ds
    scanner = ds.dataset(str(path), format="parquet").scanner(columns=columns,
                                                                  batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pydict()


def run_text_phase(state: dict, cfg: ModelConfig, data_dir: str,
                   text_model_name: str) -> None:
    from sentence_transformers import SentenceTransformer
    print("[Text] Loading E5:", text_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(text_model_name, device=device)
    try:
        dim = encoder.get_sentence_embedding_dimension()
        if dim != cfg.raw_text_dim:
            raise ValueError(f"Text dimension mismatch: encoder={dim}, config={cfg.raw_text_dim}")
        for split in ("train", "val", "test"):
            path = state["run_path"] / "embeddings" / f"{split}_text.parquet"
            writer, schema, temp = _writer(path, dim)
            n = 0
            try:
                for record in _batch_iter(state["run_path"] / "silver" / split,
                                          ["listing_key", "clean_full_text"]):
                    keys = record["listing_key"]
                    texts = record["clean_full_text"]
                    accepted = [(k, t) for k, t in zip(keys, texts)
                                if isinstance(t, str) and t.strip()]
                    if accepted:
                        embeddings = encoder.encode(["passage: " + t for _, t in accepted],
                            batch_size=cfg.text_batch_size, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
                        _write_embeddings(writer, schema, [k for k, _ in accepted],
                                          state["run_id"], np.asarray(embeddings, dtype=np.float32))
                        n += len(accepted)
            finally:
                writer.close()
            temp.replace(path)
            print(f"[Text] {split}: {n} embeddings")
    finally:
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def _load_image_manifest(data_dir: str, max_images: int) -> dict[str, list[str]]:
    """Load <=max_images distinct URLs per listing, in image_order order.

    If the manifest is missing, existing Silver picture_url remains the fallback.
    This dictionary holds URLs only, not pixel arrays, so memory remains bounded.
    """
    manifest_path = Path(data_dir) / "listing_images.csv"
    if not manifest_path.is_file():
        print(f"[Image] Manifest missing ({manifest_path}); using Silver covers")
        return {}

    staged: dict[str, list[tuple[int, str]]] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"listing_key", "image_url", "image_order"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Invalid image manifest columns: {reader.fieldnames}")

        for row in reader:
            key = (row.get("listing_key") or "").strip()
            url = (row.get("image_url") or "").strip()
            if not key or not url.startswith("https://"):
                continue
            try:
                order = int(row["image_order"])
            except (ValueError, TypeError):
                continue
            if order < 0:
                continue
            staged.setdefault(key, []).append((order, url))

    manifest: dict[str, list[str]] = {}
    for key, candidates in staged.items():
        urls: list[str] = []
        seen: set[str] = set()
        for _, url in sorted(candidates, key=lambda pair: pair[0]):
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
            if len(urls) == max_images:
                break
        if urls:
            manifest[key] = urls

    n_urls = sum(len(urls) for urls in manifest.values())
    print(f"[Image] Manifest: {len(manifest)} listings, {n_urls} candidate images "
          f"(up to {max_images}/listing)")
    return manifest


def run_image_phase(state: dict, cfg: ModelConfig, data_dir: str,
                    image_model_name: str, workers: int, max_images: int = 1) -> None:
    """Embed multiple images, then emit ONE normalized vector per listing.

    The output schema is unchanged, so Spark finalize and the fusion model
    continue to read listing_key/run_id/embedding without modification.
    """
    from sentence_transformers import SentenceTransformer

    image_sources = (_load_image_manifest(data_dir, max_images) if max_images > 1 else {})
    print("[Image] Loading CLIP:", image_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(image_model_name, device=device)
    downloader = SafeImageDownloader(cache_dir=Path(data_dir) / "cache" / "images")
    try:
        dim = encoder.get_sentence_embedding_dimension()
        if dim != cfg.raw_image_dim:
            raise ValueError(f"Image dimension mismatch: encoder={dim}, config={cfg.raw_image_dim}")

        # Reuse a thread pool rather than recreating it for every Arrow batch.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for split in ("train", "val", "test"):
                path = state["run_path"] / "embeddings" / f"{split}_image.parquet"
                writer, schema, temp = _writer(path, dim)
                valid = invalid = images_ok = 0
                try:
                    for record in _batch_iter(state["run_path"] / "silver" / split,
                                              ["listing_key", "picture_url"],
                                              batch_size=32):
                        # Microbatches prevent holding hundreds of decoded images in RAM.
                        for start in range(0, len(record["listing_key"]), 16):
                            keys = record["listing_key"][start:start + 16]
                            covers = record["picture_url"][start:start + 16]
                            jobs: list[tuple[str, str]] = []
                            for key, cover in zip(keys, covers):
                                urls = image_sources.get(key)
                                if not urls:
                                    urls = [cover] if isinstance(cover, str) and cover else []
                                jobs.extend((key, url) for url in urls)

                            images = list(pool.map(downloader.download,
                                                   [url for _, url in jobs]))
                            try:
                                accepted = [(i, img) for i, img in enumerate(images)
                                            if img is not None]
                                images_ok += len(accepted)
                                pooled: dict[str, list[np.ndarray]] = {}

                                if accepted:
                                    vectors = encoder.encode(
                                        [img for _, img in accepted],
                                        batch_size=cfg.image_batch_size,
                                        show_progress_bar=False,
                                        convert_to_numpy=True,
                                        normalize_embeddings=True,
                                    )
                                    vectors = np.asarray(vectors, dtype=np.float32)
                                    for (job_index, _), vector in zip(accepted, vectors):
                                        key = jobs[job_index][0]
                                        pooled.setdefault(key, []).append(vector)

                                out_keys: list[str] = []
                                out_vectors: list[np.ndarray] = []
                                for key in keys:
                                    member_vectors = pooled.get(key)
                                    if not member_vectors:
                                        invalid += 1
                                        continue
                                    mean_vector = np.mean(member_vectors, axis=0)
                                    norm = float(np.linalg.norm(mean_vector))
                                    if not np.isfinite(norm) or norm <= 1e-12:
                                        invalid += 1
                                        continue
                                    out_keys.append(key)
                                    out_vectors.append((mean_vector / norm).astype(np.float32))
                                    valid += 1

                                if out_keys:
                                    _write_embeddings(writer, schema, out_keys,
                                                      state["run_id"],
                                                      np.stack(out_vectors))
                            finally:
                                for image in images:
                                    if image is not None:
                                        image.close()
                finally:
                    writer.close()
                temp.replace(path)
                print(f"[Image] {split}: {valid} listings embedded, "
                      f"{invalid} without usable images; {images_ok} images encoded")
    finally:
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--text-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--image-model", default="clip-ViT-B-32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    if not 1 <= args.max_images <= 10:
        parser.error("--max-images must be between 1 and 10")
    logging.basicConfig(level=logging.INFO)
    state = load_state(args.data_dir)
    if state.get("status") not in ("prepared", "embedded"):
        raise RuntimeError(f"Pipeline status is {state.get('status')!r}; expected 'prepared' or 'embedded'.")
    cfg = ModelConfig()
    run_text_phase(state, cfg, args.data_dir, args.text_model)
    # The E5 encoder is released before CLIP is created to fit in 6 GB VRAM.
    run_image_phase(state, cfg, args.data_dir, args.image_model, args.workers, args.max_images)
    state["status"] = "embedded"
    save_state(args.data_dir, state)
    print("Next: python -m spark_etl.run_etl finalize --data-dir", args.data_dir)


if __name__ == "__main__":
    main()
