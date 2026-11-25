"""Run-scoped, bounded-memory, sequential E5 / CLIP embedding inference.

Reads Spark Silver Parquet in Arrow batches and writes embeddings keyed by
listing. Spark finalize left-joins the embeddings and marks missing modalities.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import ipaddress
import logging
import os
import socket
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError

from airbnb_image_urls import AIRBNB_IMAGE_HOSTS, resized_airbnb_image_url
from config import ModelConfig
from contracts.state import load_state, save_state
from contracts.storage import is_s3_uri, join_uri, pyarrow_fs_and_path, upload_file

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
    def _cdn_url(url: str) -> str | None:
        """Cap Airbnb requests at 720 px while preserving smaller thumbnails."""
        if urlparse(url).hostname not in AIRBNB_IMAGE_HOSTS:
            return url
        return resized_airbnb_image_url(url, max_width=720)

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
            request_url = self._cdn_url(url)
            if request_url is None:
                return None
            url = request_url
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
                        request_url = self._cdn_url(urljoin(url, loc))
                        if request_url is None:
                            return None
                        url = request_url
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


def _writer(file_path: str | Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("listing_key", pa.string()), ("run_id", pa.string()),
        ("embedding", pa.list_(pa.float32())),
    ])
    if is_s3_uri(file_path):
        fd, tmp_name = tempfile.mkstemp(suffix=".tmp.parquet")
        os.close(fd)
        tmp_path = Path(tmp_name)
    else:
        local_path = Path(file_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(".tmp.parquet")
    return pq.ParquetWriter(str(tmp_path), schema, compression="zstd"), schema, tmp_path


def _commit_parquet(temp_path: Path, target_path: str | Path) -> None:
    if is_s3_uri(target_path):
        try:
            upload_file(temp_path, target_path)
        finally:
            temp_path.unlink(missing_ok=True)
    else:
        temp_path.replace(Path(target_path))


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


def _count_rows(path: str | Path) -> int:
    import pyarrow.dataset as ds

    fs, pa_path = pyarrow_fs_and_path(path)
    return ds.dataset(pa_path, filesystem=fs, format="parquet").count_rows()


def _batch_iter(path: str | Path, columns: list[str], batch_size: int = 128):
    import pyarrow.dataset as ds

    fs, pa_path = pyarrow_fs_and_path(path)
    dataset = ds.dataset(pa_path, filesystem=fs, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        yield batch.to_pydict()


def run_text_phase(state: dict, cfg: ModelConfig, text_model_name: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer
    print("[Text] Loading E5:", text_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(text_model_name, device=device)
    try:
        dim = encoder.get_sentence_embedding_dimension()
        if dim != cfg.raw_text_dim:
            raise ValueError(f"Text dimension mismatch: encoder={dim}, config={cfg.raw_text_dim}")
        for split in ("train", "val", "test"):
            silver_path = join_uri(state["run_path"], "silver", split)
            total_rows = _count_rows(silver_path)
            path = join_uri(state["run_path"], "embeddings", f"{split}_text.parquet")
            writer, schema, temp = _writer(path)
            processed = n = 0
            started = time.monotonic()
            last_log = started
            print(f"[Text] {split}: starting ({total_rows} rows)", flush=True)
            try:
                for record in _batch_iter(silver_path, ["listing_key", "clean_full_text"]):
                    keys = record["listing_key"]
                    texts = record["clean_full_text"]
                    processed += len(keys)
                    accepted = [(k, t) for k, t in zip(keys, texts)
                                if isinstance(t, str) and t.strip()]
                    if accepted:
                        embeddings = encoder.encode(["passage: " + t for _, t in accepted],
                            batch_size=cfg.text_batch_size, show_progress_bar=False,
                            convert_to_numpy=True, normalize_embeddings=True)
                        _write_embeddings(writer, schema, [k for k, _ in accepted],
                                          state["run_id"], np.asarray(embeddings, dtype=np.float32))
                        n += len(accepted)
                    now = time.monotonic()
                    if now - last_log >= 10.0 or processed >= total_rows:
                        pct = (100.0 * processed / total_rows) if total_rows else 100.0
                        print(f"[Text] {split}: {processed}/{total_rows} ({pct:.1f}%) | embedded={n} | {now - started:.0f}s", flush=True)
                        last_log = now
            finally:
                writer.close()
            _commit_parquet(temp, path)
            print(f"[Text] {split}: done ({n} embeddings)", flush=True)
    finally:
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_image_phase(state: dict, cfg: ModelConfig, data_dir: str,
                    image_model_name: str, workers: int) -> None:
    """Write one normalized CLIP embedding for each usable Silver cover photo."""
    import torch
    from sentence_transformers import SentenceTransformer

    print("[Image] Loading CLIP:", image_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(image_model_name, device=device)
    cache_dir = Path("/tmp/image_cache") if is_s3_uri(state["run_path"]) else Path(data_dir) / "cache" / "images"
    downloader = SafeImageDownloader(cache_dir=cache_dir)
    try:
        dim = encoder.get_sentence_embedding_dimension()
        if dim != cfg.raw_image_dim:
            raise ValueError(f"Image dimension mismatch: encoder={dim}, config={cfg.raw_image_dim}")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for split in ("train", "val", "test"):
                silver_path = join_uri(state["run_path"], "silver", split)
                total_rows = _count_rows(silver_path)
                path = join_uri(state["run_path"], "embeddings", f"{split}_image.parquet")
                writer, schema, temp = _writer(path)
                total = valid = 0
                started = time.monotonic()
                last_log = started
                print(f"[Image] {split}: starting ({total_rows} rows)", flush=True)
                try:
                    for record in _batch_iter(silver_path, ["listing_key", "picture_url"], batch_size=16):
                        total += len(record["listing_key"])
                        jobs = [(key, url) for key, url in zip(record["listing_key"],
                                                              record["picture_url"], strict=True)
                                if isinstance(url, str) and url]
                        images = list(pool.map(downloader.download, [url for _, url in jobs]))
                        try:
                            accepted = [(key, image) for (key, _), image in zip(jobs, images, strict=True)
                                        if image is not None]
                            if accepted:
                                embeddings = encoder.encode(
                                    [image for _, image in accepted],
                                    batch_size=cfg.image_batch_size,
                                    show_progress_bar=False,
                                    convert_to_numpy=True,
                                    normalize_embeddings=True,
                                )
                                vectors = np.asarray(embeddings, dtype=np.float32)
                                if vectors.shape != (len(accepted), dim):
                                    raise ValueError(f"Unexpected image embeddings shape: {vectors.shape}")
                                norms = np.linalg.norm(vectors, axis=1)
                                usable = np.isfinite(norms) & (norms > 1e-12)
                                keys = [key for (key, _), keep in zip(accepted, usable) if keep]
                                if keys:
                                    normalized = vectors[usable] / norms[usable, None]
                                    _write_embeddings(writer, schema, keys, state["run_id"], normalized)
                                    valid += len(keys)
                        finally:
                            for image in images:
                                if image is not None:
                                    image.close()
                        now = time.monotonic()
                        if now - last_log >= 10.0 or total >= total_rows:
                            pct = (100.0 * total / total_rows) if total_rows else 100.0
                            print(f"[Image] {split}: {total}/{total_rows} ({pct:.1f}%) | embedded={valid} | {now - started:.0f}s", flush=True)
                            last_log = now
                finally:
                    writer.close()
                _commit_parquet(temp, path)
                print(f"[Image] {split}: {valid} listings embedded, "
                      f"{total - valid} without a usable cover embedding", flush=True)
    finally:
        del encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-id", help="Use this prepared run.")
    parser.add_argument("--text-model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--image-model", default="clip-ViT-B-32")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    logging.basicConfig(level=logging.INFO)
    state = load_state(args.data_dir, run_id=args.run_id)
    if state.get("status") not in ("prepared", "embedded"):
        raise RuntimeError(f"Pipeline status is {state.get('status')!r}; expected 'prepared' or 'embedded'.")
    if state["status"] == "embedded":
        # A failed refresh must not leave a partially replaced run ready for finalize.
        state["status"] = "prepared"
        state.pop("embedding_config", None)
        save_state(args.data_dir, state, activate=args.run_id is None)
    cfg = ModelConfig()
    run_text_phase(state, cfg, args.text_model)
    # The E5 encoder is released before CLIP is created to fit in 6 GB VRAM.
    run_image_phase(state, cfg, args.data_dir, args.image_model, args.workers)
    state["embedding_config"] = {"mode": "pretrained", "text_model": args.text_model,
                                 "image_model": args.image_model, "image_source": "cover"}
    state["status"] = "embedded"
    save_state(args.data_dir, state, activate=args.run_id is None)
    print("Next: python -m spark_etl.run_etl finalize --data-dir", args.data_dir)


if __name__ == "__main__":
    main()
