"""Patch the current Airbnb embedding extractor to support an image manifest.

Run in the project root after the CDN + disk-cache patches from steps 78-79:
    python apply_multi_image_patch.py

Does not download images, run models, or change existing Parquet/checkpoints.
"""
from __future__ import annotations

import ast
from pathlib import Path
import shutil
import sys

SOURCE = Path('extract_embedding.py')
BACKUP = Path('extract_embedding.py.before_multi_image.bak')
START = '\ndef run_image_phase('
END = '\ndef main() -> None:'

IMPLEMENTATION = '''
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
                    image_model_name: str, workers: int, max_images: int = 5) -> None:
    """Embed multiple images, then emit ONE normalized vector per listing.

    The output schema is unchanged, so Spark finalize and the fusion model
    continue to read listing_key/run_id/embedding without modification.
    """
    from sentence_transformers import SentenceTransformer

    image_sources = _load_image_manifest(data_dir, max_images)
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

'''


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f'Missing {SOURCE}; run this script from the project root')
    current = SOURCE.read_text(encoding='utf-8')
    if 'def _load_image_manifest(' in current:
        print('PATCH: ALREADY APPLIED')
        return
    if current.count(START) != 1 or current.count(END) != 1:
        raise SystemExit('ERROR: expected function anchors not found. Source unchanged.')
    if 'self._load_cached(url)' not in current or 'def _cdn_url(' not in current:
        raise SystemExit('ERROR: steps 78/79 CDN and cache updates not detected. Source unchanged.')
    if BACKUP.exists():
        raise SystemExit(f'ERROR: backup {BACKUP} already exists; source unchanged.')

    patched = current
    if 'import csv\n' not in patched:
        if 'import argparse\n' not in patched:
            raise SystemExit('ERROR: argparse import not found')
        patched = patched.replace('import argparse\n', 'import argparse\nimport csv\n', 1)

    start_at = patched.index(START)
    end_at = patched.index(END, start_at)
    patched = patched[:start_at] + '\n' + IMPLEMENTATION + patched[end_at:]

    old = '    parser.add_argument("--workers", type=int, default=4)'
    new = old + '\n    parser.add_argument("--max-images", type=int, default=5)'
    if patched.count(old) != 1:
        raise SystemExit('ERROR: argparse workers anchor mismatch; source unchanged.')
    patched = patched.replace(old, new, 1)

    validation = '    logging.basicConfig(level=logging.INFO)'
    if patched.count(validation) != 1:
        raise SystemExit('ERROR: logging anchor mismatch; source unchanged.')
    patched = patched.replace(validation,
        '    if not 1 <= args.max_images <= 10:\n'
        '        parser.error("--max-images must be between 1 and 10")\n' + validation, 1)

    call = '    run_image_phase(state, cfg, args.data_dir, args.image_model, args.workers)'
    if patched.count(call) != 1:
        raise SystemExit('ERROR: run_image_phase call mismatch; source unchanged.')
    patched = patched.replace(call,
        '    run_image_phase(state, cfg, args.data_dir, args.image_model, '\
        'args.workers, args.max_images)', 1)

    # Safety: verify patched Python syntax before writing to disk.
    ast.parse(patched, filename=str(SOURCE))
    shutil.copy2(SOURCE, BACKUP)
    SOURCE.write_text(patched, encoding='utf-8')
    print('PATCH: SUCCESS')
    print('Backup:', BACKUP)
    print('Output schema: unchanged (one image embedding per listing)')
    print('Default max images per listing: 5')
    print('No downloads, embeddings, or Spark jobs executed')

    # Report only manifest shape; does not require torch/model downloads.
    manifest = Path('data/listing_images.csv')
    if manifest.is_file():
        import csv
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        from collections import Counter
        by_listing = Counter(row['listing_key'] for row in rows)
        print('Manifest rows:', len(rows))
        print('Manifest listing count:', len(by_listing))
        print('Listings with >1 image:', sum(n > 1 for n in by_listing.values()))


if __name__ == '__main__':
    main()
