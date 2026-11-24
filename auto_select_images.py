"""Automatic CLIP-based selection of representative listing photos.

Input is an authorized image-URL manifest (not a listing-page URL):
  listing_key,image_url,image_order[,image_role,source]

Example (run from project root, in the existing venv):
  python auto_select_images.py --input data/manual_album_prototype.csv \
      --output data/selected_album_prototype.csv --max-images 3

The output is compatible with extract_embedding.py's _load_image_manifest().
This tool NEVER crawls Airbnb pages or guesses UUIDs. Image discovery needs
an independently authorized source of per-listing album URLs.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
from PIL import Image, ImageOps

# SentenceTransformer('clip-ViT-B-32') embeds both natural-language prompts
# and PIL images into the same 512-dimensional space.
PROPERTY_PROMPTS = {
    "bedroom": "A photo of a bedroom with a bed inside a vacation rental",
    "bathroom": "A photo of a bathroom with a shower or bathtub in an apartment",
    "living_room": "A photo of a furnished living room inside a vacation rental",
    "kitchen": "A photo of a kitchen with cupboards and appliances in a home",
    "dining_room": "A photo of a dining area with a table and chairs in a home",
    "workspace": "A photo of a desk and workspace inside an apartment",
    "exterior": "A photo of the exterior of a house or apartment building",
    "outdoor": "A photo of a patio, garden, balcony or terrace at a home",
    "amenity": "A photo of a swimming pool or other amenity at a vacation rental",
}
NON_PROPERTY_PROMPTS = {
    "person": "A close-up portrait photograph of a person's face",
    "map": "A map showing streets and locations",
    "logo": "A graphic with a business logo and text",
    "food": "A close-up photograph of a prepared meal on a plate",
}
ALLOWED_HOSTS = {"a0.muscache.com", "z1.muscache.cn"}
OUTPUT_FIELDS = [
    "listing_key", "image_url", "image_order", "image_role", "source",
    "clip_relevance", "clip_margin", "selection_score",
]


@dataclass
class Candidate:
    url: str
    order: int
    original_source: str
    image: Image.Image
    dhash: int = 0
    role: str = ""
    relevance: float = 0.0
    margin: float = 0.0
    vector: np.ndarray | None = None


def cdn_image_url(url: str, width: int) -> str | None:
    """Use thumbnail delivery only for known CDN URL patterns.

    For the unverified z1 CDN, accept only URLs already using /im/pictures/;
    do not silently download a large /original/ image.
    """
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or host not in ALLOWED_HOSTS:
            return None
        if parsed.username or parsed.password or parsed.port not in (None, 443):
            return None
        if parsed.path.startswith("/im/pictures/"):
            path = parsed.path
        elif host == "a0.muscache.com" and parsed.path.startswith("/pictures/"):
            path = "/im" + parsed.path
        else:
            return None
        query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                 if k != "im_w"]
        query.append(("im_w", str(width)))
        return urlunsplit((parsed.scheme, parsed.netloc, path,
                           urlencode(query), ""))
    except ValueError:
        return None


def dhash(image: Image.Image) -> int:
    """64-bit difference hash, used only with CLIP similarity to flag near duplicates."""
    pixels = np.asarray(ImageOps.grayscale(image).resize((9, 8)), dtype=np.int16)
    value = 0
    for bit in (pixels[:, 1:] > pixels[:, :-1]).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def load_manifest(path: Path, max_candidates: int) -> dict[str, list[tuple[int, str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input manifest does not exist: {path}")
    rows: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"listing_key", "image_url", "image_order"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must include {sorted(required)}")
        for row in reader:
            key = (row.get("listing_key") or "").strip()
            url = (row.get("image_url") or "").strip()
            if not key or not url or url in seen[key]:
                continue
            try:
                order = int(row.get("image_order") or "")
            except ValueError:
                continue
            if order < 0:
                continue
            # Validate before including a candidate, not after an arbitrary limit.
            if cdn_image_url(url, 320) is None:
                continue
            seen[key].add(url)
            rows[key].append((order, url, (row.get("source") or "manifest").strip()))
    # Keep the original album order, with a bounded candidate count per listing.
    return {k: sorted(v, key=lambda x: x[0])[:max_candidates]
            for k, v in rows.items() if v}


def is_duplicate(candidate: Candidate, previous: Candidate) -> bool:
    sim = float(np.dot(candidate.vector, previous.vector))
    hamming = (candidate.dhash ^ previous.dhash).bit_count()
    # Identical/near-identical pixels and the same visual content. Do not reject
    # two different bedrooms merely because their CLIP vectors are similar.
    return sim > 0.998 or (hamming <= 4 and sim > 0.94)


def choose_images(candidates: list[Candidate], max_images: int) -> list[tuple[Candidate, float]]:
    """Greedy quality/relevance + scene diversity ranking; no manual roles."""
    # Retain only one instance of near-duplicates, favoring clearer CLIP match.
    distinct: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda x: x.relevance, reverse=True):
        if not any(is_duplicate(candidate, prev) for prev in distinct):
            distinct.append(candidate)
    chosen: list[tuple[Candidate, float]] = []
    while distinct and len(chosen) < max_images:
        def score(c: Candidate) -> float:
            if not chosen:
                return c.relevance
            max_similarity = max(float(np.dot(c.vector, s.vector)) for s, _ in chosen)
            has_new_role = c.role not in {s.role for s, _ in chosen}
            # CLIP relevance dominates; novelty and category diversity help
            # avoid redundant angles of the same room.
            return c.relevance - 0.24 * max(0.0, max_similarity - 0.55) + (
                0.035 if has_new_role else 0.0
            )
        best = max(distinct, key=score)
        chosen.append((best, score(best)))
        distinct.remove(best)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="CSV containing authorized album image URLs")
    parser.add_argument("--output", required=True, type=Path,
                        help="New selected image manifest (never overwrite input)")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=40,
                        help="Maximum album photos to inspect per listing")
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--selected-width", type=int, default=720)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/images"))
    parser.add_argument("--model", default="clip-ViT-B-32")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force", action="store_true", help="Allow overwriting output only")
    args = parser.parse_args()

    if not 1 <= args.max_images <= 10:
        parser.error("--max-images must be 1..10")
    if not args.max_images <= args.max_candidates <= 100:
        parser.error("--max-candidates must be >= --max-images and <= 100")
    if not 224 <= args.thumb_width <= 1024:
        parser.error("--thumb-width must be 224..1024")
    if not 224 <= args.selected_width <= 2048:
        parser.error("--selected-width must be 224..2048")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ")
    if args.output.exists() and not args.force:
        parser.error(f"Output already exists: {args.output} (use a new path)")

    manifest = load_manifest(args.input, args.max_candidates)
    print(f"MANIFEST: {len(manifest)} listings, "
          f"{sum(map(len, manifest.values()))} candidate URLs", flush=True)
    if not manifest:
        raise SystemExit("No supported CDN image URLs found; output unchanged")

    import torch
    from sentence_transformers import SentenceTransformer
    from extract_embedding import SafeImageDownloader

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"CLIP: {args.model} | device={device}", flush=True)
    encoder = SentenceTransformer(args.model, device=device)
    labels = list(PROPERTY_PROMPTS) + list(NON_PROPERTY_PROMPTS)
    prompts = list(PROPERTY_PROMPTS.values()) + list(NON_PROPERTY_PROMPTS.values())
    text_vectors = np.asarray(encoder.encode(
        prompts, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False), dtype=np.float32)
    downloader = SafeImageDownloader(cache_dir=args.cache_dir)

    selected_rows: list[dict] = []
    counts: Counter = Counter()
    n_property = len(PROPERTY_PROMPTS)

    for number, (listing_key, entries) in enumerate(manifest.items(), 1):
        candidates: list[Candidate] = []
        for order, url, source in entries:
            thumbnail = cdn_image_url(url, args.thumb_width)
            if thumbnail is None:
                counts["unsupported"] += 1
                continue
            img = downloader.download(thumbnail)
            if img is None:
                counts["download_failed"] += 1
                continue
            if min(img.size) < 160:
                img.close()
                counts["too_small"] += 1
                continue
            candidates.append(Candidate(url, order, source, img, dhash(img)))

        try:
            if not candidates:
                counts["listings_without_images"] += 1
                print(f"[{number}/{len(manifest)}] {listing_key}: 0 usable photos", flush=True)
                continue
            vectors = np.asarray(encoder.encode(
                [c.image for c in candidates], batch_size=8,
                convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False), dtype=np.float32)
            similarities = vectors @ text_vectors.T
            usable: list[Candidate] = []
            for idx, c in enumerate(candidates):
                c.vector = vectors[idx]
                positive_idx = int(np.argmax(similarities[idx, :n_property]))
                best_property = float(similarities[idx, positive_idx])
                best_other = float(np.max(similarities[idx, n_property:]))
                c.role = labels[positive_idx]
                c.relevance = best_property
                c.margin = best_property - best_other
                # Do not hard-code high 'confidence' from uncalibrated CLIP
                # similarities; reject only clear unrelated-image matches.
                if c.margin < -0.02:
                    counts["non_property"] += 1
                    continue
                usable.append(c)
            selected = choose_images(usable, args.max_images)
            counts["duplicates_or_unselected"] += len(usable) - len(selected)
            for rank, (c, selection_score) in enumerate(selected):
                final_url = cdn_image_url(c.url, args.selected_width)
                assert final_url is not None
                selected_rows.append({
                    "listing_key": listing_key,
                    "image_url": final_url,
                    "image_order": rank,
                    "image_role": c.role,
                    "source": f"auto_clip:{c.original_source}",
                    "clip_relevance": f"{c.relevance:.5f}",
                    "clip_margin": f"{c.margin:+.5f}",
                    "selection_score": f"{selection_score:.5f}",
                })
                counts["selected"] += 1
                print(f"  {rank + 1}. {c.role:<12} CLIP={c.relevance:.3f} "
                      f"margin={c.margin:+.3f} {c.url.rsplit('/', 1)[-1][:50]}", flush=True)
            print(f"[{number}/{len(manifest)}] {listing_key}: "
                  f"{len(candidates)} decoded, {len(selected)} selected", flush=True)
        finally:
            for c in candidates:
                c.image.close()

    # Write only after all processed. Atomic replace prevents partial output.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + ".tmp")
    if temp.exists():
        raise SystemExit(f"Temporary output already exists: {temp}")
    try:
        with temp.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(selected_rows)
        temp.replace(args.output)
    finally:
        temp.unlink(missing_ok=True)

    print(f"OUTPUT: {args.output} | selected={counts['selected']} "
          f"| failed_downloads={counts['download_failed']} "
          f"| non_property={counts['non_property']}", flush=True)
    if not selected_rows:
        print("WARNING: No images selected; inspect input URLs and CDN availability")


if __name__ == "__main__":
    main()
