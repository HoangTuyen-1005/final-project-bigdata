"""Small, read-only CLIP probe for the three Albany Allen Home albums.

Run from the project root:
    python probe_multi_image.py --zip check.zip

Reads only check.zip; does NOT touch Silver/Gold, checkpoints, or pipeline_state.json.
Uses the same SentenceTransformer CLIP-ViT-B-32 encoder and image resizing as
extract_embedding.py. Encoding is done once per SHA-256-unique photo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image

MODEL = "clip-ViT-B-32"


def norm(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    n = np.linalg.norm(vector)
    if n < 1e-9:
        raise ValueError("Empty embedding vector")
    return vector / n


def load_album(zip_path: Path):
    """Return album -> ordered entries and digest -> image bytes."""
    albums: dict[str, list[dict]] = {}
    blobs: dict[str, bytes] = {}
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for metadata_path in sorted(n for n in names if n.endswith("/metadata.json")):
            metadata = json.loads(archive.read(metadata_path).decode("utf-8-sig"))
            folder = metadata_path.rsplit("/", 1)[0]
            room = metadata.get("room_title", folder)
            if room in albums:
                raise ValueError(f"Duplicate room: {room}")
            photos = []
            for row in metadata.get("photos", []):
                if not row.get("success", True):
                    continue
                filename = row.get("filename")
                name = f"{folder}/{filename}"
                if not filename or name not in names:
                    raise FileNotFoundError(f"Missing image listed by metadata: {name}")
                payload = archive.read(name)
                digest = hashlib.sha256(payload).hexdigest()
                blobs.setdefault(digest, payload)
                photos.append({
                    "sha256": digest,
                    "filename": filename,
                    "category": (row.get("stop_name") or "").lower(),
                    "index": row.get("index", len(photos) + 1),
                })
            if not photos:
                raise ValueError(f"No valid photos found for {room}")
            albums[room] = sorted(photos, key=lambda p: p["index"])
    if len(albums) != 3:
        raise ValueError(f"Expected three albums, got {len(albums)}")
    return albums, blobs


def unique_shas(photos: list[dict]) -> list[str]:
    return list(dict.fromkeys(p["sha256"] for p in photos))


def is_bedroom(entry: dict) -> bool:
    return "phong_ngu" in entry["filename"].lower() or "phòng ngủ" in entry["category"]


def selected_five(photos: list[dict]) -> list[str]:
    """2 private bedroom views, and one each of shared living, kitchen, bathroom."""
    beds = unique_shas([p for p in photos if is_bedroom(p)])[:2]
    picks = beds[:]
    categories = ("phong_khach", "bep", "phong_tam")
    for cat in categories:
        matches = [p for p in photos if cat in p["filename"].lower()]
        if matches:
            d = matches[0]["sha256"]
            if d not in picks:
                picks.append(d)
    if len(beds) == 0:
        raise ValueError("No bedroom photos: cannot test bedroom representation")
    if len(picks) < 4:
        print(f"Warning: selected_five found only {len(picks)} categories")
    return picks


def mean_normalized(digests: list[str], embeddings: dict[str, np.ndarray]) -> np.ndarray:
    return norm(np.mean(np.stack([embeddings[d] for d in digests]), axis=0))


def encode_unique(blobs: dict[str, bytes], model_name: str, batch_size: int):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoder={model_name}; device={device}; unique_photos={len(blobs)}")
    encoder = SentenceTransformer(model_name, device=device)
    digests = list(blobs)
    vectors = []
    for start in range(0, len(digests), batch_size):
        chunk = digests[start : start + batch_size]
        imgs = []
        try:
            for d in chunk:
                with Image.open(BytesIO(blobs[d])) as decoded:
                    rgb = decoded.convert("RGB")
                    rgb.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    imgs.append(rgb)
            batch = encoder.encode(
                imgs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors.extend(np.asarray(batch, dtype=np.float32))
        finally:
            for im in imgs:
                im.close()
    if len(vectors) != len(digests):
        raise ValueError("CLIP did not return one embedding per photo")
    return {d: norm(v) for d, v in zip(digests, vectors)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="check.zip", type=Path)
    parser.add_argument("--out", default=Path("data/multi_image_probe"), type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not args.zip.is_file():
        parser.error(f"No album ZIP at {args.zip}; put check.zip in project root or provide --zip")
    if args.batch_size < 1 or args.batch_size > 32:
        parser.error("batch-size must be between 1 and 32")

    albums, blobs = load_album(args.zip)
    print("Photo inventory:")
    for name, entries in albums.items():
        beds = unique_shas([e for e in entries if is_bedroom(e)])
        print(f"  {name}: {len(entries)} files, {len(unique_shas(entries))} unique, {len(beds)} bedroom")

    embeddings = encode_unique(blobs, args.model, args.batch_size)
    representations = {}
    for name, entries in albums.items():
        room_photos = unique_shas([e for e in entries if is_bedroom(e)])
        representations[name] = {
            "cover": embeddings[entries[0]["sha256"]],
            "bedroom_mean": mean_normalized(room_photos, embeddings),
            "balanced_5": mean_normalized(selected_five(entries), embeddings),
            "full_album_mean": mean_normalized(unique_shas(entries), embeddings),
        }

    rows = []
    names = list(representations)
    print("\nCosine similarity BETWEEN ROOMS (closer to 1 = more similar):")
    print("Mode                 Pairs (3)              Mean similarity   Mean distance")
    for mode in ("cover", "bedroom_mean", "balanced_5", "full_album_mean"):
        sims = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = float(np.clip(np.dot(representations[names[i]][mode],
                                           representations[names[j]][mode]), -1, 1))
                sims.append(sim)
                rows.append({"mode": mode, "room_a": names[i], "room_b": names[j],
                             "cosine_similarity": round(sim, 8),
                             "cosine_distance": round(1 - sim, 8)})
        print(f"{mode:<20} {', '.join(f'{s:.4f}' for s in sims):<25} {np.mean(sims):.4f}            {1-np.mean(sims):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "pairwise_cosine.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "room_a", "room_b", "cosine_similarity", "cosine_distance"])
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps({
        "model": args.model,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "unique_photo_count": len(blobs),
        "room_photo_count": {n: len(a) for n, a in albums.items()},
        "pairwise": rows,
        "interpretation": "CLIP geometry only; this does not measure price prediction quality.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {csv_path}")
    print("Meaning: cover should be ~1.0 for all pairs (same image); bedroom_mean should separate rooms if CLIP captures differences.")
    print("These metrics alone DO NOT prove Multi-Image improves price prediction.")


if __name__ == "__main__":
    main()
