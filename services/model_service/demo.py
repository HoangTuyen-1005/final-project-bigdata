"""Deterministic fixtures for exercising both Docker services without encoder downloads."""
import argparse
import csv
import hashlib
from io import StringIO
from pathlib import Path

from config import ModelConfig
from contracts.state import load_state, resolve_data_root, save_state
from contracts.storage import is_s3_uri, join_uri, read_bytes, uri_exists, write_bytes


def write_demo_inputs(data_dir: str | Path, cfg: ModelConfig) -> None:
    root = resolve_data_root(data_dir)
    fields = ["id", "price", "currency", "name", "description", "last_scraped",
              "amenities", *cfg.num_cols, *[name for name in cfg.cat_cols if name != "city"]]
    for city_index, city in enumerate(("Boston", "Chicago")):
        target = join_uri(root, "raw", city, "listings.csv")
        content = StringIO(newline="")
        writer = csv.DictWriter(content, fieldnames=fields)
        writer.writeheader()
        for listing_id in range(1, 101):
            row = {name: 1 + (listing_id + offset) % 15 for offset, name in enumerate(cfg.num_cols)}
            row.update({name: f"demo_{listing_id % 3}" for name in cfg.cat_cols if name != "city"})
            row.update(id=listing_id, price=f"${90 + 35 * city_index + listing_id}.00",
                       currency="USD", name=f"Demo listing {listing_id}",
                       description=f"Synthetic listing from {city}", last_scraped="2025-01-01",
                       amenities='["Wifi", "Kitchen"]')
            writer.writerow(row)
        payload = content.getvalue().encode("utf-8")
        if is_s3_uri(target):
            if uri_exists(target):
                if read_bytes(target) != payload:
                    raise ValueError(f"Demo input already exists with different contents: {target}")
            else:
                write_bytes(target, payload)
        else:
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as output:
                    output.write(payload)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError(f"Demo input already exists with different contents: {path}") from None
        print(f"[Demo] {target}")


def write_demo_embeddings(data_dir: str | Path, run_id: str, cfg: ModelConfig) -> None:
    import numpy as np
    from extract_embedding import _batch_iter, _commit_parquet, _write_embeddings, _writer

    state = load_state(data_dir, required_status="prepared", run_id=run_id)
    for split in ("train", "val", "test"):
        for kind, length, omit_every in (("text", cfg.raw_text_dim, 7), ("image", cfg.raw_image_dim, 5)):
            path = join_uri(state["run_path"], "embeddings", f"{split}_{kind}.parquet")
            writer, schema, temp = _writer(path)
            total = 0
            try:
                for batch in _batch_iter(join_uri(state["run_path"], "silver", split), ["listing_key"]):
                    keys = [key for key in batch["listing_key"] if int(key.rsplit("::", 1)[1]) % omit_every]
                    if keys:
                        vectors = []
                        for key in keys:
                            digest = hashlib.blake2s(f"{kind}:{key}".encode(), digest_size=8).digest()
                            rng = np.random.default_rng(int.from_bytes(digest, byteorder="big"))
                            vectors.append(rng.standard_normal(length, dtype=np.float32))
                        _write_embeddings(writer, schema, keys, run_id, np.stack(vectors))
                        total += len(keys)
            finally:
                writer.close()
            _commit_parquet(temp, path)
            print(f"[Demo] {split}/{kind}: {total} synthetic embeddings")
    state["embedding_config"] = {"mode": "synthetic"}
    state["status"] = "embedded"
    save_state(data_dir, state, activate=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["input", "embed"])
    parser.add_argument("--data-dir", default="/data/demo")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    cfg = ModelConfig()
    if args.phase == "input":
        write_demo_inputs(args.data_dir, cfg)
    else:
        if args.run_id is None:
            parser.error("--run-id is required for the demo embedding step")
        write_demo_embeddings(args.data_dir, args.run_id, cfg)


if __name__ == "__main__":
    main()
