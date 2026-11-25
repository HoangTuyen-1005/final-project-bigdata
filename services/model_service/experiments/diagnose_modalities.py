"""Inspect existing runs without retraining; optional embedding health audit with pyarrow.

python -m experiments.diagnose_modalities --data-dir data
This is descriptive diagnostics, NOT evidence of causal modality contribution.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from contracts.state import load_state
from contracts.storage import join_uri, pyarrow_fs_and_path, read_text, uri_exists, write_text


def raw_profile(state: dict) -> dict:
    counts = Counter()
    descriptions, prices, urls = set(), [], set()
    for source in state.get("sources", []):
        path = Path(source["path"])
        if not path.is_file():
            # Windows paths saved with the original run aren't portable to Linux.
            path = Path("data/raw/listings.csv")
        if not path.is_file():
            continue
        import gzip
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                counts["rows"] += 1
                desc = (row.get("description") or "").strip()
                if not desc:
                    counts["empty_description"] += 1
                else:
                    descriptions.add(desc)
                url = (row.get("picture_url") or "").strip()
                if not url.startswith("https://"):
                    counts["invalid_image_url"] += 1
                else:
                    urls.add(url)
                raw_price = (row.get("price") or "").strip().replace("$", "").replace(",", "")
                try:
                    prices.append(float(raw_price))
                except ValueError:
                    counts["missing_or_invalid_price"] += 1
    if not counts["rows"]:
        return {"status": "source_csv_not_accessible"}
    counts["distinct_nonempty_descriptions"] = len(descriptions)
    counts["distinct_https_image_urls"] = len(urls)
    if prices:
        counts["price_median"] = float(np.median(prices))
    return dict(counts)


def embedding_health(run_path: Path | str, metadata: dict, limit: int) -> dict:
    """Diagnostic only: bounded sample and numerical checks; no model fitting."""
    try:
        import pyarrow.dataset as ds
    except ImportError:
        return {"status": "unavailable_pyarrow_not_installed"}
    output = {}
    for split in ("train", "val", "test"):
        path = join_uri(run_path, "gold", split)
        columns = ["x_txt", "x_img"]
        if metadata.get("has_availability_mask"):
            columns.append("available_mask")
        fs, pa_path = pyarrow_fs_and_path(path)
        scanner = ds.dataset(pa_path, filesystem=fs, format="parquet").scanner(columns=columns, batch_size=256)
        batch_data = []
        length = 0
        for batch in scanner.to_batches():
            values = batch.to_pydict()
            batch_data.append(values)
            length += len(values["x_txt"])
            if length >= limit:
                break
        if not batch_data:
            output[split] = {"status": "no_rows"}
            continue
        masks = []
        stats = {}
        for name, idx in (("x_txt", 1), ("x_img", 2)):
            vecs, present = [], []
            for batch in batch_data:
                for i, vector in enumerate(batch[name]):
                    availability = batch.get("available_mask")
                    if availability is None or availability[i][idx]:
                        vecs.append(vector)
                    present.append(availability is None or availability[i][idx])
            stats[name] = {"observed": len(vecs), "total_sampled": len(present)}
            if vecs:
                matrix = np.asarray(vecs[:limit], dtype=np.float32)
                stats[name].update({
                    "embedding_dim": int(matrix.shape[1]),
                    "norm_median": float(np.median(np.linalg.norm(matrix, axis=1))),
                    "mean_coordinate_std": float(np.mean(matrix.std(axis=0))),
                    "has_nan_inf": bool(not np.isfinite(matrix).all()),
                })
        output[split] = stats
    return output


def run(data_dir: str, limit: int = 5000, *, run_id: str | None = None) -> dict:
    state = load_state(data_dir, required_status="finalized", run_id=run_id)
    root = state["run_path"]
    metadata = json.loads(read_text(join_uri(root, "gold/metadata.json"), encoding="utf-8"))
    summary = {"run_id": state["run_id"], "split_counts": metadata["counts"],
               "gold_has_availability_mask": metadata.get("has_availability_mask", False),
               "raw_profile": raw_profile(state),
               "embedding_health": embedding_health(root, metadata, limit)}
    old_uri = join_uri(root, "artifacts/results.json")
    if uri_exists(old_uri):
        result = json.loads(read_text(old_uri, encoding="utf-8"))
        metrics = {r["Configuration"]: r for r in result["ablation_diagnostic"]}
        base = metrics["full"]["RMSE_log"]
        summary["legacy_experiment"] = {
            "mean_gates_percent": result["mean_gate_weights_percent"],
            "full_test_rmse_log": base,
            "ablation_delta_rmse_vs_full": {
                name: round(row["RMSE_log"] - base, 8) for name, row in metrics.items()},
            "modality_permutation_delta_rmse": {
                row["Feature"]: row["Delta_RMSE_log"]
                for row in result.get("modality_permutation", [])},
        }
    summary["interpretation"] = (
        "Tiny gates/permutation effects show the fitted model scarcely uses the channel. "
        "They do NOT prove whether image/text lack predictive signal. Retrain each "
        "modality combination on the same complete-case cohort with multiple seeds.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-id", help="Inspect the specified finalized run.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = run(args.data_dir, args.limit, run_id=args.run_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        write_text(args.out, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
