"""Paired bootstrap of already-saved Tabular vs Full predictions on one fixed Test cohort.

Usage (from project root):
    python paired_bootstrap_chicago.py --data-dir data

Does NOT train or modify any model checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from project_state import load_state

SEEDS = (42, 123, 2026)
NEEDED = ("listing_key", "y_log", "pred_log")


def read_predictions(artifact_root: Path, modality: str, seed: int) -> pd.DataFrame:
    path = artifact_root / f"complete__{modality}__seed{seed}__aux0" / "test_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path, dtype={"listing_key": "string"})
    missing = set(NEEDED) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    if df["listing_key"].isna().any() or df["listing_key"].duplicated().any():
        raise ValueError(f"Missing or duplicated listing_key in {path}")
    if not np.isfinite(df[["y_log", "pred_log"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite predictions/targets in {path}")
    return df.loc[:, list(NEEDED)].sort_values("listing_key").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    state = load_state(args.data_dir, required_status="finalized")
    artifacts = state["run_path"] / "artifacts"
    y_ref = None
    keys_ref = None
    per_seed = []
    err_by_seed = []
    for seed in SEEDS:
        tab = read_predictions(artifacts, "tabular", seed)
        full = read_predictions(artifacts, "tabular_text_image", seed)
        if tab["listing_key"].tolist() != full["listing_key"].tolist():
            raise ValueError(f"Test keys differ between models for seed {seed}")
        y = tab["y_log"].to_numpy(dtype=float)
        if not np.allclose(y, full["y_log"].to_numpy(dtype=float), rtol=0, atol=1e-8):
            raise ValueError(f"Targets differ between models for seed {seed}")
        if keys_ref is None:
            keys_ref, y_ref = tab["listing_key"].tolist(), y
        elif keys_ref != tab["listing_key"].tolist() or not np.allclose(y_ref, y, rtol=0, atol=1e-8):
            raise ValueError(f"Test keys/targets differ across seeds; cannot jointly bootstrap")
        e_tab = np.square(tab["pred_log"].to_numpy(dtype=float) - y)
        e_full = np.square(full["pred_log"].to_numpy(dtype=float) - y)
        rmse_tab = float(np.sqrt(e_tab.mean()))
        rmse_full = float(np.sqrt(e_full.mean()))
        per_seed.append({"seed": seed, "n_test": len(y), "rmse_tabular": rmse_tab,
                         "rmse_full": rmse_full, "delta_rmse_full_minus_tabular": rmse_full - rmse_tab})
        err_by_seed.append((e_tab, e_full))

    n = len(keys_ref)
    rng = np.random.default_rng(2026)
    b = args.bootstrap
    boot_delta = np.empty((len(SEEDS), b), dtype=float)
    # Resample the SAME test listings across all seed comparisons.
    for j in range(b):
        ix = rng.integers(0, n, n)
        for k, (e_tab, e_full) in enumerate(err_by_seed):
            boot_delta[k, j] = np.sqrt(e_full[ix].mean()) - np.sqrt(e_tab[ix].mean())

    for k, r in enumerate(per_seed):
        r["bootstrap_95pct_ci"] = [float(q) for q in np.percentile(boot_delta[k], [2.5, 97.5])]
    mean_delta = float(np.mean([r["delta_rmse_full_minus_tabular"] for r in per_seed]))
    mean_ci = [float(q) for q in np.percentile(boot_delta.mean(axis=0), [2.5, 97.5])]
    result = {
        "run_id": state["run_id"],
        "method": "listing-level paired bootstrap, fixed trained seeds (42,123,2026)",
        "n_test": n, "bootstrap_replicates": b, "per_seed": per_seed,
        "mean_delta_rmse_full_minus_tabular": mean_delta,
        "mean_delta_bootstrap_95pct_ci": mean_ci,
        "interpretation": "Negative delta favors Full; CI excludes zero suggests a difference over resampled Test listings, conditional on these trained models. This does not assess training-seed or host/domain shift uncertainty."
    }
    out = artifacts / "paired_bootstrap_tabular_vs_full.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Run: {state['run_id']} | Matched Test listings: {n}")
    print("Delta = RMSE(Full) - RMSE(Tabular); negative favors Full")
    for r in per_seed:
        lo, hi = r["bootstrap_95pct_ci"]
        print(f"Seed {r['seed']}: Tabular={r['rmse_tabular']:.5f}, Full={r['rmse_full']:.5f}, "
              f"delta={r['delta_rmse_full_minus_tabular']:+.5f}, 95% CI=[{lo:+.5f}, {hi:+.5f}]")
    print(f"Mean seed delta={mean_delta:+.5f}; paired 95% CI over Test listings="
          f"[{mean_ci[0]:+.5f}, {mean_ci[1]:+.5f}]")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
