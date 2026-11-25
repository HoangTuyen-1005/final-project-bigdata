"""Retrain independently on the SAME fixed cohort/split; do not confuse with ablation.

Examples:
 python -m experiments.compare_modalities --mode real --epochs 15
 python -m experiments.compare_modalities --sets all7 --seeds 42,123
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import ModelConfig
from contracts.state import experiment_dir_name, load_state, resolve_data_root
from contracts.storage import join_uri, read_bytes, read_text, write_text

CORE = ("tabular", "tabular,text", "tabular,image", "tabular,text,image")
ALL7 = ("tabular", "text", "image", "tabular,text", "tabular,image",
        "text,image", "tabular,text,image")


def main() -> None:
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "mock"), default="real")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-id", help="Compare models on the specified finalized run.")
    parser.add_argument("--sets", choices=("core", "all7"), default="core")
    parser.add_argument("--seeds", default=str(defaults.seed), help="Comma-separated random seeds")
    parser.add_argument("--cohort", choices=("complete", "eligible"), default="complete")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--aux-loss-weight", type=float, default=defaults.auxiliary_loss_weight)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--modality-dropout", type=float, default=defaults.modality_dropout_prob)
    args = parser.parse_args()
    if args.run_id is not None and args.mode == "mock":
        parser.error("--run-id is only valid when reading the Gold pipeline")
    if args.cohort != "complete":
        print("WARNING: eligible cohorts may have DIFFERENT sample counts per modality.")
    seeds = [int(v.strip()) for v in args.seeds.split(",")]
    root = resolve_data_root(args.data_dir)
    run_args = []
    if args.mode == "real":
        state = load_state(args.data_dir, required_status="finalized", run_id=args.run_id)
        artifact_root = join_uri(state["run_path"], "artifacts")
        run_args = ["--run-id", state["run_id"]]
    else:
        artifact_root = join_uri(root, "mock_artifacts")
    cfg = replace(defaults, epochs=args.epochs, batch_size=args.batch_size,
                  auxiliary_loss_weight=args.aux_loss_weight,
                  modality_dropout_prob=args.modality_dropout)
    options = CORE if args.sets == "core" else ALL7
    records = []
    predictions = {}
    project_root = Path(__file__).resolve().parents[1]
    for seed in seeds:
        seed_cfg = replace(cfg, seed=seed)
        for modalities in options:
            command = [sys.executable, str(project_root / "main.py"), "--mode", args.mode,
                       "--data-dir", str(args.data_dir), "--modalities", modalities,
                       "--cohort", args.cohort, "--epochs", str(seed_cfg.epochs),
                       "--seed", str(seed_cfg.seed), "--batch-size", str(seed_cfg.batch_size),
                       "--aux-loss-weight", str(seed_cfg.auxiliary_loss_weight),
                       "--modality-dropout", str(seed_cfg.modality_dropout_prob),
                       "--skip-importance", "--skip-ridge-baseline", *run_args]
            print("RUN:", " ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=project_root)
            name = experiment_dir_name(seed_cfg, args.cohort, modalities.split(","))
            folder = join_uri(artifact_root, name)
            results_uri = join_uri(folder, "results.json")
            info = json.loads(read_text(results_uri, encoding="utf-8"))
            predictions[(seed, modalities)] = pd.read_csv(
                io.BytesIO(read_bytes(join_uri(folder, "test_predictions.csv")))
            )
            records.append({"modalities": modalities, "seed": seed,
                            "cohort": args.cohort, "samples": info["sample_counts"],
                            "rmse_log": info["test_metrics"]["RMSE_log"],
                            "mae_log": info["test_metrics"]["MAE_log"],
                            "r2_log": info["test_metrics"]["R2_log"],
                            "results": str(results_uri)})
    # Paired bootstrap over listing IDs (not over correlated training seeds).
    # Fixed cohort and exactly matched keys are essential for fair comparisons.
    paired = []
    if args.cohort == "complete":
        rng = np.random.default_rng(2026)
        for seed in seeds:
            baseline = predictions[(seed, "tabular")].sort_values("listing_key")
            for modalities in options:
                if modalities == "tabular":
                    continue
                candidate = predictions[(seed, modalities)].sort_values("listing_key")
                if (baseline["listing_key"].tolist() != candidate["listing_key"].tolist()
                        or not np.allclose(baseline["y_log"], candidate["y_log"])):
                    raise ValueError("Not a paired experiment: listing keys/targets differ")
                y = baseline["y_log"].to_numpy(dtype=np.float64)
                e_base = (baseline["pred_log"].to_numpy() - y) ** 2
                e_cand = (candidate["pred_log"].to_numpy() - y) ** 2
                delta = float(np.sqrt(e_cand.mean()) - np.sqrt(e_base.mean()))
                bootstrap = []
                for _ in range(1000):
                    indices = rng.integers(0, len(y), len(y))
                    bootstrap.append(np.sqrt(e_cand[indices].mean()) -
                                     np.sqrt(e_base[indices].mean()))
                paired.append({"seed": seed, "candidate": modalities,
                               "reference": "tabular", "n_test": len(y),
                               "delta_rmse_log": delta,
                               "paired_bootstrap_95pct_ci": [float(v) for v in
                                   np.percentile(bootstrap, [2.5, 97.5])]})
    out = join_uri(artifact_root, "comparison.json")
    write_text(
        out,
        json.dumps(
            {
                "note": "Independent retraining; negative paired delta favours candidate. "
                        "Bootstrap reflects test-listing uncertainty, not city/domain shift.",
                "experiments": records,
                "paired_vs_tabular": paired,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Comparison saved:", out)
    for r in records:
        print(f"{r['modalities']:22s} seed={r['seed']} n={r['samples']['test']:4d} "
              f"RMSE_log={r['rmse_log']:.5f}")


if __name__ == "__main__":
    main()
