"""Retrain independently on the SAME fixed cohort/split; do not confuse with ablation.

Examples:
 python -m experiments.compare_modalities --mode real --epochs 15
 python -m experiments.compare_modalities --sets all7 --seeds 42,123
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CORE = ("tabular", "tabular,text", "tabular,image", "tabular,text,image")
ALL7 = ("tabular", "text", "image", "tabular,text", "tabular,image",
        "text,image", "tabular,text,image")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "mock"), default="real")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sets", choices=("core", "all7"), default="core")
    parser.add_argument("--seeds", default="42", help="Comma-separated random seeds")
    parser.add_argument("--cohort", choices=("complete", "eligible"), default="complete")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    args = parser.parse_args()
    if args.cohort != "complete":
        print("WARNING: eligible cohorts may have DIFFERENT sample counts per modality.")
    seeds = [int(v.strip()) for v in args.seeds.split(",")]
    from project_state import load_state
    run_dir = (load_state(args.data_dir, required_status="finalized")["run_path"]
               if args.mode == "real" else Path(args.data_dir))
    options = CORE if args.sets == "core" else ALL7
    records = []
    predictions = {}
    project_root = Path(__file__).resolve().parents[1]
    for seed in seeds:
        for modalities in options:
            command = [sys.executable, str(project_root / "main.py"), "--mode", args.mode,
                       "--data-dir", args.data_dir, "--modalities", modalities,
                       "--cohort", args.cohort, "--epochs", str(args.epochs),
                       "--seed", str(seed), "--batch-size", str(args.batch_size),
                       "--aux-loss-weight", str(args.aux_loss_weight),
                       "--modality-dropout", str(args.modality_dropout),
                       "--skip-importance", "--skip-ridge-baseline"]
            print("RUN:", " ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=project_root)
            name = f"{args.cohort}__{'_'.join(modalities.split(','))}__seed{seed}__aux{args.aux_loss_weight:g}"
            if (args.epochs != 20 or args.batch_size != 32
                    or args.modality_dropout != 0.2):
                name += (
                    f"__ep{args.epochs}__bs{args.batch_size}"
                    f"__md{args.modality_dropout:g}"
                )
            folder = run_dir / ("artifacts" if args.mode == "real" else "mock_artifacts") / name
            info = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            predictions[(seed, modalities)] = pd.read_csv(folder / "test_predictions.csv")
            records.append({"modalities": modalities, "seed": seed,
                            "cohort": args.cohort, "samples": info["sample_counts"],
                            "rmse_log": info["test_metrics"]["RMSE_log"],
                            "mae_log": info["test_metrics"]["MAE_log"],
                            "r2_log": info["test_metrics"]["R2_log"],
                            "results": str(folder / "results.json")})
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
    out = run_dir / ("artifacts" if args.mode == "real" else "mock_artifacts") / "comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": "Independent retraining; negative paired delta favours candidate. "
                                      "Bootstrap reflects test-listing uncertainty, not city/domain shift.",
                               "experiments": records, "paired_vs_tabular": paired},
                              indent=2), encoding="utf-8")
    print("Comparison saved:", out)
    for r in records:
        print(f"{r['modalities']:22s} seed={r['seed']} n={r['samples']['test']:4d} "
              f"RMSE_log={r['rmse_log']:.5f}")


if __name__ == "__main__":
    main()
