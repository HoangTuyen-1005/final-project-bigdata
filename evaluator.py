"""Batched evaluation, model ablation diagnostics and permutation importance."""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import AirbnbMultimodalDataset


class MultimodalEvaluator:
    def __init__(self, model: nn.Module, test_dataset: AirbnbMultimodalDataset,
                 num_col_names: list[str], cat_col_names: list[str],
                 device: torch.device | None = None, batch_size: int = 64,
                 seed: int = 42) -> None:
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.dataset = test_dataset  # tensors stay on CPU; GPU only receives a batch
        self.num_col_names = num_col_names
        self.cat_col_names = cat_col_names
        self.batch_size = batch_size
        self.seed = seed
        self._base_metrics = None

    @torch.no_grad()
    def evaluate(self, shuffle: tuple[str, int | None, torch.Tensor] | None = None,
                 availability: tuple[bool, bool, bool] | None = None,
                 by_city: bool = False) -> tuple[dict[str, float], dict[str, float], pd.DataFrame | None]:
        ds, n = self.dataset, len(self.dataset)
        if n == 0:
            raise ValueError("Empty test dataset")
        gate_sum = torch.zeros(3, dtype=torch.float64)
        err_squared = err_absolute = sum_y = sum_y2 = 0.0
        raw_squared = raw_absolute = 0.0
        group = defaultdict(lambda: [0, 0.0, 0.0])
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch = {name: getattr(ds, name)[start:end] for name in
                     ("x_num", "x_cat", "x_txt", "x_img")}
            if shuffle is not None:
                mode, feature_idx, perm_indices = shuffle
                replace_idx = perm_indices[start:end]
                if mode == "tabular":
                    batch["x_num"] = ds.x_num[replace_idx]
                    batch["x_cat"] = ds.x_cat[replace_idx]
                elif mode in ("x_num", "x_cat"):
                    batch[mode] = batch[mode].clone()
                    batch[mode][:, feature_idx] = getattr(ds, mode)[replace_idx, feature_idx]
                elif mode in ("x_txt", "x_img"):
                    batch[mode] = getattr(ds, mode)[replace_idx]
                else:
                    raise ValueError(f"Unknown permutation mode: {mode}")
            inputs = [batch[name].to(self.device) for name in ("x_num", "x_cat", "x_txt", "x_img")]
            mask = ds.available_mask[start:end].to(self.device)
            if availability is not None:
                mask = mask & torch.tensor(availability, device=self.device,
                                           dtype=torch.bool).unsqueeze(0)
            predictions, gates = self.model(*inputs, available_mask=mask)
            predictions = predictions.detach().cpu().double().numpy()
            y = ds.y[start:end].double().numpy()
            diff = predictions - y
            err_squared += float(np.dot(diff, diff))
            err_absolute += float(np.abs(diff).sum())
            sum_y += float(y.sum())
            sum_y2 += float(np.dot(y, y))
            # log-price is the training target; raw-price metrics are computed separately.
            raw_diff = np.exp(np.clip(predictions, -20, 20)) - np.exp(np.clip(y, -20, 20))
            raw_squared += float(np.dot(raw_diff, raw_diff))
            raw_absolute += float(np.abs(raw_diff).sum())
            gate_sum += gates.detach().cpu().double().sum(dim=0)
            if by_city:
                for city, error in zip(ds.cities[start:end], diff):
                    group[city][0] += 1
                    group[city][1] += float(error * error)
                    group[city][2] += float(abs(error))
        sst = sum_y2 - sum_y * sum_y / n
        metrics = {"RMSE_log": math.sqrt(err_squared / n), "MAE_log": err_absolute / n,
                   "R2_log": 1 - err_squared / sst if sst > 1e-12 else None,
                   "RMSE_price": math.sqrt(raw_squared / n), "MAE_price": raw_absolute / n}
        mean_gates = (gate_sum / n).numpy()
        gate_summary = dict(zip(("Tabular", "Text", "Image"),
                                [round(float(x) * 100, 2) for x in mean_gates]))
        city_table = (pd.DataFrame([
            {"city": city, "count": entry[0], "RMSE_log": math.sqrt(entry[1] / entry[0]),
             "MAE_log": entry[2] / entry[0]}
            for city, entry in sorted(group.items())]) if by_city else None)
        return metrics, gate_summary, city_table

    def get_baseline_and_gate_weights(self) -> tuple[float, dict[str, float]]:
        metrics, gate, _ = self.evaluate()
        return metrics["RMSE_log"], {f"{k} Gate (%)": v for k, v in gate.items()}

    def _permutation_table(self, specs: list[tuple[str, str, int | None]],
                           n_repeats: int) -> pd.DataFrame:
        if n_repeats < 1:
            raise ValueError("n_repeats must be >= 1")
        base, _, _ = self.evaluate()
        base_rmse = base["RMSE_log"]
        gen = torch.Generator().manual_seed(self.seed)
        results = []
        for label, mode, feature in specs:
            deltas = []
            for _ in range(n_repeats):
                perm = torch.randperm(len(self.dataset), generator=gen)
                metrics, _, _ = self.evaluate(shuffle=(mode, feature, perm))
                deltas.append(metrics["RMSE_log"] - base_rmse)
            results.append({"Feature": label, "Baseline_RMSE_log": base_rmse,
                            "Delta_RMSE_log": float(np.mean(deltas)),
                            "Std_Delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0})
        return pd.DataFrame(results).sort_values("Delta_RMSE_log", ascending=False).reset_index(drop=True)

    def measure_modality_importance(self, n_repeats: int = 3) -> pd.DataFrame:
        return self._permutation_table([
            (f"Tabular ({len(self.num_col_names) + len(self.cat_col_names)} features)", "tabular", None),
            ("Text", "x_txt", None), ("Image", "x_img", None)], n_repeats)

    def measure_field_importance(self, n_repeats: int = 3) -> pd.DataFrame:
        specs = ([(name, "x_num", i) for i, name in enumerate(self.num_col_names)] +
                 [(name, "x_cat", i) for i, name in enumerate(self.cat_col_names)])
        return self._permutation_table(specs, n_repeats)

    def evaluate_ablation(self) -> pd.DataFrame:
        # Diagnostic for a *single trained model*, not separately retrained baselines.
        candidates = {"full": (True, True, True),
                      "tabular_only": (True, False, False),
                      "tabular_text": (True, True, False),
                      "tabular_image": (True, False, True)}
        rows = []
        for name, available in candidates.items():
            selected = torch.tensor(available, dtype=torch.bool)
            if not (self.dataset.available_mask & selected).any(dim=1).all():
                # This ablation is undefined for some rows (e.g. image-only
                # model asked to make a tabular-only prediction). Do not emit
                # NaN metrics that masquerade as a legitimate evaluation.
                continue
            metrics, _, _ = self.evaluate(availability=available)
            rows.append({"Configuration": name, **metrics})
        return pd.DataFrame(rows)


    @torch.no_grad()
    def prediction_frame(self) -> pd.DataFrame:
        """Paired test predictions for reproducible multimodal model comparisons."""
        ds = self.dataset
        rows = []
        self.model.eval()
        for start in range(0, len(ds), self.batch_size):
            end = min(start + self.batch_size, len(ds))
            inputs = [getattr(ds, n)[start:end].to(self.device)
                      for n in ("x_num", "x_cat", "x_txt", "x_img")]
            mask = ds.available_mask[start:end].to(self.device)
            pred, weights = self.model(*inputs, available_mask=mask)
            values = pred.detach().cpu().numpy()
            gating = weights.detach().cpu().numpy()
            target = ds.y[start:end].numpy()
            for i in range(end - start):
                rows.append({"listing_key": ds.keys[start+i], "city": ds.cities[start+i],
                             "y_log": float(target[i]), "pred_log": float(values[i]),
                             "gate_tabular": float(gating[i, 0]),
                             "gate_text": float(gating[i, 1]),
                             "gate_image": float(gating[i, 2])})
        return pd.DataFrame(rows)
