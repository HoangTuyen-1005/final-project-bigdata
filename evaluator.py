import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import AirbnbMultimodalDataset


class MultimodalEvaluator:
    """Bộ công cụ đánh giá Permutation Importance cho từng Field và từng Modality."""

    def __init__(
        self,
        model: nn.Module,
        test_dataset: AirbnbMultimodalDataset,
        num_col_names: list[str],
        cat_col_names: list[str],
        device: torch.device | None = None,
    ) -> None:
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.num_col_names = num_col_names
        self.cat_col_names = cat_col_names

        # Đưa toàn bộ tập Test lên Tensor để chạy hoán vị siêu tốc
        self.x_num = test_dataset.x_num.to(device)
        self.x_cat = test_dataset.x_cat.to(device)
        self.x_txt = test_dataset.x_txt.to(device)
        self.x_img = test_dataset.x_img.to(device)
        self.y_true = test_dataset.y.to(device)

    @torch.no_grad()
    def _compute_rmse(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_txt: torch.Tensor,
        x_img: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        preds, gates = self.model(x_num, x_cat, x_txt, x_img)
        mse = torch.mean((preds - self.y_true) ** 2)
        return float(torch.sqrt(mse).item()), gates

    def get_baseline_and_gate_weights(self) -> tuple[float, dict[str, float]]:
        """Tính RMSE gốc và trung bình trọng số cổng Alpha của 3 nhánh."""
        base_rmse, gates = self._compute_rmse(
            self.x_num, self.x_cat, self.x_txt, self.x_img
        )
        mean_gates = gates.mean(dim=0).cpu().numpy()
        gate_summary = {
            "Tabular Gate (%)": round(float(mean_gates[0]) * 100, 2),
            "Text Gate (%)": round(float(mean_gates[1]) * 100, 2),
            "Image Gate (%)": round(float(mean_gates[2]) * 100, 2),
        }
        return base_rmse, gate_summary

    def measure_modality_importance(self, n_repeats: int = 5) -> pd.DataFrame:
        """Ý 3: Kiểm tra mức độ ảnh hưởng của cả nhánh (Tabular vs Text vs Image)."""
        base_rmse, _ = self.get_baseline_and_gate_weights()
        n_samples = self.y_true.size(0)
        results = []

        for modality in ["Tabular (All 18 cols)", "Text Branch", "Image Branch"]:
            deltas = []
            for _ in range(n_repeats):
                perm_idx = torch.randperm(n_samples, device=self.device)

                if modality == "Tabular (All 18 cols)":
                    shuffled_rmse, _ = self._compute_rmse(
                        self.x_num[perm_idx], self.x_cat[perm_idx], self.x_txt, self.x_img
                    )
                elif modality == "Text Branch":
                    shuffled_rmse, _ = self._compute_rmse(
                        self.x_num, self.x_cat, self.x_txt[perm_idx], self.x_img
                    )
                else:
                    shuffled_rmse, _ = self._compute_rmse(
                        self.x_num, self.x_cat, self.x_txt, self.x_img[perm_idx]
                    )
                deltas.append(shuffled_rmse - base_rmse)

            results.append(
                {
                    "Modality": modality,
                    "Baseline_RMSE": round(base_rmse, 4),
                    "Shuffled_RMSE": round(base_rmse + float(np.mean(deltas)), 4),
                    "Delta_RMSE (Importance)": round(float(np.mean(deltas)), 4),
                }
            )

        return pd.DataFrame(results).sort_values(
            by="Delta_RMSE (Importance)", ascending=False
        )

    def measure_field_importance(self, n_repeats: int = 3) -> pd.DataFrame:
        """Ý 2: Đo mức độ ảnh hưởng của từng Field lẻ lên Target (thay cho Pearson)."""
        base_rmse, _ = self.get_baseline_and_gate_weights()
        n_samples = self.y_true.size(0)
        records = []

        # 1. Đo từng cột Numeric
        for col_idx, col_name in enumerate(self.num_col_names):
            deltas = []
            for _ in range(n_repeats):
                x_num_copy = self.x_num.clone()
                perm_idx = torch.randperm(n_samples, device=self.device)
                x_num_copy[:, col_idx] = x_num_copy[perm_idx, col_idx]

                rmse, _ = self._compute_rmse(x_num_copy, self.x_cat, self.x_txt, self.x_img)
                deltas.append(rmse - base_rmse)

            records.append(
                {
                    "Field_Name": col_name,
                    "Type": "Numeric (Tabular)",
                    "Delta_RMSE": round(float(np.mean(deltas)), 4),
                }
            )

        # 2. Đo từng cột Categorical
        for col_idx, col_name in enumerate(self.cat_col_names):
            deltas = []
            for _ in range(n_repeats):
                x_cat_copy = self.x_cat.clone()
                perm_idx = torch.randperm(n_samples, device=self.device)
                x_cat_copy[:, col_idx] = x_cat_copy[perm_idx, col_idx]

                rmse, _ = self._compute_rmse(self.x_num, x_cat_copy, self.x_txt, self.x_img)
                deltas.append(rmse - base_rmse)

            records.append(
                {
                    "Field_Name": col_name,
                    "Type": "Categorical (Tabular)",
                    "Delta_RMSE": round(float(np.mean(deltas)), 4),
                }
            )

        df_fields = pd.DataFrame(records).sort_values(by="Delta_RMSE", ascending=False)
        return df_fields.reset_index(drop=True)