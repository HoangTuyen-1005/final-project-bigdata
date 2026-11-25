"""PyTorch dataset reading keyed, Spark-joined Gold Parquet.

Never aligns independently produced NumPy arrays by row position. Joining and
validating the modalities is the responsibility of Spark finalize.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from contracts.storage import read_parquet_df


class AirbnbMultimodalDataset(Dataset):
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray,
                 x_txt: np.ndarray, x_img: np.ndarray, y: np.ndarray,
                 keys: list[str] | None = None, cities: list[str] | None = None,
                 available_mask: np.ndarray | None = None) -> None:
        n = len(y)
        if n == 0 or any(len(array) != n for array in (x_num, x_cat, x_txt, x_img)):
            raise ValueError("All modalities must have the same nonzero sample count")
        for name, arr in (("x_num", x_num), ("x_txt", x_txt), ("x_img", x_img), ("y", y)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} contains NaN/Inf")
        if np.any(np.asarray(x_cat) < 0):
            raise ValueError("x_cat contains a negative category index")
        if keys is not None and (len(keys) != n or len(set(keys)) != n):
            raise ValueError("Gold Parquet has duplicate or missing listing_key")
        if cities is not None and len(cities) != n:
            raise ValueError("City metadata does not align with samples")
        self.keys = keys or [f"mock_{i}" for i in range(n)]
        self.cities = cities or ["mock"] * n
        self.x_num = torch.from_numpy(np.ascontiguousarray(x_num, dtype=np.float32))
        self.x_cat = torch.from_numpy(np.ascontiguousarray(x_cat, dtype=np.int64))
        self.x_txt = torch.from_numpy(np.ascontiguousarray(x_txt, dtype=np.float32))
        self.x_img = torch.from_numpy(np.ascontiguousarray(x_img, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        availability = (np.ones((n, 3), dtype=np.bool_) if available_mask is None
                        else np.asarray(available_mask, dtype=np.bool_))
        if availability.shape != (n, 3) or not availability.any(axis=1).all():
            raise ValueError("available_mask must be (N,3) with at least one channel per row")
        self.available_mask = torch.from_numpy(np.ascontiguousarray(availability))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x_num": self.x_num[idx], "x_cat": self.x_cat[idx],
                "x_txt": self.x_txt[idx], "x_img": self.x_img[idx],
                "available_mask": self.available_mask[idx], "y": self.y[idx]}

    @classmethod
    def from_gold_parquet(cls, parquet_path: str, metadata: dict) -> "AirbnbMultimodalDataset":
        cols = ["listing_key", "city", "x_num", "x_cat", "x_txt", "x_img", "y"]
        if metadata.get("has_availability_mask", False):
            cols.append("available_mask")
        df = read_parquet_df(parquet_path, columns=cols)
        if df.empty:
            raise ValueError("Empty Gold dataset: " + parquet_path)
        matrix = {}
        for col, dtype in (("x_num", np.float32), ("x_cat", np.int64),
                           ("x_txt", np.float32), ("x_img", np.float32)):
            matrix[col] = np.stack(df[col].to_numpy()).astype(dtype)
        expected_dims = {"x_num": len(metadata["num_feature_names"]),
                         "x_cat": len(metadata["cat_cardinalities"]),
                         "x_txt": metadata["text_dim"], "x_img": metadata["image_dim"]}
        for col, expected in expected_dims.items():
            if matrix[col].ndim != 2 or matrix[col].shape[1] != expected:
                raise ValueError(f"{col}: expected width {expected}, got {matrix[col].shape}")
        for idx, cardinality in enumerate(metadata["cat_cardinalities"]):
            if (matrix["x_cat"][:, idx] >= cardinality).any():
                raise ValueError(f"Categorical column {idx} exceeds trained vocabulary size")
        return cls(matrix["x_num"], matrix["x_cat"], matrix["x_txt"], matrix["x_img"],
                   df["y"].to_numpy(dtype=np.float32),
                   df["listing_key"].astype(str).tolist(), df["city"].astype(str).tolist(),
                   np.stack(df["available_mask"].to_numpy()) if "available_mask" in df else None)


    def select_modalities(self, enabled: tuple[bool, bool, bool],
                          cohort: str = "complete") -> "AirbnbMultimodalDataset":
        """Return eligible rows with selected channels masked, never falsely available.

        complete: same three-modality cohort for ALL comparison experiments.
        eligible: all rows with >=1 selected channel (not directly comparable across runs).
        """
        if cohort not in ("complete", "eligible"):
            raise ValueError("cohort must be 'complete' or 'eligible'")
        selected = np.asarray(enabled, dtype=np.bool_)
        if selected.shape != (3,) or not selected.any():
            raise ValueError("Select at least one of tabular,text,image")
        original = self.available_mask.numpy()
        retain = original.all(axis=1) if cohort == "complete" else (original & selected).any(axis=1)
        if not retain.any():
            raise ValueError(f"No {cohort} samples eligible for selected modalities {enabled}")
        combined = original[retain] & selected
        indices = np.flatnonzero(retain)
        return AirbnbMultimodalDataset(
            self.x_num[indices].numpy(), self.x_cat[indices].numpy(),
            self.x_txt[indices].numpy(), self.x_img[indices].numpy(),
            self.y[indices].numpy(), [self.keys[i] for i in indices],
            [self.cities[i] for i in indices], combined)
