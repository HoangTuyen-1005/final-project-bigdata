from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AirbnbMultimodalDataset(Dataset):
    """Dataset quản lý đồng bộ 3 luồng dữ liệu: Tabular (Num + Cat), Text, Image."""

    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        x_txt: np.ndarray,
        x_img: np.ndarray,
        y: np.ndarray,
    ) -> None:
        assert len(x_num) == len(x_cat) == len(x_txt) == len(x_img) == len(y), (
            f"Số lượng mẫu giữa các nhánh không khớp nhau! "
            f"Tab={len(x_num)}, Txt={len(x_txt)}, Img={len(x_img)}, y={len(y)}"
        )
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_txt = torch.tensor(x_txt, dtype=torch.float32)
        self.x_img = torch.tensor(x_img, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx],
            "x_txt": self.x_txt[idx],
            "x_img": self.x_img[idx],
            "y": self.y[idx],
        }

    @classmethod
    def from_spark_parquet(
        cls,
        parquet_path: str,
        text_emb_npy_path: str,
        image_emb_npy_path: str,
    ) -> AirbnbMultimodalDataset:
        """Đọc trực tiếp file Parquet từ tầng Gold của PySpark và khởi tạo Dataset."""
        df = pd.read_parquet(parquet_path)

        # Chuyển cột Array trong Parquet thành ma trận 2D NumPy
        x_num = np.vstack(df["x_num"].to_numpy()).astype(np.float32)
        x_cat = np.vstack(df["x_cat"].to_numpy()).astype(np.int64)
        y = df["y"].to_numpy().astype(np.float32)

        # Load pre-computed embeddings tương ứng với thứ tự dòng trong Parquet
        x_txt = np.load(text_emb_npy_path).astype(np.float32)
        x_img = np.load(image_emb_npy_path).astype(np.float32)

        return cls(x_num=x_num, x_cat=x_cat, x_txt=x_txt, x_img=x_img, y=y)