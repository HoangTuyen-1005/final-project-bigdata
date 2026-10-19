import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from dataset import AirbnbMultimodalDataset
from evaluator import MultimodalEvaluator
from models.multimodal_net import AirbnbMultimodalRegressor


def create_mock_airbnb_data(
    cfg: ModelConfig, n_samples: int = 2000
) -> tuple[AirbnbMultimodalDataset, AirbnbMultimodalDataset, list[int]]:
    """Tạo dữ liệu giả lập có quy luật nhân quả thực tế để kiểm chứng pipeline."""
    rng = np.random.default_rng(cfg.seed)

    # 1. Sinh dữ liệu Tabular
    x_num = rng.normal(0, 1, size=(n_samples, len(cfg.num_cols))).astype(np.float32)
    cat_cardinalities = [25, 10, 4, 2, 2]  # Số nhóm của 5 cột Categorical
    x_cat = np.column_stack(
        [rng.integers(0, card, size=n_samples) for card in cat_cardinalities]
    )

    # 2. Sinh Embeddings cho Text (768 chiều) và Image (512 chiều)
    x_txt = rng.normal(0, 1, size=(n_samples, cfg.raw_text_dim)).astype(np.float32)
    x_img = rng.normal(0, 1, size=(n_samples, cfg.raw_image_dim)).astype(np.float32)

    # 3. Tạo Target y = log(price) phụ thuộc vào cả 3 luồng để kiểm định Evaluator:
    # - Cột số 2 (accommodates) và cột số 3 (bedrooms) tác động mạnh nhất
    # - Cột phân loại 2 (room_type) tác động mạnh
    # - Nhánh Text và Image đều đóng góp tín hiệu thật
    tabular_signal = 0.45 * x_num[:, 2] + 0.35 * x_num[:, 3] + 0.25 * (x_cat[:, 2] == 0)
    text_signal = 0.25 * np.mean(x_txt[:, :20], axis=1)
    image_signal = 0.20 * np.mean(x_img[:, :15], axis=1)
    noise = rng.normal(0, 0.08, size=n_samples)

    y = (4.5 + tabular_signal + text_signal + image_signal + noise).astype(np.float32)

    # Chia Train (80%) / Test (20%)
    split = int(n_samples * 0.8)
    train_ds = AirbnbMultimodalDataset(
        x_num[:split], x_cat[:split], x_txt[:split], x_img[:split], y[:split]
    )
    test_ds = AirbnbMultimodalDataset(
        x_num[split:], x_cat[split:], x_txt[split:], x_img[split:], y[split:]
    )
    return train_ds, test_ds, cat_cardinalities


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: ModelConfig,
    device: torch.device | None = None,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.HuberLoss()

    model.to(device)
    model.train()

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds, _ = model(
                batch["x_num"].to(device),
                batch["x_cat"].to(device),
                batch["x_txt"].to(device),
                batch["x_img"].to(device),
            )
            loss = criterion(preds, batch["y"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch["y"])

        if epoch % 5 == 0 or epoch == 1:
            print(f"[Epoch {epoch:02d}/{cfg.epochs:02d}] Train Huber Loss: {epoch_loss / len(train_loader.dataset):.4f}")


def main() -> None:
    cfg = ModelConfig()
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"-> [DEVICE] Khởi động trên GPU: [{gpu_name}] ({vram_gb:.1f} GB VRAM)")
    else:
        print("-> [DEVICE] Không tìm thấy GPU CUDA, tự động fallback sang CPU.")

    gold_dir = Path("data/gold_parquet")
    if (gold_dir / "train.parquet").exists() and (gold_dir / "train_img_emb.npy").exists():
        print("1. Phát hiện dữ liệu Gold từ PySpark! Đang nạp Parquet & NPY Embeddings...")
        with open(gold_dir / "metadata.json", "r", encoding="utf-8") as f:
            cat_cards = json.load(f)["cat_cardinalities"]

        train_ds = AirbnbMultimodalDataset.from_spark_parquet(
            parquet_path=str(gold_dir / "train.parquet"),
            text_emb_npy_path=str(gold_dir / "train_text_emb.npy"),
            image_emb_npy_path=str(gold_dir / "train_img_emb.npy"),
        )
        test_ds = AirbnbMultimodalDataset.from_spark_parquet(
            parquet_path=str(gold_dir / "test.parquet"),
            text_emb_npy_path=str(gold_dir / "test_text_emb.npy"),
            image_emb_npy_path=str(gold_dir / "test_img_emb.npy"),
        )
    else:
        print("1. Chưa có dữ liệu Parquet, khởi tạo Mock Data để kiểm thử...")
        train_ds, test_ds, cat_cards = create_mock_airbnb_data(cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    model = AirbnbMultimodalRegressor(cfg, cat_cards)
    train_model(model, train_loader, cfg, device)

    print("\n2. Đánh giá bằng MultimodalEvaluator trên tập Test...")
    evaluator = MultimodalEvaluator(
        model=model,
        test_dataset=test_ds,
        num_col_names=cfg.num_cols,
        cat_col_names=cfg.cat_cols,
        device=device,
    )

    base_rmse, gate_weights = evaluator.get_baseline_and_gate_weights()
    print(f"\n-> Baseline Test RMSE: {base_rmse:.4f}")
    print(f"-> Trọng số Cổng Gated Fusion trung bình: {gate_weights}")

    print("\n--- KẾT QUẢ Ý 3: MỨC ĐỘ ẢNH HƯỞNG CỦA TỪNG NHÁNH (MODALITY PERMUTATION) ---")
    df_modality = evaluator.measure_modality_importance()
    print(df_modality.to_string(index=False))

    print("\n--- KẾT QUẢ Ý 2: TOP 8 FIELD ẢNH HƯỞNG MẠNH NHẤT LÊN GIÁ (FIELD PERMUTATION) ---")
    df_fields = evaluator.measure_field_importance()
    print(df_fields.head(8).to_string(index=False))


if __name__ == "__main__":
    main()