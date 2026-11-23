"""
Script chẩn đoán xem mô hình có thực sự sử dụng cả 3 nhánh không.
Chạy: python diagnose_modality.py
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ModelConfig
from dataset import AirbnbMultimodalDataset
from evaluator import MultimodalEvaluator
from models.multimodal_net import AirbnbMultimodalRegressor


def train_model(model, train_loader, cfg, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.HuberLoss()
    model.to(device)
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        for batch in train_loader:
            optimizer.zero_grad()
            preds, _ = model(batch["x_num"].to(device), batch["x_cat"].to(device),
                             batch["x_txt"].to(device), batch["x_img"].to(device))
            loss = criterion(preds, batch["y"].to(device))
            loss.backward()
            optimizer.step()


# ─── Nạp dữ liệu ───────────────────────────────────────────────
cfg = ModelConfig()
torch.manual_seed(cfg.seed)
device = torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

gold_dir = Path("data/gold_parquet")
with open(gold_dir / "metadata.json", "r", encoding="utf-8") as f:
    cat_cards = json.load(f)["cat_cardinalities"]

train_ds = AirbnbMultimodalDataset.from_spark_parquet(
    str(gold_dir / "train.parquet"),
    str(gold_dir / "train_text_emb.npy"),
    str(gold_dir / "train_img_emb.npy"),
)
test_ds = AirbnbMultimodalDataset.from_spark_parquet(
    str(gold_dir / "test.parquet"),
    str(gold_dir / "test_text_emb.npy"),
    str(gold_dir / "test_img_emb.npy"),
)
train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

# ─── Huấn luyện mô hình đầy đủ 3 nhánh ──────────────────────────
print("=" * 60)
print("KIỂM CHỨNG: Mô hình có thực sự dùng cả 3 nhánh không?")
print("=" * 60)

model_full = AirbnbMultimodalRegressor(cfg, cat_cards)
train_model(model_full, train_loader, cfg, device)
model_full.eval()

evaluator = MultimodalEvaluator(model_full, test_ds, cfg.num_cols, cfg.cat_cols, device)
rmse_full, gates = evaluator.get_baseline_and_gate_weights()

print(f"\n{'─'*60}")
print("① TRỌNG SỐ CỔNG GATING (alpha) — mô hình tự phân bổ:")
print(f"{'─'*60}")
print(f"   Tabular Gate : {gates['Tabular Gate (%)']:5.2f}%")
print(f"   Text Gate    : {gates['Text Gate (%)']:5.2f}%")
print(f"   Image Gate   : {gates['Image Gate (%)']:5.2f}%")
print(f"\n   👉 RMSE khi dùng đủ 3 nhánh: {rmse_full:.4f}")

# ─── Thí nghiệm loại bỏ từng nhánh (Ablation Study) ─────────────
print(f"\n{'─'*60}")
print("② ABLATION STUDY — tắt hẳn từng nhánh:")
print(f"{'─'*60}")
print(f"   {'Cấu hình':<35} {'RMSE':>8}  {'ΔRMSE':>8}  Nhận xét")
print(f"   {'─'*35} {'─'*8}  {'─'*8}  {'─'*20}")

@torch.no_grad()
def rmse_zero_branch(evaluator, zero_tab=False, zero_txt=False, zero_img=False):
    x_num = torch.zeros_like(evaluator.x_num) if zero_tab else evaluator.x_num
    x_cat = torch.zeros_like(evaluator.x_cat) if zero_tab else evaluator.x_cat
    x_txt = torch.zeros_like(evaluator.x_txt) if zero_txt else evaluator.x_txt
    x_img = torch.zeros_like(evaluator.x_img) if zero_img else evaluator.x_img
    preds, _ = evaluator.model(x_num, x_cat, x_txt, x_img)
    return float(torch.sqrt(torch.mean((preds - evaluator.y_true) ** 2)).item())

configs = [
    ("Đủ 3 nhánh (baseline)",          False, False, False),
    ("Tắt Tabular  (chỉ Text+Image)",   True,  False, False),
    ("Tắt Text     (chỉ Tab+Image)",    False, True,  False),
    ("Tắt Image    (chỉ Tab+Text)",     False, False, True),
    ("Chỉ Tabular  (tắt Text+Image)",   False, True,  True),
    ("Chỉ Text     (tắt Tab+Image)",    True,  False, True),
    ("Chỉ Image    (tắt Tab+Text)",     True,  True,  False),
]

results = []
for name, zt, zx, zi in configs:
    rmse = rmse_zero_branch(evaluator, zt, zx, zi)
    delta = rmse - rmse_full
    sign = "🔴 tệ hơn" if delta > 0.005 else ("🟡 tương đương" if delta > -0.005 else "🟢 tốt hơn")
    print(f"   {name:<35} {rmse:>8.4f}  {delta:>+8.4f}  {sign}")
    results.append((name, rmse, delta))

# ─── Kết luận ─────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("③ KẾT LUẬN:")
print(f"{'─'*60}")

rmse_no_tab  = [r for n,r,d in results if "Tắt Tabular"  in n][0]
rmse_no_txt  = [r for n,r,d in results if "Tắt Text"     in n][0]
rmse_no_img  = [r for n,r,d in results if "Tắt Image"    in n][0]

delta_tab = rmse_no_tab - rmse_full
delta_txt = rmse_no_txt - rmse_full
delta_img = rmse_no_img - rmse_full

THRESHOLD = 0.005  # Ngưỡng: ΔRMSE > 0.005 được coi là "có đóng góp thực sự"

used = []
if delta_tab > THRESHOLD: used.append(f"Tabular (ΔRMSE={delta_tab:+.4f})")
if delta_txt > THRESHOLD: used.append(f"Text    (ΔRMSE={delta_txt:+.4f})")
if delta_img > THRESHOLD: used.append(f"Image   (ΔRMSE={delta_img:+.4f})")

if len(used) == 3:
    print("   ✅ Mô hình sử dụng CẢ 3 nhánh — tắt bất kỳ nhánh nào đều làm RMSE tăng!")
elif len(used) == 2:
    print(f"   ⚠️  Mô hình chỉ thực sự dùng 2 nhánh: {', '.join(used)}")
    not_used = {"Tabular", "Text", "Image"} - {u.split()[0] for u in used}
    print(f"   ❌ Nhánh không đóng góp: {', '.join(not_used)}")
elif len(used) == 1:
    print(f"   ❌ Mô hình thực chất chỉ dùng 1 nhánh: {used[0]}")
else:
    print("   ❓ Không xác định được đóng góp rõ ràng.")

print(f"\n   Ngưỡng đánh giá: ΔRMSE > {THRESHOLD} = có đóng góp thực sự")
print(f"{'='*60}")

