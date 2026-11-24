# Airbnb Multimodal + PySpark Big Data — Pipeline & Experiment Upgrade

Dự án Big Data môn học: dự đoán `ln(price)` cho Airbnb listings từ **Tabular + Text (multilingual E5) + Image (CLIP)** bằng FT-Transformer và gated fusion. Phiên bản này có **bảy tổ hợp modality, missing-modality masks, thí nghiệm retrain độc lập, lakehouse Bronze/Silver/Gold có persistence, QA và synthetic Spark benchmark**.

**Bắt đầu đọc:** [`document/PHAN_TICH_VA_NANG_CAP.md`](document/PHAN_TICH_VA_NANG_CAP.md) (phát hiện, căn cứ và phương pháp thực nghiệm). Dữ liệu thực, mô hình checkpoint và báo cáo cũ được giữ nguyên. **Cần chạy lại prepare → embed → finalize để tạo Gold mới có availability mask.** Có thể dùng Gold cũ để chạy thí nghiệm trên complete cases vì loader hỗ trợ schema cũ.

## Data flow

```text
Inside Airbnb detailed listings.csv[.gz] x N cities
        │
  Spark PREPARE
        ├── Bronze Parquet (original fields + source provenance; partition by city)
        ├── Spark SQL clean, deduplicate latest, deterministic hash split
        ├── Silver Parquet (train / val / test; preserve missing text and image)
        └── quality/prepare.json (counts, null rates, snapshots, times)
        │
  OFFLINE MODEL INFERENCE (resource-constrained, batched)
        ├── multilingual E5 -> text embeddings Parquet, keyed by listing_key
        └── CLIP -> image embeddings Parquet, keyed by listing_key
        │
  Spark FINALIZE
        ├── validate run_id, keys, dimensions, numerical finiteness
        ├── LEFT JOIN from Silver; available_mask=[tab,txt,img]
        ├── Train-fitted Spark MLlib (Imputer / Scaler / StringIndexer)
        ├── Gold Parquet (train / val / test, NumPy-compatible arrays)
        └── quality/finalize.json (presence coverage, rows, timings)
        │
  PyTorch FT-Transformer + projections + gated fusion
        ├── 1 / 2 / 3 modalities; symmetric dropout; optional aux losses
        └── independent experiments + paired test comparisons
```

- PySpark là pipeline xử lý dữ liệu **thật**: transformation phân tán, shuffle/dedup, Parquet, typed joins, MLlib và quản trị chất lượng. Không phải chỉ mở SparkSession cho có.
- Extraction model và PyTorch training **chạy tách khỏi Spark** để tiết kiệm tài nguyên GPU; **không** claim distributed training hay throughput production.
- Bronze/Silver/Gold là batch pipeline có lịch sử theo `run_id`, **không** tự nhận là streaming production. Spark benchmark dữ liệu tổng hợp nằm riêng, **không** dùng để train model hay báo cáo accuracy.

## Environment

- Python 3.11–3.12, Java 17, PySpark >=3.5, PyArrow, PyTorch, sentence-transformers, Pillow, requests.
- 16 GB RAM, GPU 6 GB: Spark `local[2]`, write partitions 2–4, E5 batch 16, CLIP batch 8; E5 và CLIP được load tuần tự. Windows + winutils hoặc WSL2.
- `pip install -r requirements.txt` và `pip install -r requirements-dev.txt` (kiểm thử).
- Kiểm tra: `python -m pytest tests -q`; Spark tests yêu cầu Java + PySpark.

## Real data: 4 stages

1. **Prepare:**

```bash
python -m spark_etl.run_etl prepare --data-dir data --write-partitions 2 --explain \
  --source "Albany=data/raw/listings.csv"
```

Có thể lặp `--source "CITY=data/raw/city/listings.csv.gz"` để gộp nhiều thành phố. Dùng đúng **detailed listings**, không phải summary CSV. Các thành phố nên dùng snapshot gần nhau và cùng tiền tệ (dự án chưa chuẩn hóa tỷ giá). Không nhập cả snapshot lặp lại rồi xem như các listing độc lập.

2. **Embedding:**

```bash
python extract_embedding.py --data-dir data --workers 4
```

Cần Internet truy cập ảnh và Hugging Face (có thể có ảnh không tải được); chỉ lưu embedding có ảnh/text thực. Không tải cả hai encoder lên GPU cùng lúc. File đặt trong `data/runs/<run_id>/embeddings/` và có khóa + run_id.

3. **Finalize:**

```bash
python -m spark_etl.run_etl finalize --data-dir data --write-partitions 2 --explain
```

Kết quả Gold có `x_num`, `x_cat`, `x_txt`, `x_img`, `available_mask`, `y`; missing embeddings được lưu vector zero **và phải có mask False** để không được hiểu nhầm là embedding thực. Không inner join loại các listing mất một modality. Fit MLlib trên Train **chỉ**.

4. **Train / evaluate một mô hình:**

```bash
python main.py --mode real --modalities tabular,text,image --cohort complete --epochs 20
python main.py --mode real --modalities tabular,text --cohort complete --epochs 20
python main.py --mode real --modalities text,image --cohort complete --epochs 20
python main.py --mode real --modalities tabular --cohort eligible --epochs 20
```

`--cohort complete` bảo đảm cả train/val/test đều dùng danh sách complete-case để **so sánh công bằng** giữa các tổ hợp; `--cohort eligible` giữ mọi listing có ít nhất một modality trong tổ hợp, nhưng **mẫu quan sát có thể khác nhau**, không được so sánh metric trực tiếp nếu cohort khác nhau.

`--aux-loss-weight 0.1` thêm Huber loss của mỗi nhánh quan sát được, giúp thử nghiệm chống collapse mà không ép gate weights bằng nhau. `--modality-dropout 0.2` áp dụng dropout **đối xứng**. Đây là hyperparameters cần chọn bằng Validation, **không** mặc định tốt hơn.

## Diagnose và retrain 7 tổ hợp

```bash
python -m experiments.diagnose_modalities --data-dir data --out document/diagnosis.json
python -m experiments.compare_modalities --mode real --sets core --seeds 42,123,2026 --epochs 20
python -m experiments.compare_modalities --mode real --sets all7 --seeds 42 --epochs 20 --aux-loss-weight 0.1
```

- `core` retrain Tab, Tab+Text, Tab+Image, Tab+Text+Image từ đầu; `all7` thêm Text, Image, Text+Image.
- `comparison.json` chứa metrics, test sizes, paired bootstrap (95% CI theo listing IDs) so với Tabular. Mỗi model cũng lưu `test_predictions.csv` keyed để kiểm tra; CI không thay thế repeated seeds hay đánh giá city-out-of-domain.
- Gate weights, ablation same-model và permutation importance là **diagnostic**; không được diễn giải như tỷ lệ đóng góp nhân quả.

## Benchmark riêng để chứng minh xử lý Big Data

```bash
python -m spark_etl.benchmark --rows 20000 --partitions 2 --out data/benchmark/20k --explain
python -m spark_etl.benchmark --rows 100000 --partitions 4 --out data/benchmark/100k --explain
```

Benchmark tạo dữ liệu **tổng hợp để thử đường ống xử lý**: event duplicates, window dedup, partitioned Parquet, Spark shuffle/groupBy, keyed left joins, đánh giá availability, report wall time/disk bytes/execution plan. **Không** tạo ra dữ liệu Airbnb mới, và **không** chạy E5/CLIP hay training trên dữ liệu giả benchmark. So sánh số liệu benchmark theo số hàng và partition; ghi rõ cấu hình máy, Java/PySpark và nguồn tài nguyên.

## Caveats & reproducibility

- Hiện vật cũ trong `data/runs/.../artifacts/results.json` và checkpoints **không phải kết quả của code mới**. Hãy retrain để có metrics của bản nâng cấp.
- Data chỉ 500 dòng của Albany; không chứng minh khả năng scale hàng triệu listing của encoder hoặc GPU training. Spark stress test độc lập để minh họa khả năng xử lý dữ liệu.
- Kỹ thuật masking và aux loss không bảo đảm ba modality luôn hữu ích. Nếu model Tabular-only vẫn tốt nhất ở nhiều seeds/dataset, nên báo cáo kết quả đó và giới hạn dữ liệu, không ép gates bằng nhau.
- Lưu ý: `document/explain.md` là tài liệu lịch sử từ trước khi cập nhật. **README mới + source** là nguồn hướng dẫn hiện hành.
