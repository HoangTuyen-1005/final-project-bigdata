# Danh sách vấn đề còn lại cần theo dõi & xử lý (Big Data & Model)

*Cập nhật sau khi hoàn tất tái cấu trúc SOA (`orchestrator-service`, `data-service`, `model-service`, `inference-service`) + MinIO S3 Lakehouse.*

---

## 1. Phần Big Data (`data-service`, Spark Cluster, Storage, Orchestration)

- **[BD-01] ~~Chưa có bước tự động đồng bộ `data/raw/` lên MinIO cho `real` mode~~** *(Đã xử lý)*
  - **Giải pháp**: `resolve_sources()` trong `services/data_service/spark_etl/run_etl.py` tự động phát hiện và upload các file `*/listings.csv[.gz]` từ `/data/raw/` lên `s3a://airbnb-lakehouse/raw/` nếu chưa có trên MinIO.

- **[BD-02] ~~Hiệu năng đọc CSV đơn dung lượng lớn (`multiLine=true`)~~** *(Đã xử lý)*
  - **Giải pháp**: `resolve_sources()` tự động chia các file CSV ≥ 16 MB (`Hawaii`, `Los Angeles`, `New York`) bằng `csv.reader` thành nhiều part `.csv.gz` (10,000 dòng/part) trên `s3a://airbnb-lakehouse/raw/<city>/parts/` để Spark đọc song song trên các worker ngay từ `Stage 0`.

- **[BD-03] ~~`benchmark.py` chưa hỗ trợ ghi `s3a://` (MinIO) & chưa chạy lại benchmark trên hạ tầng mới~~** *(Đã xử lý & chạy kiểm chứng)*
  - **Giải pháp**: Đã cập nhật `benchmark.py` và `contracts/storage.py` (`parquet_size_bytes`) để ghi trực tiếp lên MinIO `s3a://`, chạy thành công mốc `20k` rows trên `spark://spark-master:7077` (`bronze: 11.592s`, `silver: 7.084s`, `gold: 8.765s`; report lưu tại `s3a://airbnb-lakehouse/benchmark/20k/benchmark_report.json`).

- **[BD-04] ~~Trạng thái job & pipeline lưu in-memory~~** *(Đã xử lý)*
  - **Giải pháp**: `JobManager` (`contracts/jobs.py`) và `orchestrator-service` (`services/orchestrator_service/api.py`) tự động lưu lịch sử job/pipeline dưới dạng JSON lên MinIO S3 (`_service_jobs/`, `_pipelines/`) và nạp lại khi container khởi động (đánh dấu `interrupted` nếu job đang chạy dở lúc tắt container).

- **[BD-05] ~~Dọn dẹp nốt dữ liệu local cũ trong `data/`~~** *(Đã xử lý)*
  - **Giải pháp**: Đã xóa `data/demo`, `data/cache`, `data/multi_image_probe`; chỉ giữ lại `data/raw/`.

---

## 2. Phần Model (`model-service`, `inference-service`, Multimodal Fusion)

- **[ML-01] ~~Chưa có kết quả thực nghiệm trên dữ liệu thật (`real` mode)~~** *(Đã chạy xong `real-us-v2`)*
  - **Kết quả**: Đã chạy thành công trên `121,673` dòng thô (`98,580` dòng Gold từ 5 thành phố Mỹ), đạt `RMSE_log = 0.3582`, `R²_log = 0.8371` (`best_epoch = 12`), gate weights cân bằng (`Tabular 38.63%`, `Text 35.78%`, `Image 25.59%`).

- **[ML-02] ~~Chưa rebuild & chạy so sánh đa tổ hợp (`compare_modalities.py`)~~** *(Đã hoàn thành trên `real-us-v2`)*
  - **Kết quả** (`complete` cohort: `77,171 train / 9,672 val / 9,648 test`, `seed=42`, `epochs=20`, GPU CUDA):
    - `tabular`: `RMSE_log = 0.38129`, `R² = 0.81640`
    - `tabular,text`: `RMSE_log = 0.36550`, `R² = 0.83129` (`ΔRMSE = -0.01579`, 95% CI `[-0.02021, -0.01153]`)
    - `tabular,image`: `RMSE_log = 0.40089`, `R² = 0.79704` (`ΔRMSE = +0.01960`, 95% CI `[+0.01576, +0.02367]`)
    - `tabular,text,image` (Full): `RMSE_log = 0.35787`, `R² = 0.83826` (`ΔRMSE = -0.02342`, 95% CI `[-0.02812, -0.01906]`)
  - Báo cáo lưu tại `s3a://airbnb-lakehouse/runs/real-us-v2/artifacts/comparison.json`.

- **[ML-03] ~~Thời gian chạy E5/CLIP & cache ảnh khi scale lên tập dữ liệu lớn~~** *(Đã xử lý)*
  - **Giải pháp**: Đã gắn named volume `image_cache:/tmp/image_cache` cho `model-service` trong `compose.yaml` và cấu hình mặc định PyTorch CUDA (`cu126`) trên RTX 3060 GPU cho `model-service` & `inference-service`.

- **[ML-04] ~~`diagnose_modalities.py` chưa đọc `raw_profile` từ S3~~** *(Đã xử lý)*
  - **Giải pháp**: Đã cập nhật `raw_profile()` đọc trực tiếp `.csv` / `.csv.gz` từ `s3a://` qua `contracts.storage.read_bytes`.

- **[ML-05] ~~`inference-service` `/predict` chỉ nhận embedding vector sẵn cho Text/Image~~** *(Đã xử lý)*
  - **Giải pháp**: Đã thêm `POST /encode` trên `model-service` (E5 + CLIP trên GPU) và cập nhật `POST /predict` trên `orchestrator-service` tự động gọi `model-service/encode` khi client truyền `text` hoặc `image_url` thô rồi chuyển tiếp sang `inference-service`.
