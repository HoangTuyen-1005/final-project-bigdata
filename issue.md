# Danh sách vấn đề còn lại cần theo dõi & xử lý (Big Data & Model)

*Cập nhật sau khi hoàn tất tái cấu trúc SOA (`orchestrator-service`, `data-service`, `model-service`, `inference-service`) + MinIO S3 Lakehouse.*

---

## 1. Phần Big Data (`data-service`, Spark Cluster, Storage, Orchestration)

- **[BD-01] ~~Chưa có bước tự động đồng bộ `data/raw/` lên MinIO cho `real` mode~~** *(Đã xử lý)*
  - **Giải pháp**: `resolve_sources()` trong `services/data_service/spark_etl/run_etl.py` tự động phát hiện và upload các file `*/listings.csv[.gz]` từ `/data/raw/` lên `s3a://airbnb-lakehouse/raw/` nếu chưa có trên MinIO.

- **[BD-02] Hiệu năng đọc CSV đơn dung lượng lớn (`multiLine=true`)**
  - **Hiện trạng**: `Hawaii/listings.csv` (117.4 MB), `Los Angeles/listings.csv` (124.1 MB), `New York/listings.csv` (74.1 MB) là các file CSV đơn chưa nén, đọc với `multiLine=true` nên Spark khó chia nhỏ partition ở bước đọc đầu vào (chỉ thực sự song song tốt từ tầng Bronze Parquet trở đi).
  - **Hướng fix**: Theo dõi thời gian stage `prepare` qua `quality/prepare.json` và Spark UI (`:8080`); nếu cần, chia nhỏ hoặc dùng dữ liệu hợp lý khi chạy trên máy local 8–16 GB RAM.

- **[BD-03] Chưa chạy lại Spark Synthetic Benchmark (`benchmark.py`) trên hạ tầng mới**
  - **Hiện trạng**: Chưa có số liệu benchmark mới (wall time, physical plan, bytes written) trên cụm Spark 2 worker + MinIO S3A cho các mốc `20k` / `100k` rows.
  - **Hướng fix**: Chạy `POST /jobs/benchmark` (hoặc `docker compose run --rm data-service benchmark ...`) để lấy số liệu phục vụ báo cáo Big Data.

- **[BD-04] Trạng thái job & pipeline lưu in-memory**
  - **Hiện trạng**: `JobManager` (`contracts/jobs.py`) và `orchestrator-service` lưu `_jobs` / `_pipelines` trong RAM. Nếu container khởi động lại, lịch sử `/jobs` và `/pipelines` bị mất (dữ liệu & `state.json` trên MinIO vẫn còn).
  - **Mức độ**: Thấp (chấp nhận được với đồ án môn học; chỉ cần lưu ý không restart container giữa lúc pipeline đang chạy).

- **[BD-05] Dọn dẹp nốt dữ liệu local cũ trong `data/`**
  - **Hiện trạng**: Vẫn còn `data/demo`, `data/cache`, `data/multi_image_probe` từ trước khi chuyển sang MinIO.

---

## 2. Phần Model (`model-service`, `inference-service`, Multimodal Fusion)

- **[ML-01] Chưa có kết quả thực nghiệm trên dữ liệu thật (`real` mode)**
  - **Hiện trạng**: Hệ thống mới được kiểm chứng E2E trên `demo` mode (200 dòng synthetic, 1 epoch CPU). Chưa có checkpoint và metrics thực tế (`RMSE_log`, `MAE_log`, `R²`, per-city) từ dữ liệu Inside Airbnb thật + E5 + CLIP.
  - **Hướng fix**: Chạy hoàn chỉnh luồng `real` mode trên dữ liệu Mỹ (`USD`) trong `data/raw/`.

- **[ML-02] Chưa rebuild & chạy so sánh đa tổ hợp (`compare_modalities.py`)**
  - **Hiện trạng**: Đã sửa `compare_modalities.py`, `diagnose_modalities.py` và `services/model_service/api.py` (`/jobs/compare`, `/jobs/diagnose`) để hỗ trợ MinIO S3, nhưng **chưa rebuild lại `model-service`** sau bản sửa này và chưa chạy đối chứng `--sets core` / `all7` với `--cohort complete`, `--epochs 20`, nhiều seeds (`42,123,2026`) và `--aux-loss-weight 0.1`.
  - **Hướng fix**: Rebuild `model-service` (`docker compose up -d --build model-service`) và chạy thực nghiệm để có `comparison.json` chứng minh hiệu quả của Multimodal Fusion so với Tabular-only.

- **[ML-03] Thời gian chạy E5/CLIP & cache ảnh khi scale lên tập dữ liệu lớn** *(Đã gắn volume `image_cache`)*
  - **Hiện trạng**: Đã gắn named volume `image_cache:/tmp/image_cache` cho `model-service` trong `compose.yaml` để giữ cache ảnh qua các lần recreate container. Tuy nhiên tổng dữ liệu Mỹ trong `data/raw/` (~322 MB) vẫn có hàng chục nghìn listings nên chạy E5 + CLIP trên CPU sẽ chậm nếu không dùng `compose.gpu.yaml`.

- **[ML-04] `diagnose_modalities.py` chưa đọc `raw_profile` từ S3**
  - **Hiện trạng**: Hàm `raw_profile(state)` trong `services/model_service/experiments/diagnose_modalities.py` vẫn kiểm tra `Path(source["path"])`, nên khi `sources` trỏ tới `s3a://...` sẽ trả về `{"status": "source_csv_not_accessible"}` (phần `embedding_health` trên Gold S3 vẫn hoạt động bình thường).

- **[ML-05] `inference-service` `/predict` chỉ nhận embedding vector sẵn cho Text/Image**
  - **Hiện trạng**: Để giữ `inference-service` gọn nhẹ, `POST /predict` nhận `numeric`, `categorical`, và tùy chọn `text_embedding` (768-dim) / `image_embedding` (512-dim); nếu bỏ trống Text/Image sẽ dự đoán bằng Tabular (`available_mask=[True, False, False]`).
  - **Hướng fix (nếu cần demo UI nhập text/ảnh trực tiếp)**: Bổ sung endpoint `/encode` ở `model-service` để `orchestrator-service` gọi encode trước khi chuyển sang `inference-service`.
