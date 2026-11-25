# Airbnb Multimodal + PySpark Big Data — SOA & Lakehouse Architecture

Dự án Big Data môn học: dự đoán `ln(price)` cho Airbnb listings (đơn vị `USD`) từ **Tabular + Text (multilingual E5) + Image (CLIP)** bằng FT-Transformer và gated fusion, triển khai theo **kiến trúc SOA (Service-Oriented Architecture) + Data Lakehouse (MinIO S3 + Spark Standalone)** chạy 100% cục bộ bằng Docker Compose.

**Tài liệu phân tích phương pháp:** [`document/PHAN_TICH_VA_NANG_CAP.md`](document/PHAN_TICH_VA_NANG_CAP.md) · **Hướng dẫn chạy chi tiết:** [`document/HUONG_DAN_CHAY_VI.md`](document/HUONG_DAN_CHAY_VI.md)

---

## 1. Kiến trúc SOA & Cấu trúc thư mục

```text
Client / orchestrator.py
        │  HTTP REST (Control Plane)
        ▼
orchestrator-service (:8000) ───► POST /predict ───► inference-service (:8003)
        │                                                   ▲
        ├──► data-service (:8001)  ──► Spark Cluster        │ load best_model.pt
        │       (prepare / finalize / benchmark)            │ & metadata.json
        └──► model-service (:8002)                          │
                (embed / train / compare / demo-*)          │
                        │                                   │
                        └────────► MinIO S3 (:9000) ◄───────┘
                          (Data Plane: raw, bronze, silver,
                           embeddings, gold, artifacts)
```

```text
├── contracts/                      # Contract dùng chung: S3 storage, state, job runner
├── services/
│   ├── orchestrator_service/       # API Gateway & Control Plane (:8000)
│   ├── data_service/               # Spark ETL, MLlib & Benchmark (:8001, spark_etl/)
│   ├── model_service/              # E5/CLIP & PyTorch Training (:8002, main.py, models/, ...)
│   └── inference_service/          # Online Serving API (:8003, /predict, /load)
├── orchestrator.py                 # CLI client điều phối qua Gateway (:8000)
├── compose.yaml                    # Hạ tầng 9 container (MinIO, Spark Cluster, 4 SOA services)
├── compose.gpu.yaml                # Cấu hình bổ sung NVIDIA GPU cho model-service
└── tests/                          # Unit & integration tests
```

---

## 2. Các service & cổng truy cập

| Service | Vai trò | Cổng / UI |
|---|---|---|
| `orchestrator-service` | API Gateway & điều phối toàn bộ pipeline | `http://127.0.0.1:8000/docs` |
| `data-service` | Spark driver xử lý `prepare`, `finalize`, `benchmark` | `http://127.0.0.1:8001/docs` |
| `model-service` | Chạy `embed`, `train`, `compare`, `diagnose`, `demo-*` | `http://127.0.0.1:8002/docs` |
| `inference-service` | Nạp `best_model.pt` từ MinIO và phục vụ `POST /predict` | `http://127.0.0.1:8003/docs` |
| `minio` (+ `minio-init`) | Object Storage S3 (`airbnb-lakehouse` bucket) | API `:9000` · Console `http://127.0.0.1:9001` |
| `spark-master` + 2 `spark-worker` | Cụm Spark Standalone (mỗi worker 1 core, 1024m RAM) | UI `http://127.0.0.1:8080` |

---

## 3. Khởi chạy & Demo nhanh với Docker

Yêu cầu **Docker Desktop / Docker Engine (Compose v2)** với tối thiểu ~8 GB RAM khả dụng.

### Bước 1: Build và khởi động toàn bộ hệ thống SOA
```bash
docker compose up -d --build
```

### Bước 2: Chạy pipeline Demo End-to-End qua Gateway
```bash
python orchestrator.py demo --run-id demo-s3
```
Lệnh sẽ gọi `http://127.0.0.1:8000/pipelines/run` để thực thi tuần tự:
`demo-input` → `prepare` (Spark Bronze/Silver) → `demo-embed` → `finalize` (Spark Gold + `available_mask` + `tabular_preprocessor`) → `train` (PyTorch FT-Transformer) → `load-inference` → test `POST /predict`.

Dữ liệu và hiện vật của run nằm trên MinIO bucket `airbnb-lakehouse` tại `demo/runs/<run_id>/`.

### Bước 3: Thử dự đoán giá trực tuyến qua API Gateway
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"numeric":{"accommodates":4,"bedrooms":2,"bathrooms":1.5},"categorical":{"city":"Boston","room_type":"Entire home/apt"}}'
```

Để dừng toàn bộ cụm:
```bash
docker compose down
```

---

## 4. Chạy dữ liệu thật (`real` mode)

1. Đặt các tệp **Detailed Listings** của Inside Airbnb (giá `USD`) vào `data/raw/<city>/listings.csv` hoặc `listings.csv.gz`, ví dụ:
   ```text
   data/raw/
   ├── Boston/listings.csv.gz
   └── Chicago/listings.csv
   ```
   *(Thư mục `data/raw` được mount read-only vào `data-service` tại `/data/raw`.)*

2. Chạy toàn bộ pipeline qua Gateway (lần đầu `model-service` sẽ tải model E5 và CLIP vào volume `hf_cache`):
   ```bash
   python orchestrator.py real --epochs 20
   ```

3. Nếu máy có NVIDIA GPU (Linux hoặc Windows WSL2), bật CUDA bằng:
   ```powershell
   $env:COMPOSE_FILE = "compose.yaml;compose.gpu.yaml"
   docker compose up -d --build
   ```

---

## 5. Benchmark Spark & So sánh 7 tổ hợp Modality

- **Spark Benchmark (dữ liệu tổng hợp, đo riêng throughput/execution plan của Spark ETL)**:
  ```bash
  docker compose run --rm data-service benchmark --rows 20000 --partitions 2 --out /tmp/benchmark/20k --explain
  ```
- **Chẩn đoán & So sánh các tổ hợp modality (`core` / `all7`)**:
  ```bash
  docker compose run --rm model-service diagnose --data-dir /data --out /tmp/diagnosis.json
  docker compose run --rm model-service compare --data-dir /data --run-id <run_id> --sets core --epochs 20
  ```

---

## 6. Ghi chú thực nghiệm & kiểm thử
- `PySpark` xử lý ETL phân tán thật (Bronze partition theo `city`, Window dedup, deterministic hash split, LEFT JOIN giữ missing modality, MLlib fit **chỉ trên Train**).
- Extraction (E5/CLIP) và PyTorch training tách khỏi Spark để tối ưu RAM/VRAM; dữ liệu lớn (Parquet, embeddings, checkpoints) trao đổi qua MinIO S3, còn điều phối và dự đoán online giao tiếp qua HTTP REST API.
- Bộ unit test nằm trong `tests/` (cấu hình `pythonpath` trong [pytest.ini](pytest.ini)), chạy khi môi trường có `pytest` + `pyspark` + `torch`: `python -m pytest tests -q`.
