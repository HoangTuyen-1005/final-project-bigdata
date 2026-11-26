# Checklist Kiểm Tra & Benchmark Hệ Thống Trước Khi Hoàn Thiện

Tài liệu tổng hợp các lệnh cần chạy để nghiệm thu toàn bộ hệ thống **SOA + MinIO Lakehouse + Multimodal Fusion** sau khi hoàn thiện.

---

## 1. Kiểm tra hạ tầng Docker, Gateway & GPU

- [ ] **Khởi động toàn bộ cụm 9 container**
  ```powershell
  docker compose up -d
  ```
- [ ] **Kiểm tra sức khỏe tổng thể qua API Gateway (`:8000`)**
  ```powershell
  Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 4
  ```
  *Kỳ vọng*: `status: "ok"`, cả `data-service`, `model-service`, `inference-service` đều `"ok"`.
- [ ] **Kiểm tra GPU CUDA trong `model-service` và `inference-service`**
  ```powershell
  docker compose exec model-service python -c "import torch; print('model-service:', torch.__version__, '| cuda:', torch.cuda.is_available())"
  docker compose exec inference-service python -c "import torch; print('inference-service:', torch.__version__, '| cuda:', torch.cuda.is_available())"
  ```
  *Kỳ vọng*: `cuda: True` (RTX 3060).

---

## 2. Kiểm tra Pipeline End-to-End

- [ ] **Smoke test Demo (offline, ~1 phút)**
  ```powershell
  python orchestrator.py demo --run-id check-demo
  ```
- [ ] **Chạy dữ liệu thật 5 thành phố Mỹ (`real` mode — chạy trên GPU)**
  ```powershell
  python orchestrator.py real --run-id real-final --epochs 20
  ```
  Theo dõi tiến độ trực tiếp ở tab PowerShell khác:
  ```powershell
  docker compose logs -f model-service
  ```
- [ ] **Kiểm tra báo cáo chất lượng & kết quả trên MinIO (`http://127.0.0.1:9001`)**
  - `runs/real-final/quality/prepare.json`
  - `runs/real-final/quality/finalize.json`
  - `runs/real-final/artifacts/<experiment>/results.json`
  - Hoặc in nhanh `results.json`:
    ```powershell
    docker compose exec model-service python -c "from contracts.storage import list_s3_keys, read_text; keys=[k for k in list_s3_keys('s3a://airbnb-lakehouse/runs/real-final/artifacts') if k.endswith('results.json')]; print(read_text(keys[0]))"
    ```

---

## 3. Thực nghiệm Model: Chẩn đoán & So sánh công bằng (`complete` cohort)

*(Dùng lại Gold đã tạo của `real-final`, không cần chạy lại `prepare` / `embed`.)*

- [ ] **Chạy chẩn đoán Gold (`diagnose_modalities.py`)**
  ```powershell
  $diag = '{"data_dir":"/data","run_id":"real-final","limit":5000}'
  Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8002/jobs/diagnose -ContentType 'application/json' -Body $diag
  ```
- [ ] **So sánh retrain độc lập 4 tổ hợp (`Tabular`, `Tab+Text`, `Tab+Image`, `Full`) với nhiều seeds**
  ```powershell
  $cmp = '{"mode":"real","data_dir":"/data","run_id":"real-final","sets":"core","seeds":"42,123,2026","cohort":"complete","epochs":20}'
  Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8002/jobs/compare -ContentType 'application/json' -Body $cmp
  ```
  Theo dõi job: `Invoke-RestMethod http://127.0.0.1:8002/jobs`
  Kết quả lưu tại MinIO: `runs/real-final/artifacts/comparison.json` (chứa bảng `RMSE_log`, `R²` và `paired_bootstrap_95pct_ci` so với `tabular`).

---

## 4. Kiểm tra Online Inference (`inference-service` qua Gateway)

- [ ] **Nạp checkpoint `real-final` vào `inference-service`** *(nếu chưa auto-load)*
  ```powershell
  $load = '{"data_dir":"/data","run_id":"real-final"}'
  Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8003/load -ContentType 'application/json' -Body $load
  ```
- [ ] **Test `POST /predict` qua API Gateway (`:8000`)**
  ```powershell
  $pred = '{"numeric":{"accommodates":4,"bedrooms":2,"bathrooms":1.5,"latitude":41.88,"longitude":-87.63},"categorical":{"city":"Chicago","room_type":"Entire home/apt"}}'
  Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $pred
  ```

---

## 5. Chạy Spark Synthetic Benchmark (Báo cáo Big Data)

Chạy trên cụm 2 Spark workers + MinIO S3 (`spark://spark-master:7077`), so sánh các mốc kích thước dữ liệu & partitions:

- [ ] **Mốc 20,000 rows (`partitions=2`)**
  ```powershell
  docker compose run --rm data-service benchmark --rows 20000 --partitions 2 --out /data/benchmark/20k --explain
  ```
- [ ] **Mốc 100,000 rows (`partitions=4`)**
  ```powershell
  docker compose run --rm data-service benchmark --rows 100000 --partitions 4 --out /data/benchmark/100k --explain
  ```
- [ ] **Mốc 500,000 rows (`partitions=8`, tùy chọn nếu máy đủ RAM)**
  ```powershell
  docker compose run --rm data-service benchmark --rows 500000 --partitions 8 --out /data/benchmark/500k --explain
  ```
- [ ] **Đọc lại báo cáo benchmark từ MinIO**
  ```powershell
  docker compose exec data-service python -c "from contracts.storage import read_text; print(read_text('s3a://airbnb-lakehouse/benchmark/100k/benchmark_report.json'))"
  ```

---

## 6. Unit & Regression Tests *(tùy chọn nếu máy host có cài `pytest` + `pyspark` + `torch`)*

```powershell
python -m pytest tests -q
```
