# Dự Án Big Data & Multimodal Deep Learning: Định Giá Căn Hộ Airbnb Đa Thành Phố

Dự án nghiên cứu và triển khai hệ thống học sâu đa phương thức (**Multimodal Deep Learning**) kết hợp xử lý dữ liệu lớn phân tán (**PySpark ETL & MLlib**) nhằm dự đoán giá niêm yết căn hộ Airbnb trên 4 thị trường trọng điểm: **New York City, Los Angeles, Hawaii và London**.

Mô hình tích hợp đồng thời 3 luồng dữ liệu (Modality):
1. **Dữ liệu dạng Bảng (Tabular):** 12 đặc trưng số + 6 đặc trưng phân loại xử lý qua **FT-Transformer** (Feature Tokenizer Transformer).
2. **Dữ liệu Văn bản (Text):** Tên, mô tả, tổng quan khu vực mã hóa qua **Multilingual-E5-base** (768 chiều).
3. **Dữ liệu Hình ảnh (Image):** Ảnh đại diện thực tế tải và mã hóa qua **CLIP-ViT-B/32** (512 chiều).
4. **Cổng Hợp nhất Thích ứng (Gated Fusion):** Cơ chế học tự động trọng số đóng góp của từng luồng dữ liệu kết hợp **Modality Dropout** chống triệt tiêu nhánh.

---

## 1. Cấu Trúc Dự Án

```
final-project-bigdata/
├── config.py                     # Cấu hình siêu tham số, phân tầng mẫu và danh mục cột
├── main.py                       # Pipeline huấn luyện, suy luận và Permutation Importance (hỗ trợ MPS/CUDA)
├── dataset.py                    # PyTorch Dataset đồng bộ 3 luồng dữ liệu từ Gold Parquet
├── evaluator.py                  # Đo lường Permutation Importance cho từng nhánh và từng cột
├── extract_embedding.py          # Trích xuất offline Embeddings (E5 + CLIP) từ URL song song đa luồng
├── diagnose_modality.py          # Ablation Study kiểm chứng mức độ đóng góp thực tế của 3 nhánh
├── evaluation_dashboard.html     # Dashboard trực quan hóa tương tác kết quả bằng Chart.js
├── DOCS_LEGAL_AND_MULTICITY.md   # Phân tích chi tiết các luật (NYC Local Law 18, LA HSO, Honolulu Bill 41, London 90-day)
├── requirements.txt              # Thư viện phụ thuộc
├── spark_etl/
│   ├── spark_session.py          # Khởi tạo SparkSession tối ưu bộ nhớ JVM
│   ├── cleaner.py                # Xử lý sạch dữ liệu, cắt trần luật 30 đêm, gán cờ nhị phân & city_name
│   ├── feature_pipeline.py       # Spark MLlib Pipeline: Imputer + StandardScaler + StringIndexer
│   └── run_etl.py                # Pipeline Bronze -> Silver -> Stratified Sampling -> Gold Parquet
└── data/
    ├── raw/                      # Thư mục chứa các file listings.csv của các thành phố
    └── gold_parquet/             # Tầng dữ liệu sạch sẵn sàng nạp cho Deep Learning (Parquet + NPY)
```

---

## 2. Hướng Dẫn Cài Đặt Môi Trường

Yêu cầu hệ thống:
* **Python:** 3.10 – 3.12
* **Java:** JDK 17 hoặc JDK 21/22 (yêu cầu để chạy Apache Spark JVM)
* **Thiết bị tăng tốc:** Hỗ trợ Apple Silicon GPU (MPS) hoặc NVIDIA CUDA (tự động fallback CPU nếu không có)

Kích hoạt môi trường ảo:
```bash
cd /Users/votan/Documents/HK7/BigData/CK/final-project-bigdata
source .venv/bin/activate
```

---

## 3. Quy Trình Chạy Toàn Bộ Hệ Thống Từ Đầu Đến Cuối

### Bước 1: Thu thập và chuẩn bị dữ liệu thô (Bronze Layer)
Đặt các file dữ liệu Inside Airbnb vào thư mục `data/raw/`:
* `Hawaii_listings.csv`
* `London_listings.csv`
* `Los_Angeles_listings.csv`
* `New_York_listings.csv`

### Bước 2: Chạy Spark ETL Pipeline (Bronze -> Silver -> Gold)
Lệnh này sẽ:
1. Đọc song song toàn bộ các file CSV thô.
2. Trích xuất tự động `city_name` từ metadata file.
3. Áp dụng quy chuẩn pháp lý: Cắt trần `minimum_nights` tại 30 ngày và tạo đặc trưng `is_long_term_stay`.
4. Thực hiện **Stratified Sampling** (lấy mẫu ngẫu nhiên 4,000 căn hộ mỗi thành phố để cân bằng dữ liệu).
5. Chuẩn hóa phân tán bằng Spark MLlib (`StandardScaler`, `StringIndexer`).
6. Xuất ra `data/gold_parquet/train.parquet` và `test.parquet`.

```bash
python -m spark_etl.run_etl
```

### Bước 3: Trích xuất Offline Embeddings cho Ảnh & Text
Tiến hành tải ảnh song song đa luồng từ URL và mã hóa đặc trưng không gian đa phương thức:
* **Text:** Mã hóa qua `intfloat/multilingual-e5-base` (768 chiều).
* **Image:** Tải song song (32 workers) và mã hóa qua `sentence-transformers/clip-ViT-B-32` (512 chiều).

```bash
python extract_embedding.py
```
*Kết quả xuất ra 4 file npy: `train_text_emb.npy`, `train_img_emb.npy`, `test_text_emb.npy`, `test_img_emb.npy`.*

### Bước 4: Huấn luyện mô hình Multimodal Deep Learning
Huấn luyện mạng nơ-ron kết hợp **FT-Transformer + Gated Fusion** và tự động tính toán Permutation Feature Importance:

```bash
python main.py
```

### Bước 5: Chạy chẩn đoán Ablation Study (Kiểm định 3 nhánh)
Xác thực mô hình có thực sự tận dụng đồng thời cả 3 nguồn thông tin hay bị liệt nhánh:

```bash
python diagnose_modality.py
```

### Bước 6: Quan sát kết quả trên Dashboard Trực quan
Mở file giao diện báo cáo tương tác trực tiếp trên trình duyệt Web:
```bash
open evaluation_dashboard.html
```

---

## 4. Bảng Kết Quả Thực Nghiệm & Đánh Giá Ý Nghĩa

| Chỉ số / Phân tích | Giá trị ghi nhận | Ý nghĩa thực tiễn |
|---|---|---|
| **Baseline Test RMSE** | ~0.3746 | Sai số dự đoán trung bình trên thang đo Log Price |
| **Trọng số Gated Fusion** | Tabular: ~29.5% \| Text: ~16.1% \| Image: ~54.4% | Cả 3 nhánh đều đóng góp trọng số thực chất, ảnh căn hộ chiếm ưu thế thị giác |
| **Ảnh hưởng của nhánh (ΔRMSE)** | Tabular: +0.4174 \| Image: +0.0850 \| Text: +0.0451 | Tabular cung cấp xương sống cấu trúc; Ảnh và Text tinh chỉnh chi tiết |
| **Top 1 Feature ảnh hưởng nhất** | `minimum_nights_capped` (ΔRMSE = 0.2091) | Phản ánh chính xác tác động của các điều luật lưu trú đô thị |
