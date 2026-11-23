# [BÁO CÁO KỸ THUẬT] PHÂN TÍCH LUỒNG XỬ LÝ & KIẾN TRÚC BIG DATA

## DỰ ÁN: ĐỊNH GIÁ BẤT ĐỘNG SẢN AIRBNB ĐA PHƯƠNG THỨC & ĐA THÀNH PHỐ

**Người lập:** Senior Data Engineer / Big Data Architect
**Mục đích:** Tài liệu kỹ thuật chia sẻ nội bộ nhóm (Sử dụng đồng bộ trên Google Docs)
**Phiên bản codebase:** 2.0 (Multi-City: New York, Los Angeles, Hawaii, London)

---

# PHẦN 1: TỔNG QUAN & LUỒNG XỬ LÝ DỮ LIỆU (DATA PIPELINE FLOW)

Hệ thống được thiết kế theo mô hình **Medallion Architecture (Bronze -> Silver -> Gold)** kết hợp **Decoupled Multimodal Pipeline**, tách bạch hoàn toàn giữa công đoạn xử lý phân tán dữ liệu bảng (Distributed Tabular ETL) và công đoạn trích xuất đặc trưng phi cấu trúc (Multimodal Offline Embedding).

## 1.1 Sơ đồ khối tổng quan luồng dữ liệu (Data Pipeline Architecture)

```
[ NGUỒN DỮ LIỆU THÔ (RAW) ]
  • Hawaii_listings.csv (~117 MB)
  • London_listings.csv (~215 MB)
  • Los_Angeles_listings.csv (~124 MB)
  • New_York_listings.csv (~74 MB)
  Tổng: ~530 MB | ~300,000 listings thô | Định dạng MultiLine CSV
        |
        v [TẦNG BRONZE: Ingestion]
  spark_etl/cleaner.py: AirbnbSparkCleaner.load_raw_listings()
  - Native Spark CSV Reader (multiLine=True, inferSchema=False)
        |
        v [TẦNG SILVER: Transformation & Enrichment]
  spark_etl/cleaner.py: AirbnbSparkCleaner.clean_and_extract_features()
  - Bóc tách city_name từ metadata input_file_name()
  - Làm sạch giá: regexp_replace, cast double, filter [$10, $5000], tạo log(price)
  - Phân tích regex bathrooms_text ("1.5 baths" -> 1.5, "Half-bath" -> 0.5)
  - Đếm tiện nghi amenities_count từ JSON array
  - Chuẩn hóa luật lưu trú: minimum_nights_capped (clip [1, 30]), cờ is_long_term_stay
  - Làm sạch văn bản mô tả (gộp name, description, overview; xóa HTML)
        |
        v [STRATIFIED SAMPLING]
  spark_etl/run_etl.py: Window.partitionBy("city_name").orderBy(rand())
  - Lấy mẫu phân tầng cân bằng: Tối đa 4,000 listings / thành phố
  - Tổng dữ liệu sau lấy mẫu: ~16,000 records
        |
        +-----------------------------------+
        |                                   |
        v (80% Train: ~12,800)              v (20% Test: ~3,200)
  train_raw                            test_raw
        |                                   |
        +-----------------+-----------------+
                          |
                          v [TẦNG GOLD: Spark MLlib Pipeline]
  spark_etl/feature_pipeline.py: SparkTabularFeaturizer
  - Pipeline FIT DUY NHẤT trên train_raw, TRANSFORM cho cả Train và Test:
    + Imputer (Median): Điền khuyết 12 cột số
    + VectorAssembler + StandardScaler: Chuẩn hóa Z-score (mean=0, std=1)
    + StringIndexer (handleInvalid="keep"): Mã hóa 6 cột Categorical
  - Trích xuất: x_num (vector_to_array), x_cat (array long), y (log_price)
  - Xuất bản phẩm lưu trữ (Storage):
    + data/gold_parquet/train.parquet (~12,800 rows)
    + data/gold_parquet/test.parquet (~3,200 rows)
    + data/gold_parquet/metadata.json (cat_cardinalities)
    + data/gold_parquet/spark_mllib_pipeline/ (fitted model artifact)
        |
        +-----------------------------------+
        |                                   |
        v                                   v
  [ NHÁNH OFFLINE EMBEDDINGS ]         [ NHÁNH METADATA & TABULAR ]
  extract_embedding.py                 dataset.py: AirbnbMultimodalDataset
  - Đọc Parquet lấy clean_full_text     - Đọc Parquet lấy x_num, x_cat, y
  - Đọc Parquet lấy picture_url         - Khóa chặt Row Alignment (Index khớp 100%)
  - Multilingual-E5-base (GPU/CPU):
    Vector hóa text -> 768 chiều (L2 norm)
  - ThreadPoolExecutor (32 workers):
    Tải ảnh song song, timeout 5s
    CLIP ViT-B/32 mã hóa -> 512 chiều (L2 norm)
    Fallback: Zero-vector cho link lỗi
  - Lưu trữ:
    + train_text_emb.npy (38 MB)
    + train_img_emb.npy (25 MB)
    + test_text_emb.npy (9.3 MB)
    + test_img_emb.npy (6.2 MB)
        |                                   |
        +-----------------+-----------------+
                          |
                          v [TẦNG MODEL SERVING & TRAINING]
  main.py -> models/multimodal_net.py
  - FT-Transformer: Biến đổi x_num và x_cat qua FeatureTokenizer & CLS Token (128 chiều)
  - Projection Layers: Chiếu Text (768 -> 128) và Image (512 -> 128)
  - GatedMultimodalFusion: Tự động học trọng số cổng alpha_tab, alpha_txt, alpha_img
  - Modality Dropout (20%): Chống hiện tượng liệt nhánh
  - Head Regressor: Dự đoán log(price)
        |
        v [TẦNG ĐÁNH GIÁ (EVALUATION)]
  evaluator.py & diagnose_modality.py
  - Ý 2 (Field Importance): Đo Delta RMSE từng thuộc tính (Top: minimum_nights_capped)
  - Ý 3 (Modality Importance): Đo đóng góp từng nhánh bằng Permutation & Ablation Study
  - evaluation_dashboard.html: Trực quan hóa trực quan qua giao diện Chart.js
```

## 1.2 Phân tích vai trò và logic xử lý cốt lõi của từng module

### A. Tầng Khởi tạo Spark Session (`spark_etl/spark_session.py`)

- **Vai trò:** Cấu hình môi trường runtime phân tán JVM.
- **Điểm chốt:** Khóa cứng trình thực thi Python qua `PYSPARK_PYTHON = sys.executable` (tránh xung đột môi trường ảo trên macOS/Linux), tự động nhận diện `winutils.exe` nếu chạy trên Windows, kích hoạt tính năng **Adaptive Query Execution (AQE)** và tối ưu hóa **Apache Arrow** để tăng tốc I/O giữa PySpark và Pandas/NumPy.

### B. Tầng Làm sạch dữ liệu (`spark_etl/cleaner.py`)

- **Vai trò:** Chuyển đổi dữ liệu từ Bronze (thô) sang Silver (sạch), 100% sử dụng **Native Spark SQL Functions** (tránh dùng Python UDF gây nghẽn serialization).
- **Logic xử lý nghiệp vụ quan trọng:**
  1. `city_name`: Dùng `F.input_file_name()` lấy tên file thô mà Spark executor đang đọc, phân loại thành phố ngay tại tầng executor.
  2. `price`: Loại bỏ ký tự `$` và dấu phẩy `,`, ép sang `DoubleType`, loại bỏ mẫu rác (`price < $10` hoặc `> $5,000`), tạo mục tiêu `log_price`.
  3. `bathrooms`: Dùng JVM Regex bóc tách chữ (`"1.5 baths"` -> `1.5`, `"Half-bath"` -> `0.5`).
  4. `minimum_nights_capped` & `is_long_term_stay`: Áp dụng Domain Knowledge luật lưu trú. Giới hạn `minimum_nights` trong ngưỡng `[1, 30]` đêm để triệt tiêu ngoại lai cực đoan (365, 1125 đêm), đồng thời tạo cờ nhị phân `is_long_term_stay = 1.0` nếu số đêm gốc `>= 30`.

### C. Tầng Chuẩn hóa & Mã hóa Phân tán (`spark_etl/feature_pipeline.py`)

- **Vai trò:** Xây dựng Spark MLlib Pipeline chuẩn hóa tập trung, triệt tiêu nguy cơ **Data Leakage (Rò rỉ dữ liệu)**.
- **Logic cốt lõi:**
  - `Imputer(strategy="median")`: Điền khuyết các giá trị số bằng trung vị tính trên tập Train.
  - `StandardScaler(withMean=True, withStd=True)`: Chuẩn hóa Z-score về trung bình 0, phương sai 1 (bắt buộc cho FT-Transformer).
  - `StringIndexer(handleInvalid="keep")`: Mã hóa cột phân loại sang chỉ số `0, 1, 2...`, gán nhãn mới chưa từng thấy vào bucket riêng.
  - **Quy tắc vàng:** Chỉ gọi `.fit(train_raw)` để học tham số phân phối từ tập Train, sau đó dùng PipelineModel này để `.transform()` cho cả Train và Test.

### D. Tầng Trích xuất Embedding Đa luồng (`extract_embedding.py`)

- **Vai trò:** Xử lý dữ liệu phi cấu trúc ngoại tuyến (Offline Processing).
- **Logic cốt lõi:**
  - Đọc trực tiếp từ file Parquet của tầng Gold nhằm giữ đúng thứ tự dòng (Row Alignment).
  - Sử dụng `ThreadPoolExecutor(32 workers)` với `requests.Session` tái sử dụng kết nối TCP để tải ảnh song song theo từng batch 512 ảnh.
  - Áp dụng cơ chế **Zero-vector Fallback**: Nếu ảnh chết (404/timeout), tự động bù vector `0` kích thước 512 chiều, đảm bảo ma trận không bị lệch shape.
  - Vector văn bản trích xuất bằng mô hình đa ngôn ngữ `intfloat/multilingual-e5-base` (768 chiều), vector ảnh bằng `clip-ViT-B-32` (512 chiều), toàn bộ được chuẩn hóa L2 norm trước khi lưu `.npy`.

### E. Tầng Mô hình hóa & Hợp nhất Đa phương thức (`models/` & `main.py`)

- **Vai trò:** Huấn luyện mạng học sâu Multimodal Regressor.
- **Logic cốt lõi:**
  - `FTTransformerEncoder`: Thay vì dùng mạng MLP thông thường, mỗi đặc trưng dạng bảng được chiếu thành 1 Token 128 chiều kèm token `[CLS]` để học mối quan hệ tương hỗ giữa các cột thông qua Multi-Head Attention.
  - `GatedMultimodalFusion`: Học động 3 trọng số cổng $\alpha_{tab}, \alpha_{txt}, \alpha_{img}$ thông qua mạng Softmax sao cho tổng bằng 1.0.
  - `Modality Dropout (20%)`: Che ngẫu nhiên nhánh Bảng hoặc Văn bản trong quá trình train để ép mô hình phải học đặc trưng thẩm mỹ từ nhánh Ảnh, tránh hiện tượng sập nhánh (Modality Collapse).

---

# PHẦN 2: PHÂN TÍCH LÝ DO THIẾT KẾ (ARCHITECTURAL RATIONALE)

## 2.1 Tại sao chọn kiến trúc Medallion (Bronze - Silver - Gold)?

- **Tính module hóa và khả năng tái lập (Reproducibility):** Khi cần tinh chỉnh mô hình học sâu hoặc thử nghiệm kiến trúc mạng mới, kỹ sư không cần chạy lại toàn bộ quá trình ETL Spark tốn kém. Tầng Gold (Parquet + NPY) đóng vai trò là "Single Source of Truth" bất biến cho giai đoạn AI.
- **Tiết kiệm tài nguyên:** Việc phân tầng cho phép debug độc lập: nếu lỗi xảy ra ở bước làm sạch văn bản, chỉ cần fix logic tại tầng Silver mà không ảnh hưởng tới logic tính toán của MLlib Pipeline.

## 2.2 Tại sao tách biệt Offline Embeddings thay vì trích xuất Online trong PyTorch DataLoader?

- **Tránh nút thắt I/O mạng trong Training Loop:** Tải 16,000 bức ảnh qua giao thức HTTP mất từ 30 đến 45 phút. Nếu trích xuất online trong mỗi epoch huấn luyện, một lượt train 20 epochs sẽ tiêu tốn hơn **15 giờ** chỉ để chờ I/O mạng.
- **Tối ưu GPU Compute:** Bằng cách pre-compute toàn bộ embeddings ra đĩa cứng dưới định dạng nhị phân `.npy`, thời gian nạp dữ liệu vào GPU/MPS chỉ tính bằng mili-giây qua memory-map, giúp quá trình huấn luyện 20 epochs hoàn tất trong **chưa đầy 3 phút**.

## 2.3 Tại sao sử dụng Spark MLlib Pipeline thay vì Scikit-Learn?

- **Khả năng mở rộng quy mô lớn (Scalability):** Scikit-Learn yêu cầu gom toàn bộ dữ liệu vào bộ nhớ RAM của một máy đơn (Driver). Khi dữ liệu tăng từ 500 MB lên 50 GB hoặc 500 GB, Scikit-Learn sẽ gặp lỗi `MemoryError / Out-Of-Memory (OOM)`. Spark MLlib thực hiện tính toán Mean, Std, Median trên kiến trúc phân tán (MapReduce/RDD aggregates) của toàn cụm.
- **Tính đóng gói (Portability):** Model pipeline được lưu thành thư mục chuẩn `spark_mllib_pipeline/`, có thể nạp lại trực tiếp trên cụm Production (Databricks, AWS EMR) để phục vụ chấm điểm batch (Batch Scoring) cho hàng triệu listings mới.

## 2.4 Ý đồ kỹ thuật đằng sau FT-Transformer và Gated Fusion

- Dữ liệu dạng bảng (Tabular) trong định giá bất động sản có tính không đồng nhất cao (heterogeneous). Các nghiên cứu SOTA chỉ ra FT-Transformer vượt trội hơn hẳn ResNet/MLP trong việc nắm bắt tương tác phi tuyến tính (ví dụ: sự kết hợp giữa `city_name = New York` VÀ `minimum_nights_capped = 30` mang ý nghĩa hoàn toàn khác với ở `Hawaii`).
- Cơ chế **Gated Fusion** mang lại khả năng diễn giải định lượng (Interpretability): Giúp kỹ sư và chuyên viên phân tích nghiệp vụ trả lời chính xác mô hình đang tin cậy vào nguồn thông tin nào nhiều nhất (Thực tế: Ảnh chiếm ~57%, Bảng chiếm ~24%, Văn bản chiếm ~18%).

---

# PHẦN 3: ĐÁNH GIÁ MỨC ĐỘ "CHUẨN BIG DATA" (BIG DATA READINESS AUDIT)

## 3.1 Bảng đánh giá 5 trụ cột Big Data

| Trụ cột đánh giá                                    | Trạng thái hiện tại        | Đánh giá kỹ thuật chi tiết                                                                                                                                                            |
| -------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Distributed Processing (Xử lý phân tán)** | **ĐẠT (8.5/10)**       | Sử dụng 100% Native Spark SQL functions, VectorAssembler, StandardScaler phân tán. Không dùng Python UDF làm tắc nghẽn serialization JVM.                                          |
| **2. Shuffle & Partition Optimization**            | **KHÁ (7.0/10)**        | Đã kích hoạt Spark AQE (`adaptive.enabled=true`, `coalescePartitions=true`). Tuy nhiên logic phân vùng dữ liệu sau khi lấy mẫu chưa được repartition tối ưu theo node. |
| **3. Memory Management (Quản lý RAM)**           | **TRUNG BÌNH (6.5/10)** | Có ý thức dùng `.cache()` và `.unpersist()`, nhưng tồn tại hiện tượng cache chồng lấn gây lãng phí bộ nhớ lưu trữ executor.                                         |
| **4. Scalability (Khả năng mở rộng)**          | **KHÁ (7.5/10)**        | Tầng ETL scale tốt trên cụm. Tuy nhiên tầng trích xuất Embedding và DataLoader huấn luyện vẫn phụ thuộc vào bộ nhớ cục bộ của Driver.                                   |
| **5. Fault Tolerance & Resilience**                | **YẾU (5.0/10)**        | Chưa có cơ chế Checkpoint/Resume khi tải ảnh ngoại tuyến. Quá trình tải ảnh 45 phút nếu gặp sự cố mạng ở phút 40 sẽ phải chạy lại từ đầu.                        |

---

## 3.2 Phân tích các nút thắt cổ chai (Bottlenecks) nếu dữ liệu phình to gấp 100 lần (30 triệu dòng / ~50 GB)

Nếu hệ thống được cấp phát dữ liệu quy mô toàn cầu (hàng chục triệu listings), các điểm yếu sau trong mã nguồn sẽ gây sự cố crash hệ thống:

### 🔴 Bottleneck 1: Hiện tượng Double Cache gây cạn kiệt bộ nhớ Executor RAM

- **Vị trí code:** `spark_etl/run_etl.py` (Dòng 72 và Dòng 91)
- **Vấn đề:**
  ```python
  silver_df.cache()
  total_valid = silver_df.count() # Action 1: Cache toàn bộ 30 triệu dòng vào Storage Memory
  ...
  silver_df = silver_df.withColumn(...).filter(...) # Lấy mẫu
  silver_df.cache() # Cache tiếp tập đã lấy mẫu mà CHƯA giải phóng bộ nhớ của DataFrame cũ!
  sampled_total = silver_df.count()
  ```
- **Hậu quả khi scale 100x:** 30 triệu dòng ban đầu chiếm dụng toàn bộ Storage Memory của cụm Spark. Lệnh `.cache()` thứ hai sẽ kích hoạt cơ chế tràn đĩa (Disk Spill), làm suy giảm nghiêm trọng tốc độ I/O hoặc gây lỗi `Container killed by YARN for exceeding memory limits`.

---

### 🔴 Bottleneck 2: Bóc tách chuỗi JSON bằng Regex Split thay vì Schema Parser chuẩn

- **Vị trí code:** `spark_etl/cleaner.py` (Dòng 76 - 81)
- **Vấn đề:**
  ```python
  F.split(F.regexp_replace(F.col("amenities"), r"[\[\]\"]", ""), ",")
  ```
- **Hậu quả khi scale 100x:** Regex replace trên chuỗi ký tự khổng lồ là thao tác tiêu tốn nhiều chu kỳ CPU của JVM. Đặc biệt, nếu tên tiện nghi chứa dấu phẩy (ví dụ: `"Clothing storage: closet, dresser"`), hàm `split(",")` sẽ phân tách sai thành 2 tiện nghi độc lập, làm sai lệch phân phối đặc trưng `amenities_count`.

---

### 🟡 Bottleneck 3: Chia Train/Test bằng `randomSplit` không đảm bảo tính phân tầng (Non-stratified Split)

- **Vị trí code:** `spark_etl/run_etl.py` (Dòng 96)
- **Vấn đề:**
  ```python
  train_raw, test_raw = silver_df.randomSplit([0.8, 0.2], seed=cfg.seed)
  ```
- **Hậu quả khi scale 100x:** `randomSplit` thực hiện băm ngẫu nhiên trên từng partition độc lập. Với các thành phố có số lượng listings ít hoặc phân bố dữ liệu đuôi dài (long-tail categories), tập Test có nguy cơ thiếu hụt hoàn toàn đại diện của một thành phố cụ thể, dẫn đến sai lệch đánh giá mô hình.

---

### 🟡 Bottleneck 4: Quá trình trích xuất ảnh thiếu Checkpoint / State Recovery

- **Vị trí code:** `extract_embedding.py` (Dòng 100 - 139)
- **Vấn đề:** Vòng lặp tải ảnh chạy tuần tự qua các chunk lớn và chỉ lưu file `.npy` ở dòng cuối cùng của hàm.
- **Hậu quả khi scale 100x:** Với quy mô 1 triệu ảnh (thời gian tải khoảng 8 - 12 tiếng), nếu gặp sự cố sập nguồn, tràn RAM hoặc gián đoạn mạng ở 95%, toàn bộ công sức tải trước đó sẽ bị mất trắng vì không có cơ chế lưu vết (Checkpointing/Savepoint).

---

### 🟡 Bottleneck 5: Hardcode cấu hình bộ nhớ và đường dẫn tương đối

- **Vị trí code:**
  - `spark_etl/spark_session.py` (Dòng 23 - 24): Hardcode `.config("spark.driver.memory", "4g")`.
  - `main.py` (Dòng 110), `extract_embedding.py` (Dòng 165): Hardcode đường dẫn tương đối `Path("data/gold_parquet")`.
- **Hậu quả khi scale 100x:** Khi triển khai trên hệ thống lập lịch tự động (Apache Airflow, Kubernetes Pod, Databricks Jobs), các worker chạy ở các thư mục thực thi khác nhau sẽ báo lỗi `FileNotFoundError`. Ngoài ra, mức RAM 4GB không đủ để xử lý các phép toán phức tạp khi dữ liệu tăng trưởng.

---

### 🟢 Bottleneck 6: Thực thi mã nguồn ở phạm vi toàn cục (Global Scope)

- **Vị trí code:** `diagnose_modality.py` (Dòng 38 - 144)
- **Vấn đề:** Toàn bộ code nạp dữ liệu, khởi tạo mô hình và chạy huấn luyện được đặt trực tiếp ở ngoài thân hàm (không nằm trong `def main():`).
- **Hậu quả:** Bất kỳ module nào import file này để tái sử dụng hàm tính toán đều sẽ vô tình kích hoạt toàn bộ quá trình train lại mô hình từ đầu, gây lãng phí GPU và tài nguyên tính toán.

---

# PHẦN 4: ĐỀ XUẤT CẢI TIẾN & LỘ TRÌNH NÂNG CẤP (ACTIONABLE NEXT STEPS)

Dưới đây là danh mục các hành động kỹ thuật cụ thể nhằm đưa pipeline đạt chuẩn Production Big Data, được sắp xếp theo mức độ ưu tiên từ cao xuống thấp:

## 4.1 Nhóm Ưu tiên CAO (Thực hiện ngay trước khi đưa lên Production)

### [FIX 1] Tối ưu hóa vòng đời Cache trong `spark_etl/run_etl.py`

- **Mục tiêu:** Giải phóng bộ nhớ đệm trước khi tạo vùng nhớ đệm mới.
- **Mã nguồn cải tiến:**

```python
# Giải phóng bộ nhớ tầng Silver thô trước khi cache dữ liệu phân tầng
silver_df.cache()
total_valid = silver_df.count()

if max_samples_per_city is not None:
    window_spec = Window.partitionBy("city_name").orderBy(F.rand(seed=cfg.seed))
    sampled_df = (
        silver_df.withColumn("row_num", F.row_number().over(window_spec))
        .filter(F.col("row_num") <= max_samples_per_city)
        .drop("row_num")
    )
    silver_df.unpersist() # Giải phóng ngay DataFrame cũ khỏi RAM
    sampled_df.cache()
    sampled_total = sampled_df.count()
    silver_df = sampled_df
```

### [FIX 2] Sử dụng `from_json` với Schema cố định cho cột tiện nghi trong `spark_etl/cleaner.py`

- **Mục tiêu:** Phân tích JSON chuẩn xác 100%, bảo vệ hệ thống trước các tiện nghi có chứa dấu phẩy.
- **Mã nguồn cải tiến:**

```python
from pyspark.sql.types import ArrayType, StringType

# Thay thế đoạn regex split cũ:
df_clean = df_clean.withColumn(
    "amenities_count",
    F.when(
        F.col("amenities").isNull() | (F.col("amenities") == "[]"), F.lit(0.0)
    ).otherwise(
        F.size(F.from_json(F.col("amenities"), ArrayType(StringType()))).cast("double")
    ),
)
```

### [FIX 3] Bổ sung cơ chế Resume / Skip trong `extract_embedding.py`

- **Mục tiêu:** Chống sập tiến trình, cho phép tiếp tục chạy tiếp nếu file embedding đã tồn tại.
- **Mã nguồn cải tiến:**

```python
def process_split(self, parquet_path: Path, output_prefix: Path) -> None:
    text_out = Path(f"{output_prefix}_text_emb.npy")
    img_out = Path(f"{output_prefix}_img_emb.npy")
  
    # Kiểm tra tính sẵn sàng để bỏ qua nếu đã trích xuất xong
    if text_out.exists() and img_out.exists():
        print(f"-> [SKIP] File embeddings của {parquet_path.name} đã tồn tại đầy đủ.")
        return
    # Tiếp tục tiến trình trích xuất bình thường...
```

---

## 4.2 Nhóm Ưu tiên TRUNG BÌNH (Cải thiện độ tin cậy và Khả năng bảo trì)

### [FIX 4] Phân tầng Train/Test Split theo `city_name` trong `spark_etl/run_etl.py`

- **Mục tiêu:** Đảm bảo tỷ lệ 80/20 được phân bổ đồng đều tuyệt đối trên từng thành phố.
- **Mã nguồn cải tiến:**

```python
# Phân tầng chuẩn 80/20 trên từng cửa sổ thành phố
split_window = Window.partitionBy("city_name").orderBy(F.rand(seed=cfg.seed))
silver_with_rank = silver_df.withColumn("rank_pct", F.cume_dist().over(split_window))

train_raw = silver_with_rank.filter(F.col("rank_pct") <= 0.8).drop("rank_pct")
test_raw = silver_with_rank.filter(F.col("rank_pct") > 0.8).drop("rank_pct")
```

### [FIX 5] Đóng gói toàn bộ script chạy thành hàm `main()` chuẩn

- **Mục tiêu:** Áp dụng chuẩn Clean Code cho `diagnose_modality.py`.
- **Mã nguồn cải tiến:**

```python
# diagnose_modality.py
def run_ablation_study() -> None:
    # Chuyển toàn bộ logic từ dòng 38 vào đây...
    pass

if __name__ == "__main__":
    run_ablation_study()
```

### [FIX 6] Động hóa đường dẫn bằng `PROJECT_ROOT` tuyệt đối

- **Mục tiêu:** Đảm bảo script thực thi ổn định từ bất kỳ thư mục nào trên hệ thống.
- **Mã nguồn cải tiến:**

```python
# Trong main.py, extract_embedding.py, diagnose_modality.py
PROJECT_ROOT = Path(__file__).resolve().parent
gold_dir = PROJECT_ROOT / "data" / "gold_parquet"
```

---

## 4.3 Nhóm Ưu tiên THẤP (Tối ưu hóa nâng cao cho Cụm Enterprise)

### [FIX 7] Tham số hóa cấu hình Spark qua biến môi trường (Environment Variables)

- **Mã nguồn cải tiến trong `spark_etl/spark_session.py`:**

```python
import os

driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "4g")
executor_memory = os.environ.get("SPARK_EXECUTOR_MEMORY", "4g")
master_url = os.environ.get("SPARK_MASTER", "local[*]")

spark = (
    SparkSession.builder.appName(app_name)
    .master(master_url)
    .config("spark.driver.memory", driver_memory)
    .config("spark.executor.memory", executor_memory)
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)
```

### [FIX 8] Nâng cấp mô hình: Thêm Cosine Annealing Learning Rate Scheduler & Gradient Clipping

- **Vị trí code:** `main.py`
- **Mục tiêu:** Giúp FT-Transformer hội tụ sâu hơn ở các epoch cuối và ngăn chặn hiện tượng nhảy vọt gradient khi gặp đặc trưng ngoại lai.

---

# TỔNG KẾT BẢNG ĐIỂM HỆ THỐNG (ARCHITECT SCORECARD)

```
+------------------------------------+------------+---------------------------------------+
| Hạng mục kiến trúc                 | Điểm số    | Nhận xét của Kiến trúc sư             |
+------------------------------------+------------+---------------------------------------+
| 1. Pipeline Architecture           | 9.0 / 10   | Medallion chuẩn, phân tầng rõ ràng    |
| 2. Chống rò rỉ dữ liệu (Leakage)   | 10.0 / 10  | Fit trên Train, Transform đồng bộ     |
| 3. Domain-aware Feature Engineering| 10.0 / 10  | Xuất sắc (Cắt trần luật lưu trú 30đêm)|
| 4. Mô hình học sâu đa phương thức  | 9.0 / 10   | FT-Transformer + Gated Fusion SOTA    |
| 5. Tối ưu hóa phân tán Big Data    | 7.5 / 10   | Native Spark tốt, cần sửa double cache|
| 6. Khả năng chịu lỗi (Resilience)  | 5.5 / 10   | Cần bổ sung Checkpoint khi tải ảnh    |
| 7. Chuẩn mã nguồn Clean Code       | 8.0 / 10   | Cần wrap hàm main, tuyệt đối hóa path |
+------------------------------------+------------+---------------------------------------+
| TỔNG ĐIỂM ĐÁNH GIÁ                 | 8.4 / 10   | ĐẠT XUẤT SẮC CẤP ĐỘ ĐỒ ÁN / LAB      |
|                                    |            | Cần 3 Fix ưu tiên cao để lên Prod     |
+------------------------------------+------------+---------------------------------------+
```
