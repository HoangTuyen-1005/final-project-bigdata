# TÀI LIỆU PHÂN TÍCH QUY ĐỊNH PHÁP LÝ & CHIẾN LƯỢC ĐA THÀNH PHỐ (MULTI-CITY AIRBNB)

Tài liệu này tổng hợp cơ sở lý thuyết, phân tích dữ liệu thực tế và giải pháp kỹ thuật xử lý các chính sách nhà ở và luật lưu trú ảnh hưởng đến bài toán định giá căn hộ Airbnb Đa phương thức (Multimodal Deep Learning).

---

## 1. Cơ sở pháp lý & Tác động thực tế trong tập dữ liệu

Khi mở rộng bài toán từ một thành phố đơn lẻ sang quy mô đa vùng địa lý (**New York, Los Angeles, Hawaii, London**), các quy định pháp lý địa phương chi phối trực tiếp hành vi định giá của chủ nhà (`price`) và số đêm lưu trú tối thiểu (`minimum_nights`).

### 1.1. New York City — Local Law 18 (Short-Term Rental Registration Law - 2023)
* **Quy định:** Cấm hoàn toàn việc cho thuê ngắn hạn dưới 30 ngày trừ khi chủ nhà có đăng ký chính thức với chính quyền thành phố, bắt buộc phải có mặt tại căn hộ trong suốt thời gian khách ở và không được khóa cửa phòng khách.
* **Tác động dữ liệu:**
  * **74.5%** tổng số listing trên Airbnb NYC đặt `minimum_nights = 30` ngày để lách luật chuyển sang mô hình dài hạn (Long-term residential).
  * **Chênh lệch giá:** Giá thuê ngắn hạn trung vị là **$360.5/đêm**, trong khi các căn dài hạn (>= 30 đêm) chỉ có giá trung vị **$143.0/đêm** (chiết khấu sâu tới **-60.3%**).

### 1.2. Los Angeles — Home-Sharing Ordinance (HSO)
* **Quy định:** Pháp lệnh Home-Sharing của thành phố Los Angeles quy định rõ ràng: Chỉ cho phép cho thuê ngắn hạn (< 30 ngày) tại **nhà chính chủ (Primary Residence)** — nơi chủ nhà cư trú tối thiểu 183 ngày/năm. Giới hạn tối đa **120 ngày cho thuê ngắn hạn/năm** với đăng ký tiêu chuẩn. Bất động sản thứ hai, nhà đầu tư hoặc pháp nhân (LLC, công ty) **bị cấm hoàn toàn** cho thuê ngắn hạn. Nếu muốn vượt 120 ngày, chủ nhà phải xin giấy phép Extended Home-Sharing với quy trình nghiêm ngặt hơn (đã hoạt động tối thiểu 6 tháng, không vi phạm trong 3 năm, phải thông báo cho hàng xóm).
* **Lưu ý:** California AB 537 (có hiệu lực 01/07/2024) là luật riêng biệt về **minh bạch giá (Junk Fee Ban)** — cấm các nền tảng ẩn phí dọn phòng, phí resort, không liên quan trực tiếp đến hạn chế số ngày cho thuê.
* **Tác động dữ liệu:**
  * **54.8%** listing có `minimum_nights >= 30` ngày — phần lớn là các chủ nhà sở hữu nhiều bất động sản buộc chuyển sang hợp đồng thuê theo tháng vì không đủ điều kiện đăng ký Home-Sharing.
  * Mức chiết khấu cho khách thuê dài hạn đạt **-46.9%** so với ngắn hạn ($240.5/đêm vs $127.7/đêm).

### 1.3. Hawaii — Honolulu Bill 41 (Ordinance 22-7, 2022)
* **Quy định:** Tại Oahu (Honolulu), Bill 41 được thông qua tháng 4/2022 nhằm siết chặt cho thuê ngắn hạn ngoài khu vực Resort (Waikiki, Ko Olina, Turtle Bay). Ban đầu Bill 41 đặt ngưỡng tối thiểu **90 ngày**, nhưng tòa án liên bang (Thẩm phán Derrick Watson) đã ra lệnh cấm thi hành điều khoản này vào tháng 10/2022, giữ nguyên ngưỡng cũ là **30 ngày** cho các vùng ngoài resort. Các đảo khác (Maui, Big Island, Kauai) có quy định riêng theo từng quận, nhìn chung vẫn thân thiện hơn với du lịch ngắn hạn.
* **Lưu ý:** HRS §514E (Hawaii Revised Statutes, Chapter 514E) là luật riêng biệt chỉ quy định về **Timeshare (sở hữu kỳ nghỉ)**, không phải luật quản lý cho thuê ngắn hạn trên Airbnb.
* **Tác động dữ liệu:**
  * Chỉ có **7.6%** listing đặt `minimum_nights >= 30` ngày — phần lớn listing trên Airbnb Hawaii nằm trong khu vực Resort được phép cho thuê ngắn hạn.
  * Trung vị số đêm tối thiểu toàn bang chỉ là **2 ngày**.

### 1.4. London — Deregulation Act 2015 (Section 44) & 90-Day Rule
* **Quy định:** Greater London Council (General Powers) Act 1973 (Section 25) ban đầu **cấm hoàn toàn** mọi hình thức cho thuê ngắn hạn tại London mà không có giấy phép quy hoạch. Deregulation Act 2015 (Section 44, có hiệu lực từ tháng 1/2017) đã **nới lỏng** luật này: cho phép chủ nhà cho thuê toàn bộ căn hộ (Entire Home) tối đa **90 đêm trong một năm dương lịch** mà không cần xin giấy phép thay đổi mục đích sử dụng đất. Giới hạn 90 đêm áp dụng cho **bất động sản** (không phải nền tảng) — nếu đăng trên cả Airbnb và Booking.com thì tổng số đêm cộng dồn. Airbnb tự động chặn đặt phòng khi listing đạt 90 đêm. Nếu vượt quá mà không có Planning Permission, chủ nhà có thể bị phạt tới **£20,000**.
* **Tác động dữ liệu:**
  * Do trần giới hạn là 90 đêm/năm, các chủ nhà giữ nguyên số đêm tối thiểu rất ngắn để tối đa hóa số lượt khách trên từng đêm được phép (chỉ **3.0%** listing đặt `>= 30 ngày`, trung vị là **2 ngày**).

---

## 2. Các rủi ro đối với mô hình Deep Learning nếu không xử lý

1. **Hiện tượng Outliers cực đoan của `minimum_nights`:** Nhiều chủ nhà nhập giá trị 365, 730 hoặc 1,125 đêm. Nếu đưa trực tiếp vào StandardScaler và Transformer, các giá trị này sẽ kéo lệch phương sai, làm bão hòa attention weights.
2. **Nghịch lý giá / đêm (Price Inversion):** Khi không có nhãn ngữ cảnh thành phố và cờ dài hạn, mô hình sẽ nhầm tưởng rằng "căn hộ càng bắt buộc ở nhiều đêm thì giá thuê mỗi đêm càng rẻ" mà không hiểu được đây là chiết khấu hợp đồng tháng theo luật.
3. **Mất cân bằng dữ liệu giữa các thành phố (Sampling Bias):** Dữ liệu London (~92k dòng hợp lệ) áp đảo hoàn toàn Hawaii (~39k) và New York (~30k), khiến mô hình thiên vị đặc trưng của các đô thị lớn tại Anh.

---

## 3. Kiến trúc kỹ thuật và Phương pháp xử lý

### 3.1. Cắt trần giá trị ngoại lai (Domain-aware Capping)
Giá trị `minimum_nights` được cắt trần tại ngưỡng pháp lý 30 ngày:
$$\text{minimum\_nights\_capped} = \min(\max(\text{minimum\_nights}, 1.0), 30.0)$$

### 3.2. Bổ sung cờ nhị phân phân loại thị trường (Binary Regulatory Flag)
Tạo đặc trưng chỉ báo hợp đồng dài hạn:
$$\text{is\_long\_term\_stay} = \begin{cases} 1.0 & \text{nếu } \text{minimum\_nights} \ge 30 \\ 0.0 & \text{ngược lại} \end{cases}$$

### 3.3. Nhận diện ngữ cảnh thành phố (Contextual City Indexing)
Trích xuất tự động `city_name` từ metadata đường dẫn file bằng Native Spark JVM, mã hóa thành Categorical Embedding 128 chiều qua FT-Transformer:
```python
file_path_lower = F.lower(F.input_file_name())
df = df.withColumn(
    "city_name",
    F.when(file_path_lower.contains("hawaii"), "Hawaii")
    .when(file_path_lower.contains("london"), "London")
    .when(file_path_lower.contains("los_angeles"), "Los Angeles")
    .when(file_path_lower.contains("new_york"), "New York")
    .otherwise("Other")
)
```

### 3.4. Lấy mẫu phân tầng cân bằng (Stratified Sampling on Big Data)
Sử dụng Spark Window Partitioning để lấy ngẫu nhiên có hạt giống cố định (Deterministic seed) tối đa $N = 4,000$ mẫu hợp lệ cho từng thành phố:
```python
window_spec = Window.partitionBy("city_name").orderBy(F.rand(seed=42))
silver_sampled = silver_df.withColumn("row_num", F.row_number().over(window_spec)) \
                          .filter(F.col("row_num") <= max_samples_per_city) \
                          .drop("row_num")
```
Tổng mẫu Gold sau lấy mẫu đạt $\approx 16,000$ căn hộ, giúp thời gian cào ảnh CLIP/E5 duy trì trong khoảng hợp lý (20-30 phút) và đảm bảo công bằng phân phối địa lý.

---

## 4. Bảng danh mục đặc trưng sau cập nhật

### Nhánh Số (12 thuộc tính - Chuẩn hóa Z-score phân tán)
* `latitude`, `longitude`: Tọa độ địa lý.
* `accommodates`, `bedrooms`, `beds`, `bathrooms`: Quy mô căn hộ.
* `amenities_count`: Tổng số tiện nghi trích xuất từ mảng JSON.
* `calculated_host_listings_count`: Quy mô kinh doanh của chủ nhà.
* **`minimum_nights_capped`**: Số đêm tối thiểu đã chuẩn hóa theo ngưỡng luật (1 - 30).
* **`is_long_term_stay`**: Cờ nhận diện cho thuê dài hạn (0.0 hoặc 1.0).
* `number_of_reviews`, `review_scores_rating`: Uy tín và đánh giá từ khách.

### Nhánh Phân loại (6 thuộc tính - FT-Transformer Feature Tokenizer)
* `neighbourhood_cleansed`: Phân vùng dân cư / quận.
* `property_type`: Loại hình bất động sản.
* `room_type`: Toàn bộ căn hộ, phòng riêng, phòng chia sẻ.
* `host_is_superhost`: Trạng thái Chủ nhà siêu cấp.
* `instant_bookable`: Cho phép đặt phòng tức thì.
* **`city_name`**: Tên thành phố đại diện cho chính sách pháp lý riêng.
