# Dữ liệu đầu vào

Tạo một thư mục cho mỗi thành phố Mỹ. Trong thư mục đó, đặt một tệp **Detailed
Listings** của Inside Airbnb với tên `listings.csv` hoặc `listings.csv.gz`. Ví dụ:

```text
data/raw/Boston/listings.csv.gz
data/raw/Chicago/listings.csv
```

Tên thư mục được dùng trong khóa `city::id`. Giữ tên này nhất quán khi thay
snapshot. Giá trong tệp nguồn dùng USD.

Từ thư mục gốc của project, chạy:

```bash
python -m spark_etl.run_etl prepare
```
