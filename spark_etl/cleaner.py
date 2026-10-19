from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class AirbnbSparkCleaner:
    """Làm sạch dữ liệu thô Inside Airbnb bằng 100% Native Spark SQL Expressions."""

    @staticmethod
    def load_raw_listings(spark, input_path: str) -> DataFrame:
        """
        Đọc dữ liệu thô (hỗ trợ đọc 1 file hoặc đọc hàng loạt file bằng wildcard *.csv.gz).
        Bắt buộc bật multiLine=True vì cột description của Airbnb chứa rất nhiều ký tự xuống dòng.
        """
        return (
            spark.read.option("header", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .option("quote", '"')
            .option("inferSchema", "false")  # Đọc dạng String trước để tự ép kiểu an toàn
            .csv(input_path)
        )

    @staticmethod
    def clean_and_extract_features(df: DataFrame) -> DataFrame:
        """Biến đổi dữ liệu thô sang bảng Silver sạch, loại bỏ rác và trích xuất đặc trưng."""

        # 1. Làm sạch cột Target `price`: xóa '$' và ',', ép kiểu Double, lọc khoảng hợp lý
        df_clean = df.withColumn(
            "price_clean",
            F.regexp_replace(F.col("price"), r"[\$,]", "").cast("double"),
        ).filter(
            F.col("price_clean").isNotNull()
            & (F.col("price_clean") >= 10.0)
            & (F.col("price_clean") <= 5000.0)
        )

        # Tạo cột mục tiêu y = log(price)
        df_clean = df_clean.withColumn("log_price", F.log(F.col("price_clean")))

        # 2. Lọc bỏ các mẫu không có URL ảnh hợp lệ
        df_clean = df_clean.filter(
            F.col("picture_url").isNotNull() & F.col("picture_url").startswith("http")
        )

        # 3. Bóc tách cột `bathrooms` từ `bathrooms_text` bằng Regex trên JVM:
        # VD: "1.5 shared baths" -> 1.5 | "Half-bath" -> 0.5
        extracted_bath = F.regexp_extract(F.col("bathrooms_text"), r"(\d+\.?\d*)", 1)
        df_clean = df_clean.withColumn(
            "bathrooms",
            F.when(
                F.lower(F.col("bathrooms_text")).contains("half"), F.lit(0.5)
            ).otherwise(
                F.when(extracted_bath != "", extracted_bath.cast("double")).otherwise(
                    F.col("bathrooms").cast("double")
                )
            ),
        )

        # 4. Đếm số lượng tiện nghi `amenities_count` từ chuỗi JSON Array '["Wifi", "Pool"]'
        # Thay vì parse JSON tốn CPU, ta tách mảng bằng dấu phẩy sau khi xóa ngoặc vuông
        df_clean = df_clean.withColumn(
            "amenities_count",
            F.when(
                F.col("amenities").isNull() | (F.col("amenities") == "[]"), F.lit(0.0)
            ).otherwise(
                F.size(F.split(F.regexp_replace(F.col("amenities"), r"[\[\]\"]", ""), ",")).cast("double")
            ),
        )

        # 5. Làm sạch tỷ lệ % của Host ("98%" -> 0.98)
        # TODO: Tạm thời không tự ý điền 0.0 vào các cột tỷ lệ host vì sẽ làm sai lệch phân phối.
        # Sau này sẽ thu thập đủ dữ liệu hoặc áp dụng mô hình Imputation (MICE / KNN) chuyên sâu sau.
        for rate_col in ["host_response_rate", "host_acceptance_rate"]:
            df_clean = df_clean.withColumn(
                rate_col,
                F.regexp_replace(F.col(rate_col), "%", "").cast("double") / 100.0,
            )

        # 6. Ép kiểu các cột Numeric còn lại
        numeric_casts = [
            "latitude",
            "longitude",
            "accommodates",
            "bedrooms",
            "beds",
            "calculated_host_listings_count",
            "minimum_nights",
            "number_of_reviews",
            "review_scores_rating",
        ]
        for col_name in numeric_casts:
            df_clean = df_clean.withColumn(col_name, F.col(col_name).cast("double"))

        # TODO: Tạm thời bỏ qua (loại bỏ) các record bị khuyết giá trị trong dataset (Listwise Deletion).
        # Phương pháp này bảo toàn tính chân thực của phân phối dữ liệu, tránh điền giá trị giả định (như 0.0).
        # Sau này sẽ nghiên cứu áp dụng các kỹ thuật Imputation tối ưu hơn (MICE / KNN) để tận dụng lại các record này.
        required_numeric_cols = [
            "latitude",
            "longitude",
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "amenities_count",
            "calculated_host_listings_count",
            "minimum_nights",
            "number_of_reviews",
            "review_scores_rating",
        ]
        for col_name in required_numeric_cols:
            df_clean = df_clean.filter(F.col(col_name).isNotNull())

        # 7. Làm sạch các cột Categorical (điền 'Unknown' hoặc 'f' nếu thiếu)
        df_clean = (
            df_clean.fillna(
                {
                    "neighbourhood_cleansed": "Unknown",
                    "property_type": "Other",
                    "room_type": "Unknown",
                    "host_is_superhost": "f",
                    "instant_bookable": "f",
                }
            )
        )

        # 8. Làm sạch nhánh Text: Gộp name + description + neighborhood_overview và xóa thẻ HTML
        raw_text_concat = F.concat_ws(
            " . ",
            F.coalesce(F.col("name"), F.lit("")),
            F.coalesce(F.col("description"), F.lit("")),
            F.coalesce(F.col("neighborhood_overview"), F.lit("")),
        )
        df_clean = df_clean.withColumn(
            "clean_full_text",
            F.trim(
                F.regexp_replace(
                    F.regexp_replace(raw_text_concat, r"<[^>]+>", " "), r"\s+", " "
                )
            ),
        )

        return df_clean