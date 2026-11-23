from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class AirbnbSparkCleaner:
    """Làm sạch dữ liệu thô Inside Airbnb bằng 100% Native Spark SQL Expressions."""

    @staticmethod
    def load_raw_listings(spark, input_path: str) -> DataFrame:
        """
        Đọc dữ liệu thô (hỗ trợ đọc 1 file hoặc đọc hàng loạt file bằng wildcard *.csv).
        Bắt buộc bật multiLine=True vì cột description của Airbnb chứa nhiều ký tự xuống dòng.
        """
        return (
            spark.read.option("header", "true")
            .option("multiLine", "true")
            .option("escape", '"')
            .option("quote", '"')
            .option("inferSchema", "false")
            .csv(input_path)
        )

    @staticmethod
    def clean_and_extract_features(df: DataFrame) -> DataFrame:
        """Biến đổi dữ liệu thô sang bảng Silver sạch, loại bỏ rác và trích xuất đặc trưng đa thành phố & luật thuê nhà."""

        # 0. Trích xuất tên thành phố từ input_file_name() nếu chưa có
        file_path_lower = F.lower(F.input_file_name())
        df_clean = df.withColumn(
            "city_name",
            F.when(file_path_lower.contains("hawaii"), F.lit("Hawaii"))
            .when(file_path_lower.contains("london"), F.lit("London"))
            .when(file_path_lower.contains("los_angeles"), F.lit("Los Angeles"))
            .when(file_path_lower.contains("new_york"), F.lit("New York"))
            .otherwise(F.lit("Other")),
        )

        # 1. Làm sạch cột Target `price`: xóa '$' và ',', ép kiểu Double, lọc khoảng hợp lý (10$ - 5000$)
        df_clean = df_clean.withColumn(
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
        df_clean = df_clean.withColumn(
            "amenities_count",
            F.when(
                F.col("amenities").isNull() | (F.col("amenities") == "[]"), F.lit(0.0)
            ).otherwise(
                F.size(
                    F.split(
                        F.regexp_replace(F.col("amenities"), r"[\[\]\"]", ""), ","
                    )
                ).cast("double")
            ),
        )

        # 5. Xử lý quy định / luật lưu trú (Local Law 18 NYC, HSO LA, Bill 41 Hawaii, 90-day London):
        # - minimum_nights_capped: Cắt trần tại 30 đêm nhằm triệt tiêu các outlier (365, 1125 đêm)
        # - is_long_term_stay: Cờ nhị phân (1.0 nếu >= 30 đêm, 0.0 nếu < 30 đêm)
        raw_min_nights = F.coalesce(F.col("minimum_nights").cast("double"), F.lit(1.0))
        df_clean = df_clean.withColumn(
            "minimum_nights_capped",
            F.when(raw_min_nights > 30.0, F.lit(30.0))
            .when(raw_min_nights < 1.0, F.lit(1.0))
            .otherwise(raw_min_nights),
        )
        df_clean = df_clean.withColumn(
            "is_long_term_stay",
            F.when(raw_min_nights >= 30.0, F.lit(1.0)).otherwise(F.lit(0.0)),
        )

        # 6. Ép kiểu các cột Numeric còn lại
        numeric_casts = [
            "latitude",
            "longitude",
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "amenities_count",
            "calculated_host_listings_count",
            "minimum_nights_capped",
            "is_long_term_stay",
            "number_of_reviews",
            "review_scores_rating",
        ]
        for col_name in numeric_casts:
            df_clean = df_clean.withColumn(col_name, F.col(col_name).cast("double"))

        # Lọc bỏ các dòng bị khuyết các thuộc tính số quan trọng (Listwise Deletion an toàn)
        for col_name in numeric_casts:
            df_clean = df_clean.filter(F.col(col_name).isNotNull())

        # 7. Làm sạch các cột Categorical (điền 'Unknown' hoặc 'f' nếu thiếu)
        df_clean = df_clean.fillna(
            {
                "neighbourhood_cleansed": "Unknown",
                "property_type": "Other",
                "room_type": "Unknown",
                "host_is_superhost": "f",
                "instant_bookable": "f",
                "city_name": "Other",
            }
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
