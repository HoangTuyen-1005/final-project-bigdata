import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import ModelConfig
from spark_etl.cleaner import AirbnbSparkCleaner
from spark_etl.feature_pipeline import SparkTabularFeaturizer
from spark_etl.spark_session import create_spark_session


def find_raw_data_path(default_path: str = "data/listings.csv") -> str:
    """Tự động tìm kiếm file dữ liệu thô (.csv hoặc .csv.gz) trong data/ hoặc data/raw/."""
    candidates = [
        "data/listings.csv",
        "data/raw/listings.csv",
        "data/raw/listings.csv.gz",
        "data/listings.csv.gz",
        "data/raw/*listings.csv*",
        "data/*listings.csv*",
    ]
    for pattern in candidates:
        path_obj = Path(pattern)
        if "*" in pattern:
            if path_obj.parent.exists():
                matches = [f for f in path_obj.parent.glob(path_obj.name) if f.is_file()]
                if matches:
                    return matches[0].resolve().as_posix()
        elif path_obj.is_file():
            return path_obj.resolve().as_posix()
    return Path(default_path).resolve().as_posix()


def run_spark_etl_job(
    raw_csv_path: str | None = None,
    output_dir: str = "data/gold_parquet",
) -> None:
    cfg = ModelConfig()
    if raw_csv_path is None:
        raw_csv_path = find_raw_data_path()

    spark = create_spark_session()

    print(f"1. [Bronze -> Silver] Đọc dữ liệu từ: {raw_csv_path}")
    raw_df = AirbnbSparkCleaner.load_raw_listings(spark, raw_csv_path)
    silver_df = AirbnbSparkCleaner.clean_and_extract_features(raw_df)

    # Cache bảng Silver lên RAM vì ta sẽ dùng nó cho cả bước chia tách và fit MLlib
    silver_df.cache()
    total_valid = silver_df.count()
    print(f"-> Số lượng căn hộ hợp lệ sau khi lọc: {total_valid:,} dòng.")

    # 2. Chia Train / Test (80/20) TRƯỚC KHI chuẩn hóa để chống Data Leakage
    train_raw, test_raw = silver_df.randomSplit([0.8, 0.2], seed=cfg.seed)

    print("2. [Silver -> Gold] Huấn luyện Spark MLlib Pipeline (Imputer + Scaler + Indexer)...")
    featurizer = SparkTabularFeaturizer(num_cols=cfg.num_cols, cat_cols=cfg.cat_cols)
    train_gold, test_gold, pipeline_model, cat_cardinalities = (
        featurizer.fit_and_transform(train_raw, test_raw)
    )

    # 3. Lưu xuống định dạng chuẩn Big Data: Apache Parquet + Lưu Pipeline Model
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_gold.write.mode("overwrite").parquet(str(out_path / "train.parquet"))
    test_gold.write.mode("overwrite").parquet(str(out_path / "test.parquet"))
    pipeline_model.write().overwrite().save(str(out_path / "spark_mllib_pipeline"))

    # Lưu metadata cat_cardinalities để PyTorch FT-Transformer đọc lên
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"cat_cardinalities": cat_cardinalities}, f, indent=2)

    print(f"3. Hoàn tất! Dữ liệu Gold đã lưu tại: {output_dir}")
    print(f"-> Kích thước từ điển Categorical (cat_cardinalities): {cat_cardinalities}")

    silver_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    run_spark_etl_job()