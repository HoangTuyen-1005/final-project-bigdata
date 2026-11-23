import json
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để chạy trực tiếp script hoặc module đều tìm thấy config
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from config import ModelConfig
from spark_etl.cleaner import AirbnbSparkCleaner
from spark_etl.feature_pipeline import SparkTabularFeaturizer
from spark_etl.spark_session import create_spark_session


def find_raw_data_path(default_path: str = "data/raw/*listings.csv") -> str:
    """
    Tự động tìm kiếm file dữ liệu thô (.csv hoặc .csv.gz) trong data/raw/ hoặc data/.
    Ưu tiên đọc toàn bộ các thành phố bằng pattern wildcard nếu có nhiều file.
    """
    raw_dir = project_root / "data" / "raw"
    if raw_dir.exists():
        csv_files = list(raw_dir.glob("*listings.csv")) + list(raw_dir.glob("*listings.csv.gz"))
        if len(csv_files) > 1:
            return (raw_dir / "*listings.csv*").as_posix()
        elif len(csv_files) == 1:
            return csv_files[0].resolve().as_posix()

    candidates = [
        project_root / "data" / "listings.csv",
        project_root / "data" / "raw" / "listings.csv",
        project_root / "data" / "raw" / "listings.csv.gz",
        project_root / "data" / "listings.csv.gz",
        project_root / "data" / "raw" / "*listings.csv*",
        project_root / "data" / "*listings.csv*",
    ]
    for pattern_path in candidates:
        pattern = str(pattern_path)
        path_obj = Path(pattern)
        if "*" in pattern:
            if path_obj.parent.exists():
                matches = [f for f in path_obj.parent.glob(path_obj.name) if f.is_file()]
                if matches:
                    return pattern
        elif path_obj.is_file():
            return path_obj.resolve().as_posix()

    return (project_root / default_path).resolve().as_posix()


def run_spark_etl_job(
    raw_csv_path: str | None = None,
    output_dir: str = "data/gold_parquet",
    max_samples_per_city: int | None = 4000,
) -> None:
    cfg = ModelConfig()
    if raw_csv_path is None:
        raw_csv_path = find_raw_data_path()

    spark = create_spark_session()

    print(f"1. [Bronze -> Silver] Đọc dữ liệu từ: {raw_csv_path}")
    raw_df = AirbnbSparkCleaner.load_raw_listings(spark, raw_csv_path)
    silver_df = AirbnbSparkCleaner.clean_and_extract_features(raw_df)

    silver_df.cache()
    total_valid = silver_df.count()
    print(f"-> Tổng số lượng căn hộ hợp lệ trước khi lấy mẫu: {total_valid:,} dòng.")

    # Thống kê số lượng listing theo từng thành phố
    print("-> Phân bố số lượng listings theo thành phố:")
    city_counts = silver_df.groupBy("city_name").count().collect()
    for row in city_counts:
        print(f"   • {row['city_name']}: {row['count']:,} listings")

    # Áp dụng Stratified Sampling theo thành phố (nếu được cấu hình)
    if max_samples_per_city is not None:
        print(f"\n-> Áp dụng Stratified Sampling: Tối đa {max_samples_per_city:,} mẫu/thành phố...")
        window_spec = Window.partitionBy("city_name").orderBy(F.rand(seed=cfg.seed))
        silver_df = (
            silver_df.withColumn("row_num", F.row_number().over(window_spec))
            .filter(F.col("row_num") <= max_samples_per_city)
            .drop("row_num")
        )
        silver_df.cache()
        sampled_total = silver_df.count()
        print(f"-> Tổng số lượng căn hộ sau Stratified Sampling: {sampled_total:,} dòng.")

    # 2. Chia Train / Test (80/20) TRƯỚC KHI chuẩn hóa để chống Data Leakage
    train_raw, test_raw = silver_df.randomSplit([0.8, 0.2], seed=cfg.seed)

    print("\n2. [Silver -> Gold] Huấn luyện Spark MLlib Pipeline (Imputer + Scaler + Indexer)...")
    featurizer = SparkTabularFeaturizer(num_cols=cfg.num_cols, cat_cols=cfg.cat_cols)
    train_gold, test_gold, pipeline_model, cat_cardinalities = (
        featurizer.fit_and_transform(train_raw, test_raw)
    )

    # 3. Lưu xuống định dạng chuẩn Big Data: Apache Parquet + Lưu Pipeline Model
    out_path = project_root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_gold.write.mode("overwrite").parquet(str(out_path / "train.parquet"))
    test_gold.write.mode("overwrite").parquet(str(out_path / "test.parquet"))
    pipeline_model.write().overwrite().save(str(out_path / "spark_mllib_pipeline"))

    # Lưu metadata cat_cardinalities để PyTorch FT-Transformer đọc lên
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"cat_cardinalities": cat_cardinalities}, f, indent=2)

    print(f"\n3. Hoàn tất! Dữ liệu Gold đã lưu tại: {out_path}")
    print(f"-> Kích thước từ điển Categorical (cat_cardinalities): {cat_cardinalities}")

    silver_df.unpersist()
    spark.stop()


if __name__ == "__main__":
    cfg = ModelConfig()
    run_spark_etl_job(max_samples_per_city=cfg.max_samples_per_city)
