import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def create_spark_session(app_name: str = "Airbnb_Multimodal_BigData_ETL") -> SparkSession:
    """Khởi tạo SparkSession với cấu hình tối ưu cho ETL và MLlib."""
    # 1. Khóa cứng trình thông dịch Python cho driver và worker executors
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # 2. Tự động nhận diện HADOOP_HOME cho Windows để chạy winutils.exe
    project_root = Path(__file__).resolve().parent.parent
    hadoop_home = project_root / "hadoop"
    if (hadoop_home / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        os.environ["PATH"] = str(hadoop_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")  # Khi chạy cluster (YARN/K8s) thì đổi hoặc truyền qua spark-submit
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        # Bật Adaptive Query Execution (AQE) của Spark 3.x để tối ưu hóa vật lý tự động
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Bật Apache Arrow để tăng tốc độ chuyển từ Spark DataFrame sang NumPy/PyTorch
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark