"""Local Spark defaults may be overridden by spark-submit/Databricks."""
import os
import sys
from pathlib import Path
from pyspark import SparkConf
from pyspark.sql import SparkSession

from contracts.storage import StorageConfig


def create_spark_session(app_name: str = "Airbnb_Multimodal_BigData_ETL") -> SparkSession:
    # Do not override configured cluster Python executors or master.
    if not os.environ.get("SPARK_MASTER"):
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    hadoop_home = Path(__file__).resolve().parent.parent / "hadoop"
    if os.name == "nt" and (hadoop_home / "bin" / "winutils.exe").exists():
        os.environ.setdefault("HADOOP_HOME", str(hadoop_home))
        bin_dir = str(hadoop_home / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    builder = SparkSession.builder.appName(app_name)
    active_master = SparkConf(loadDefaults=True).get("spark.master", None)
    if os.environ.get("SPARK_MASTER"):
        builder = builder.master(os.environ["SPARK_MASTER"])
    elif not os.environ.get("DATABRICKS_RUNTIME_VERSION") and not active_master:
        builder = builder.master("local[2]").config("spark.driver.memory", "4g")
    if driver_host := os.environ.get("SPARK_DRIVER_HOST"):
        builder = builder.config("spark.driver.host", driver_host)
    # Local machines should not launch 200 tiny shuffle partitions by default.
    # Respect cluster/spark-submit settings and explicit overrides.
    local_default = not os.environ.get("SPARK_MASTER") and not active_master
    shuffle = os.environ.get("SPARK_SQL_SHUFFLE_PARTITIONS")
    if shuffle:
        builder = builder.config("spark.sql.shuffle.partitions", shuffle)
    elif local_default:
        builder = builder.config("spark.sql.shuffle.partitions", "8")
    builder = (builder.config("spark.sql.adaptive.enabled", "true")
               .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
               .config("spark.sql.adaptive.skewJoin.enabled", "true"))
    storage = StorageConfig.from_env()
    if storage.backend == "s3" or os.environ.get("S3_ENDPOINT"):
        ssl_enabled = "true" if storage.endpoint.lower().startswith("https://") else "false"
        builder = (
            builder.config("spark.hadoop.fs.s3a.endpoint", storage.endpoint)
            .config("spark.hadoop.fs.s3a.access.key", storage.access_key)
            .config("spark.hadoop.fs.s3a.secret.key", storage.secret_key)
            .config("spark.hadoop.fs.s3a.path.style.access", "true")
            .config("spark.hadoop.fs.s3a.connection.ssl.enabled", ssl_enabled)
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            )
        )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
