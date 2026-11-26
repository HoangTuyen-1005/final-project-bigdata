"""Honest, isolated PySpark batch stress test: NOT real listings or model training.

Simulates growing Bronze input, duplicated events, a window deduplication,
shuffle aggregations, independently keyed embedding joins and partitioned lake
storage (local or MinIO s3a://). Reports wall time and file sizes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from contracts.state import resolve_data_root
from contracts.storage import StorageConfig, is_s3_uri, join_uri, parquet_size_bytes, to_s3a_uri, write_text
from spark_etl.spark_session import create_spark_session


def _resolve_benchmark_out(output: str | Path) -> str:
    text = str(output)
    if is_s3_uri(text):
        return to_s3a_uri(text).rstrip("/")
    if text.startswith("/data"):
        return str(resolve_data_root(text)).rstrip("/")
    cfg = StorageConfig.from_env()
    if cfg.backend == "s3":
        return f"s3a://{cfg.bucket}/{text.lstrip('/')}"
    return str(Path(text).resolve())


def run(rows: int, partitions: int, output: str | Path, explain: bool = False) -> dict:
    if rows < 100 or partitions < 1:
        raise ValueError("Require --rows >=100 and --partitions >=1")
    out_uri = _resolve_benchmark_out(output)
    parts = [p for p in out_uri.replace("\\", "/").split("/") if p]
    if "runs" in parts or parts[-1] in ("data", "raw", "bronze", "silver", "gold"):
        raise ValueError("Choose a dedicated benchmark output directory")
    if not is_s3_uri(out_uri):
        Path(out_uri).mkdir(parents=True, exist_ok=True)
    spark = create_spark_session("Airbnb_PySpark_Benchmark")
    timings = {}
    try:
        base = spark.range(rows, numPartitions=partitions)
        bronze = (base.select(F.concat(F.lit("listing_"), F.col("id")).alias("listing_key"),
                  F.concat(F.lit("City"), (F.col("id") % 4).cast("string")).alias("city"),
                  (F.lit(30.0) + (F.col("id") % 450)).cast("double").alias("price"),
                  F.lit(1).alias("version")))
        # Inject duplicates intentionally; latest version should win deterministically.
        duplicates = bronze.filter(F.pmod(F.xxhash64("listing_key"), F.lit(97)) == 0)
        duplicates = duplicates.withColumn("version", F.lit(2))
        bronze = bronze.unionByName(duplicates)
        bronze_uri = join_uri(out_uri, "bronze")
        start = time.monotonic()
        (bronze.repartition(partitions).write.mode("overwrite")
         .partitionBy("city").parquet(bronze_uri))
        timings["bronze_write_s"] = round(time.monotonic() - start, 3)

        start = time.monotonic()
        raw = spark.read.parquet(bronze_uri)
        rank = Window.partitionBy("listing_key").orderBy(F.col("version").desc())
        bucket = F.pmod(F.xxhash64("listing_key", F.lit(42)), F.lit(10))
        silver = (raw.withColumn("rn", F.row_number().over(rank))
                  .filter(F.col("rn") == 1).drop("rn")
                  .withColumn("log_price", F.log("price"))
                  .withColumn("split", F.when(bucket < 8, F.lit("train"))
                              .otherwise(F.lit("test"))))
        # Forces distributed groupBy/shuffle; no collect/toPandas of the large dataset.
        counts = {r["split"]: r["count"] for r in silver.groupBy("split").count().collect()}
        silver_uri = join_uri(out_uri, "silver")
        (silver.repartition(partitions).write.mode("overwrite")
         .partitionBy("split", "city").parquet(silver_uri))
        timings["silver_dedup_shuffle_write_s"] = round(time.monotonic() - start, 3)

        start = time.monotonic()
        clean = spark.read.parquet(silver_uri)
        # Synthetic embeddings, for testing the DATA PIPELINE only.
        # No E5/CLIP inference or claimed ML accuracy anywhere in this module.
        txt = (clean.filter(F.pmod(F.xxhash64("listing_key"), F.lit(11)) != 0)
               .select("listing_key", F.array_repeat(F.lit(0.1).cast("float"), 8).alias("x_txt")))
        img = (clean.filter(F.pmod(F.xxhash64("listing_key"), F.lit(7)) != 0)
               .select("listing_key", F.array_repeat(F.lit(0.2).cast("float"), 8).alias("x_img")))
        joined = (clean.join(txt, "listing_key", "left")
                  .join(img, "listing_key", "left")
                  .withColumn("available_mask", F.array(F.lit(True),
                         F.col("x_txt").isNotNull(), F.col("x_img").isNotNull())))
        if explain:
            print("--- Spark join execution plan ---")
            joined.explain(mode="formatted")
        availability = [r.asDict() for r in joined.groupBy("available_mask").count().collect()]
        gold_uri = join_uri(out_uri, "gold")
        (joined.repartition(partitions).write.mode("overwrite")
         .partitionBy("split", "city").parquet(gold_uri))
        timings["gold_join_write_s"] = round(time.monotonic() - start, 3)
        observed = spark.read.parquet(gold_uri).count()
        if observed != rows or sum(counts.values()) != rows:
            raise AssertionError("Lost or duplicated a listing during benchmark")
        report = {"kind": "SYNTHETIC_PIPELINE_BENCHMARK_NOT_TRAINING_DATA",
                  "output_uri": out_uri,
                  "rows_requested": rows, "bronze_rows_including_duplicates": rows + duplicates.count(),
                  "silver_rows": sum(counts.values()), "gold_rows": observed,
                  "spark_master": spark.sparkContext.master,
                  "spark_default_parallelism": spark.sparkContext.defaultParallelism,
                  "spark_shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
                  "partitions_requested": partitions,
                  "split_counts": counts, "availability": availability,
                  "timings": timings,
                  "disk_bytes": {stage: parquet_size_bytes(join_uri(out_uri, stage))
                                 for stage in ("bronze", "silver", "gold")}}
        write_text(join_uri(out_uri, "benchmark_report.json"), json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--out", default="/data/benchmark/synthetic_20k")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.rows, args.partitions, args.out, args.explain), indent=2))


if __name__ == "__main__":
    main()
