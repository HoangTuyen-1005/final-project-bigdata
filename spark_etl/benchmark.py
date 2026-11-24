"""Honest, isolated PySpark batch stress test: NOT real listings or model training.

Simulates growing Bronze input, duplicated events, a window deduplication,
shuffle aggregations, independently keyed embedding joins and partitioned lake
storage. Reports wall time and file sizes, rather than claiming a real cluster.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from spark_etl.spark_session import create_spark_session


def size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*.parquet") if p.is_file())


def run(rows: int, partitions: int, output: Path, explain: bool = False) -> dict:
    if rows < 100 or partitions < 1:
        raise ValueError("Require --rows >=100 and --partitions >=1")
    output = output.resolve()
    # Destroy *only* benchmark assets, never the project's real data/ or runs/.
    if "runs" in output.parts or output.name in ("data", "raw", "bronze", "silver", "gold"):
        raise ValueError("Choose a dedicated benchmark output directory")
    output.mkdir(parents=True, exist_ok=True)
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
        start = time.monotonic()
        (bronze.repartition(partitions).write.mode("overwrite")
         .partitionBy("city").parquet(str(output / "bronze")))
        timings["bronze_write_s"] = round(time.monotonic() - start, 3)

        start = time.monotonic()
        raw = spark.read.parquet(str(output / "bronze"))
        rank = Window.partitionBy("listing_key").orderBy(F.col("version").desc())
        bucket = F.pmod(F.xxhash64("listing_key", F.lit(42)), F.lit(10))
        silver = (raw.withColumn("rn", F.row_number().over(rank))
                  .filter(F.col("rn") == 1).drop("rn")
                  .withColumn("log_price", F.log("price"))
                  .withColumn("split", F.when(bucket < 8, F.lit("train"))
                              .otherwise(F.lit("test"))))
        # Forces distributed groupBy/shuffle; no collect/toPandas of the large dataset.
        counts = {r["split"]: r["count"] for r in silver.groupBy("split").count().collect()}
        (silver.repartition(partitions).write.mode("overwrite")
         .partitionBy("split", "city").parquet(str(output / "silver")))
        timings["silver_dedup_shuffle_write_s"] = round(time.monotonic() - start, 3)

        start = time.monotonic()
        clean = spark.read.parquet(str(output / "silver"))
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
        (joined.repartition(partitions).write.mode("overwrite")
         .partitionBy("split", "city").parquet(str(output / "gold")))
        timings["gold_join_write_s"] = round(time.monotonic() - start, 3)
        observed = spark.read.parquet(str(output / "gold")).count()
        if observed != rows or sum(counts.values()) != rows:
            raise AssertionError("Lost or duplicated a listing during benchmark")
        report = {"kind": "SYNTHETIC_PIPELINE_BENCHMARK_NOT_TRAINING_DATA",
                  "rows_requested": rows, "bronze_rows_including_duplicates": rows + duplicates.count(),
                  "silver_rows": sum(counts.values()), "gold_rows": observed,
                  "spark_master": spark.sparkContext.master,
                  "spark_default_parallelism": spark.sparkContext.defaultParallelism,
                  "spark_shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
                  "partitions_requested": partitions,
                  "split_counts": counts, "availability": availability,
                  "timings": timings,
                  "disk_bytes": {stage: size_bytes(output / stage)
                                 for stage in ("bronze", "silver", "gold")}}
        (output / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--out", default="data/benchmark/synthetic_20k")
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.rows, args.partitions, Path(args.out), args.explain), indent=2))


if __name__ == "__main__":
    main()
