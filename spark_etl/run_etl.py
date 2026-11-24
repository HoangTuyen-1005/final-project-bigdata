"""Batch lakehouse: persisted Bronze -> quality-controlled Silver -> keyed Gold.

GPU embedding extraction is a separate, resource-aware stage. Spark handles
all joins, validation, imputation, feature scaling and persisted Parquet layers.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

from config import ModelConfig
from project_state import get_paths, load_state, save_state
from spark_etl.cleaner import AirbnbSparkCleaner
from spark_etl.feature_pipeline import SparkTabularFeaturizer
from spark_etl.spark_session import create_spark_session


def resolve_sources(sources: list[str], data_dir: str = "data") -> list[tuple[str, str]]:
    """CITY=/absolute/or/relative/file.csv[.gz], or <data_dir>/raw/<city>/..."""
    if sources:
        result = []
        for spec in sources:
            if "=" not in spec:
                raise ValueError(f"--source requires CITY=PATH, got: {spec}")
            city, raw_path = spec.split("=", 1)
            if not city.strip() or not Path(raw_path).is_file():
                raise FileNotFoundError(f"Invalid city or missing raw file: {spec}")
            result.append((city.strip(), str(Path(raw_path).resolve())))
    else:
        root = Path(data_dir)
        candidates = sorted((root / "raw").glob("*/listings.csv*"))
        if candidates:
            result = [(path.parent.name, str(path.resolve())) for path in candidates if path.is_file()]
        else:
            single = next((p for p in [root / "raw/listings.csv", root / "raw/listings.csv.gz",
                                      root / "listings.csv", root / "listings.csv.gz"]
                           if p.is_file()), None)
            if single is None:
                raise FileNotFoundError("No sources found. Use --source 'CITY=path/to/listings.csv.gz'.")
            result = [("Albany", str(single.resolve()))]
    if not result or len({city for city, _ in result}) != len(result):
        raise ValueError("Each source must have a unique city label")
    return result


def save_quality(run_dir: Path, stage: str, report: dict) -> None:
    out = run_dir / "quality" / f"{stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")


def prepare(data_dir: str, sources: list[str], write_partitions: int = 2,
            explain: bool = False) -> None:
    cfg = ModelConfig()
    inputs = resolve_sources(sources, data_dir)
    root, _ = get_paths(data_dir)
    run_id = uuid.uuid4().hex
    run_dir = root / "runs" / run_id
    silver_dir = run_dir / "silver"
    started = time.monotonic()
    spark = create_spark_session()
    try:
        # Bronze is a REAL immutable, run-scoped physical Parquet snapshot,
        # not just a label for the in-memory original CSV DataFrame.
        city_dfs = []
        for city, input_path in inputs:
            print(f"[Bronze] Reading {city}: {input_path}")
            raw = AirbnbSparkCleaner.load_raw_listings(spark, input_path)
            raw = (raw.withColumn("city", F.lit(city))
                   .withColumn("_source_path", F.lit(input_path))
                   .withColumn("_ingested_at", F.current_timestamp()))
            city_dfs.append(raw)
        bronze = city_dfs[0]
        for other in city_dfs[1:]:
            bronze = bronze.unionByName(other, allowMissingColumns=True)
        (bronze.repartition(write_partitions).write.mode("overwrite")
         .partitionBy("city").parquet(str(run_dir / "bronze")))
        bronze = spark.read.parquet(str(run_dir / "bronze"))
        raw_counts = {row["city"]: row["count"] for row in bronze.groupBy("city").count().collect()}

        silver = AirbnbSparkCleaner.clean_and_extract_features(bronze, cfg)
        selected = list(dict.fromkeys(["listing_key", "id", "city", "log_price",
                    "clean_full_text", "picture_url", *cfg.num_cols, *cfg.cat_cols,
                    *(["last_scraped"] if "last_scraped" in silver.columns else [])]))
        silver = silver.select(*selected)
        clean_before_dedup = silver.count()

        # Latest scrape wins, with reproducible content digest tie-breaking.
        scrape_col = (F.coalesce(F.col("last_scraped").cast("string"), F.lit(""))
                      if "last_scraped" in silver.columns else F.lit(""))
        digest = F.sha2(F.to_json(F.struct(*[F.col(c) for c in sorted(silver.columns)])), 256)
        rank = Window.partitionBy("listing_key").orderBy(scrape_col.desc(), digest.desc())
        silver = silver.withColumn("_rank", F.row_number().over(rank)).filter("_rank = 1").drop("_rank")
        bucket = F.pmod(F.xxhash64(F.col("listing_key"), F.lit(cfg.seed)), F.lit(10000))
        silver = silver.withColumn("split", F.when(bucket < 8000, F.lit("train"))
                                   .when(bucket < 9000, F.lit("val"))
                                   .otherwise(F.lit("test")))
        silver = silver.persist(StorageLevel.DISK_ONLY)
        try:
            counts = {row["split"]: row["count"] for row in silver.groupBy("split").count().collect()}
            city_split_counts = [r.asDict() for r in silver.groupBy("city", "split").count().collect()]
            null_rates = silver.agg(*[
                F.avg(F.col(c).isNull().cast("double")).alias(c) for c in cfg.num_cols
            ]).first().asDict()
            availability = silver.agg(
                F.sum((F.length(F.trim("clean_full_text")) > 0).cast("long")).alias("text_nonempty"),
                F.sum(F.col("picture_url").isNotNull().cast("long")).alias("image_url_valid")
            ).first().asDict()
            if "last_scraped" in silver.columns:
                snapshot_days = [r.asDict() for r in silver.groupBy("city").agg(
                    F.min("last_scraped").alias("first_snapshot"),
                    F.max("last_scraped").alias("last_snapshot")).collect()]
            else:
                snapshot_days = []
            if any(counts.get(x, 0) == 0 for x in ("train", "val", "test")):
                raise ValueError("Split empty; add more distinct listings")
            if explain:
                print("[Spark physical plan: Silver]")
                silver.explain(mode="formatted")
            for split in ("train", "val", "test"):
                (silver.filter(F.col("split") == split).drop("split")
                 .repartition(write_partitions)
                 .write.mode("overwrite").parquet(str(silver_dir / split)))
        finally:
            silver.unpersist()
        report = {"run_id": run_id, "stage": "prepare", "source_count": len(inputs),
                  "raw_rows": sum(raw_counts.values()), "raw_rows_by_city": raw_counts,
                  "clean_rows_before_dedup": clean_before_dedup,
                  "silver_rows": sum(counts.values()),
                  "rejected_id_or_price_rows": sum(raw_counts.values()) - clean_before_dedup,
                  "duplicate_rows_removed": clean_before_dedup - sum(counts.values()),
                  "silver_split_counts": counts, "city_split_counts": city_split_counts,
                  "numeric_null_rates": null_rates, "text_nonempty": availability["text_nonempty"],
                  "image_url_valid": availability["image_url_valid"],
                  "snapshot_days": snapshot_days, "write_partitions": write_partitions,
                  "elapsed_seconds": round(time.monotonic() - started, 2)}
        save_quality(run_dir, "prepare", report)
        state = {"run_id": run_id, "run_dir": f"runs/{run_id}", "status": "prepared",
                 "sources": [{"city": c, "path": p} for c, p in inputs],
                 "bronze_counts": raw_counts, "silver_counts": counts,
                 "silver_city_counts": city_split_counts,
                 "numeric_null_rates": null_rates, "snapshot_days": snapshot_days,
                 "prepare_seconds": report["elapsed_seconds"], "seed": cfg.seed}
        save_state(data_dir, state)  # Publish active run only after all writes succeed.
        print(f"[Bronze/Silver] run={run_id}; counts={counts}; QA={run_dir / 'quality/prepare.json'}")
    finally:
        spark.stop()


def _check_embedding(df, name: str, split: str, run_id: str, dimension: int) -> None:
    if df.filter(F.col("run_id").isNull() | (F.col("run_id") != run_id)).limit(1).count():
        raise ValueError(f"{name} embedding run_id mismatch: {split}")
    if df.filter(F.col("listing_key").isNull()).limit(1).count():
        raise ValueError(f"Null {name} listing_key in {split}")
    if df.groupBy("listing_key").count().filter("count > 1").limit(1).count():
        raise ValueError(f"Duplicate {name} embedding keys in {split}")
    vec = "embedding" if "embedding" in df.columns else f"x_{name}"
    bad = (F.col(vec).isNull() | (F.size(F.col(vec)) != dimension) |
           F.exists(F.col(vec), lambda x: x.isNull() | F.isnan(x) | (F.abs(x) > 1e20)))
    if df.filter(bad).limit(1).count():
        raise ValueError(f"Bad {name} embedding values/dimension in {split}")



def join_modalities(raw, txt, img, text_dim: int, image_dim: int):
    """Spark-only LEFT join with an explicit presence mask and zero placeholders.

    Caller must validate embedding run_id/key/dimension before this operation.
    """
    txt_col = "x_txt" if "x_txt" in txt.columns else "embedding"
    img_col = "x_img" if "x_img" in img.columns else "embedding"
    frame = (raw.join(txt.select("listing_key", F.col(txt_col).alias("x_txt")),
                      "listing_key", "left")
                .join(img.select("listing_key", F.col(img_col).alias("x_img")),
                      "listing_key", "left"))
    frame = frame.withColumn("text_present", F.col("x_txt").isNotNull())
    frame = frame.withColumn("image_present", F.col("x_img").isNotNull())
    frame = frame.withColumn("available_mask", F.array(
        F.lit(True), F.col("text_present"), F.col("image_present")))
    frame = frame.withColumn("x_txt", F.coalesce(
        F.col("x_txt"), F.array_repeat(F.lit(0.0).cast("float"), text_dim)))
    frame = frame.withColumn("x_img", F.coalesce(
        F.col("x_img"), F.array_repeat(F.lit(0.0).cast("float"), image_dim)))
    return frame

def finalize(data_dir: str, write_partitions: int = 2, explain: bool = False) -> None:
    cfg = ModelConfig()
    state = load_state(data_dir, required_status="embedded")
    run_dir = state["run_path"]
    started = time.monotonic()
    spark = create_spark_session()
    cached = []
    try:
        joined, counts, coverage = {}, {}, {}
        for split in ("train", "val", "test"):
            raw = spark.read.parquet(str(run_dir / "silver" / split))
            txt = spark.read.parquet(str(run_dir / "embeddings" / f"{split}_text.parquet"))
            img = spark.read.parquet(str(run_dir / "embeddings" / f"{split}_image.parquet"))
            _check_embedding(txt, "text", split, state["run_id"], cfg.raw_text_dim)
            _check_embedding(img, "image", split, state["run_id"], cfg.raw_image_dim)
            frame = join_modalities(raw, txt, img, cfg.raw_text_dim, cfg.raw_image_dim)
            frame = frame.persist(StorageLevel.DISK_ONLY)
            cached.append(frame)
            counts[split] = frame.count()
            if counts[split] != state["silver_counts"].get(split):
                raise ValueError(f"Keyed left joins changed Silver count for {split}")
            if counts[split] == 0:
                raise ValueError(f"Empty Gold split: {split}")
            coverage[split] = [r.asDict() for r in frame.groupBy(
                "text_present", "image_present").count().collect()]
            joined[split] = frame
        print("[Join] All Silver rows retained:", counts)
        print("[Join] Presence by split (tabular always True):", coverage)
        city_retained = []
        for split in ("train", "val", "test"):
            city_retained.extend([dict(r.asDict(), split=split) for r in
                                  joined[split].groupBy("city").count().collect()])
        featurizer = SparkTabularFeaturizer(cfg.num_cols, cfg.cat_cols,
                                           cfg.missing_numeric_strategy,
                                           cfg.unknown_category_train_prob, cfg.seed)
        frames, model, cards = featurizer.fit_and_transform(
            joined["train"], joined["val"], joined["test"])
        gold_dir = run_dir / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)
        if explain:
            print("[Spark physical plan: Gold]")
            frames["train"].explain(mode="formatted")
        for name, frame in frames.items():
            (frame.repartition(write_partitions).write.mode("overwrite")
             .parquet(str(gold_dir / name)))
        model.write().overwrite().save(str(gold_dir / "spark_mllib_pipeline"))
        metadata = {"run_id": state["run_id"], "cat_cardinalities": cards,
                    "cat_col_names": cfg.cat_cols, "num_feature_names": featurizer.num_feature_names,
                    "raw_num_cols": cfg.num_cols, "counts": counts,
                    "coverage": coverage, "has_availability_mask": True,
                    "text_dim": cfg.raw_text_dim, "image_dim": cfg.raw_image_dim,
                    "target": "natural_log_price"}
        (gold_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
        report = {"run_id": state["run_id"], "stage": "finalize",
                  "silver_counts": state["silver_counts"], "gold_counts": counts,
                  "modality_presence_by_split": coverage,
                  "city_split_counts": city_retained,
                  "write_partitions": write_partitions,
                  "elapsed_seconds": round(time.monotonic() - started, 2)}
        save_quality(run_dir, "finalize", report)
        state["status"] = "finalized"
        state["gold_counts"] = counts
        state["gold_city_counts"] = city_retained
        state["finalize_seconds"] = report["elapsed_seconds"]
        save_state(data_dir, state)
        print("[Gold] Finished:", gold_dir)
    finally:
        for df in cached:
            df.unpersist()
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "finalize"])
    parser.add_argument("--source", action="append", default=[], metavar="CITY=CSV_PATH")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--write-partitions", type=int, default=2)
    parser.add_argument("--explain", action="store_true",
                        help="Print Spark physical execution plan for assessment")
    args = parser.parse_args()
    if args.write_partitions < 1:
        parser.error("--write-partitions must be >=1")
    if args.phase == "prepare":
        prepare(args.data_dir, args.source, args.write_partitions, args.explain)
    else:
        if args.source:
            parser.error("--source only valid for prepare")
        finalize(args.data_dir, args.write_partitions, args.explain)


if __name__ == "__main__":
    main()
