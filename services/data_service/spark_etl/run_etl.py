"""Batch lakehouse: persisted Bronze -> quality-controlled Silver -> keyed Gold.

GPU embedding extraction is a separate, resource-aware stage. Spark handles
all joins, validation, imputation, feature scaling and persisted Parquet layers.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel

from config import PRICE_CURRENCY, ModelConfig
from contracts.state import DEFAULT_DATA_DIR, get_paths, load_state, resolve_data_root, save_state, validate_run_id
from contracts.storage import StorageConfig, is_s3_uri, join_uri, list_s3_keys, to_s3a_uri, upload_file, uri_exists, write_text
from spark_etl.cleaner import AirbnbSparkCleaner
from spark_etl.feature_pipeline import SparkTabularFeaturizer
from spark_etl.spark_session import create_spark_session


SHARD_THRESHOLD_BYTES = 16 * 1024 * 1024  # 16 MiB
SHARD_ROWS_PER_PART = 10_000


def _shard_csv_to_s3(local_file: Path, parts_prefix: str, rows_per_part: int = SHARD_ROWS_PER_PART) -> None:
    """Split a large multiline CSV into gzipped CSV parts on S3 for parallel Spark reads."""
    csv.field_size_limit(min(sys.maxsize, 10 * 1024 * 1024))
    opener = gzip.open if local_file.suffix == ".gz" else open
    part_idx = 0

    def _upload_part(header: list[str], rows: list[list[str]], idx: int) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as gz_out:
                writer = csv.writer(gz_out)
                writer.writerow(header)
                writer.writerows(rows)
            upload_file(tmp_path, join_uri(parts_prefix, f"part-{idx:04d}.csv.gz"))
        finally:
            tmp_path.unlink(missing_ok=True)

    with opener(local_file, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            return
        batch: list[list[str]] = []
        for row in reader:
            batch.append(row)
            if len(batch) >= rows_per_part:
                _upload_part(header, batch, part_idx)
                part_idx += 1
                batch = []
        if batch or part_idx == 0:
            _upload_part(header, batch, part_idx)
            part_idx += 1
    write_text(join_uri(parts_prefix, "_SUCCESS"), "", encoding="utf-8")
    print(f"[Shard] {local_file.name} -> {part_idx} gzipped parts at {parts_prefix}", flush=True)


def resolve_sources(
    sources: list[str], data_dir: str | Path = DEFAULT_DATA_DIR
) -> list[tuple[str, str]]:
    """Resolve explicitly labeled US snapshots or discover raw/<city>/listings.csv[.gz]."""
    result = []
    if sources:
        for spec in sources:
            city, separator, raw_path = spec.partition("=")
            if not separator or not city.strip():
                raise ValueError(f"--source requires CITY=PATH, got: {spec}")
            if is_s3_uri(raw_path):
                s3_uri = to_s3a_uri(raw_path)
                if not uri_exists(s3_uri):
                    raise FileNotFoundError(f"Listings file does not exist: {s3_uri}")
                result.append((city.strip(), s3_uri))
            else:
                path = Path(raw_path)
                if not path.is_file():
                    raise FileNotFoundError(f"Listings file does not exist: {path}")
                result.append((city.strip(), str(path.resolve())))
    else:
        root = resolve_data_root(data_dir)
        if is_s3_uri(root):
            raw_prefix = join_uri(root, "raw")
            local_raw = Path("/data/raw")
            if str(root) == f"s3a://{StorageConfig.from_env().bucket}" and local_raw.is_dir():
                local_candidates = sorted((*local_raw.glob("*/listings.csv"), *local_raw.glob("*/listings.csv.gz")))
                for local_file in local_candidates:
                    if not local_file.is_file():
                        continue
                    city = local_file.parent.name
                    if local_file.stat().st_size >= SHARD_THRESHOLD_BYTES:
                        parts_prefix = join_uri(raw_prefix, city, "parts")
                        if not uri_exists(join_uri(parts_prefix, "_SUCCESS")):
                            _shard_csv_to_s3(local_file, parts_prefix)
                    else:
                        target_uri = join_uri(raw_prefix, city, local_file.name)
                        if not uri_exists(target_uri):
                            print(f"[Sync] Uploading {local_file} -> {target_uri}", flush=True)
                            upload_file(local_file, target_uri)
            city_sources: dict[str, list[str]] = {}
            sharded_cities: set[str] = set()
            for uri in list_s3_keys(raw_prefix):
                rel = uri.removeprefix(raw_prefix + "/")
                parts = rel.split("/")
                if len(parts) == 3 and parts[1] == "parts" and parts[2] == "_SUCCESS":
                    sharded_cities.add(parts[0])
                elif len(parts) == 2 and parts[1] in ("listings.csv", "listings.csv.gz"):
                    city_sources.setdefault(parts[0], []).append(uri)
            for city in sorted(sharded_cities | set(city_sources)):
                if city in sharded_cities:
                    result.append((city, join_uri(raw_prefix, city, "parts")))
                else:
                    for uri in city_sources.get(city, []):
                        result.append((city, uri))
        else:
            raw_dir = Path(root) / "raw"
            candidates = sorted((*raw_dir.glob("*/listings.csv"), *raw_dir.glob("*/listings.csv.gz")))
            result = [
                (path.parent.name, str(path.resolve())) for path in candidates if path.is_file()
            ]
    if not result:
        raise FileNotFoundError(
            f"No labeled US listings found under {join_uri(resolve_data_root(data_dir), 'raw')}. "
            "Place one listings.csv or listings.csv.gz in a subfolder for each city, "
            "or specify --source 'CITY=path/to/listings.csv.gz'."
        )
    if any("::" in city for city, _ in result):
        raise ValueError("City labels cannot contain '::', the listing_key separator")
    if len({city.casefold() for city, _ in result}) != len(result):
        raise ValueError("Use one snapshot per city. If both CSV and CSV.GZ exist, select one with --source.")
    if len({path for _, path in result}) != len(result):
        raise ValueError("A listings file cannot be assigned to multiple cities")
    return result


def save_quality(run_dir: Path | str, stage: str, report: dict) -> None:
    out = join_uri(run_dir, "quality", f"{stage}.json")
    write_text(out, json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare(data_dir: str, sources: list[str], write_partitions: int = 2,
            explain: bool = False, *, run_id: str | None = None) -> None:
    cfg = ModelConfig()
    inputs = resolve_sources(sources, data_dir)
    root, _ = get_paths(data_dir)
    run_id = uuid.uuid4().hex if run_id is None else validate_run_id(run_id)
    run_dir = join_uri(root, "runs", run_id)
    if not is_s3_uri(run_dir):
        Path(run_dir).mkdir(parents=True, exist_ok=False)
    silver_dir = join_uri(run_dir, "silver")
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
        bronze_dir = join_uri(run_dir, "bronze")
        (bronze.repartition(write_partitions).write.mode("overwrite")
         .partitionBy("city").parquet(bronze_dir))
        bronze = spark.read.parquet(bronze_dir)
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
                 .write.mode("overwrite").parquet(join_uri(silver_dir, split)))
        finally:
            silver.unpersist()
        report = {"run_id": run_id, "stage": "prepare", "source_count": len(inputs),
                  "price_currency": PRICE_CURRENCY,
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
                 "price_currency": PRICE_CURRENCY,
                 "sources": [{"city": c, "path": p} for c, p in inputs],
                 "bronze_counts": raw_counts, "silver_counts": counts,
                 "silver_city_counts": city_split_counts,
                 "numeric_null_rates": null_rates, "snapshot_days": snapshot_days,
                 "prepare_seconds": report["elapsed_seconds"], "seed": cfg.seed}
        save_state(data_dir, state)  # Publish active run only after all writes succeed.
        print(f"[Bronze/Silver] run={run_id}; counts={counts}; QA={join_uri(run_dir, 'quality/prepare.json')}")
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

def finalize(data_dir: str, write_partitions: int = 2, explain: bool = False,
             *, run_id: str | None = None) -> None:
    cfg = ModelConfig()
    state = load_state(data_dir, required_status="embedded", run_id=run_id)
    run_dir = state["run_path"]
    started = time.monotonic()
    spark = create_spark_session()
    cached = []
    try:
        joined, counts, coverage = {}, {}, {}
        for split in ("train", "val", "test"):
            raw = spark.read.parquet(join_uri(run_dir, "silver", split))
            txt = spark.read.parquet(join_uri(run_dir, "embeddings", f"{split}_text.parquet"))
            img = spark.read.parquet(join_uri(run_dir, "embeddings", f"{split}_image.parquet"))
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
        gold_dir = join_uri(run_dir, "gold")
        if not is_s3_uri(gold_dir):
            Path(gold_dir).mkdir(parents=True, exist_ok=True)
        if explain:
            print("[Spark physical plan: Gold]")
            frames["train"].explain(mode="formatted")
        for name, frame in frames.items():
            (frame.repartition(write_partitions).write.mode("overwrite")
             .parquet(join_uri(gold_dir, name)))
        model.write().overwrite().save(join_uri(gold_dir, "spark_mllib_pipeline"))
        metadata = {"run_id": state["run_id"], "cat_cardinalities": cards,
                    "cat_col_names": cfg.cat_cols, "num_feature_names": featurizer.num_feature_names,
                    "raw_num_cols": cfg.num_cols, "counts": counts,
                    "coverage": coverage, "has_availability_mask": True,
                    "text_dim": cfg.raw_text_dim, "image_dim": cfg.raw_image_dim,
                    "target": "natural_log_price", "price_currency": state.get("price_currency"),
                    "embedding_config": state.get("embedding_config"),
                    "tabular_preprocessor": featurizer.export_preprocessor_stats(model)}
        write_text(join_uri(gold_dir, "metadata.json"), json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        report = {"run_id": state["run_id"], "stage": "finalize",
                  "price_currency": state.get("price_currency"),
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
        save_state(data_dir, state, activate=run_id is None)
        print("[Gold] Finished:", gold_dir)
    finally:
        for df in cached:
            df.unpersist()
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "finalize"])
    parser.add_argument(
        "--source", action="append", default=[], metavar="CITY=CSV_PATH",
        help="Select sources explicitly for prepare (repeatable). "
             "When omitted, discover <data-dir>/raw/<city>/listings.csv or listings.csv.gz.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="Root for input snapshots and pipeline outputs (default: %(default)s).")
    parser.add_argument("--run-id", help="Create or finalize this specific run.")
    parser.add_argument("--write-partitions", type=int, default=2)
    parser.add_argument("--explain", action="store_true",
                        help="Print Spark physical execution plan for assessment")
    args = parser.parse_args()
    if args.write_partitions < 1:
        parser.error("--write-partitions must be >=1")
    if args.phase == "prepare":
        prepare(args.data_dir, args.source, args.write_partitions, args.explain, run_id=args.run_id)
    else:
        if args.source:
            parser.error("--source only valid for prepare")
        finalize(args.data_dir, args.write_partitions, args.explain, run_id=args.run_id)


if __name__ == "__main__":
    main()
