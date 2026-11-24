"""Native Spark transformations for Inside Airbnb. All numerical NULLs survive to MLlib."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from config import ModelConfig


class AirbnbSparkCleaner:
    @staticmethod
    def load_raw_listings(spark, input_path: str) -> DataFrame:
        # Multiline descriptions must be parsed as CSV fields.
        return (spark.read.option("header", "true").option("multiLine", "true")
                .option("escape", '"').option("quote", '"')
                .option("inferSchema", "false").csv(input_path))

    @staticmethod
    def clean_and_extract_features(df: DataFrame, cfg: ModelConfig | None = None) -> DataFrame:
        cfg = cfg or ModelConfig()
        required = {"id", "city", "price"}
        absent = required - set(df.columns)
        if absent:
            raise ValueError(f"Raw data is missing mandatory columns: {sorted(absent)}")

        # Some city exports omit optional fields; preserve them as NULL for imputation.
        optional = set(cfg.num_cols + cfg.cat_cols + [
            "name", "description", "picture_url", "neighborhood_overview",
            "bathrooms_text", "amenities",
        ]) - set(df.columns)
        for field in sorted(optional):
            df = df.withColumn(field, F.lit(None).cast("string"))

        df = df.withColumn("id", F.trim(F.col("id").cast("string")))
        df = df.filter(F.col("id").isNotNull() & (F.length("id") > 0))
        df = df.withColumn("listing_key", F.concat_ws("::", F.col("city"), F.col("id")))

        df = df.withColumn("price_clean", F.regexp_replace("price", r"[^0-9.\-]", "").cast("double"))
        df = df.filter(F.col("price_clean").between(cfg.price_min, cfg.price_max))
        df = df.withColumn("log_price", F.log("price_clean"))

        # Do NOT drop rows without text or image. Missing modality is represented
        # by availability masks in Gold; a URL alone is not proof an image exists.
        desc = F.trim(F.regexp_replace(F.regexp_replace(F.coalesce(F.col("description"), F.lit("")),
                                                        r"<[^>]+>", " "), r"\s+", " "))
        df = df.withColumn("_description_clean", desc)
        df = df.withColumn("_description_clean", F.when(
            F.lower("_description_clean").isin("null", "none", "n/a", "na", "unknown", "-"),
            F.lit("")).otherwise(F.col("_description_clean")))
        url = F.trim(F.coalesce(F.col("picture_url"), F.lit("")))
        df = df.withColumn("picture_url", F.when(url.rlike(r"(?i)^https://\S+$"), url)
                           .otherwise(F.lit(None).cast("string")))

        # Example: "1 and a half baths" -> 1.5, "1.5 baths" -> 1.5, "Half-bath" -> 0.5.
        bathroom_text = F.lower(F.coalesce(F.col("bathrooms_text"), F.lit("")))
        extracted = F.regexp_extract(bathroom_text, r"(\d+(?:\.\d+)?)", 1)
        has_number = F.length(extracted) > 0
        numeric = extracted.cast("double")
        df = df.withColumn("bathrooms",
            F.when(bathroom_text.rlike(r"\d+\s+and\s+a\s+half"), numeric + F.lit(0.5))
             .when(has_number, numeric)
             .when(bathroom_text.contains("half"), F.lit(0.5))
             .otherwise(F.col("bathrooms").cast("double")))

        # Parse amenities as JSON array, never split by comma within quoted names.
        amenity_array = F.from_json(F.col("amenities"), ArrayType(StringType()))
        df = df.withColumn("amenities_count", F.size(amenity_array).cast("double"))

        for name in cfg.num_cols:
            df = df.withColumn(name, F.col(name).cast("double"))
            df = df.withColumn(name,
                F.when(F.isnan(F.col(name)), F.lit(None).cast("double")).otherwise(F.col(name)))

        # Missing boolean must NOT silently become False.
        for name in cfg.cat_cols:
            value = F.trim(F.coalesce(F.col(name).cast("string"), F.lit("")))
            df = df.withColumn(name, F.when(F.length(value) == 0, F.lit("Unknown")).otherwise(value))

        # Filter empty text parts so fully missing text remains genuinely empty.
        parts = F.array(*[
            F.trim(F.regexp_replace(F.regexp_replace(
                F.coalesce(F.col(c), F.lit("")), r"<[^>]+>", " "), r"\s+", " "))
            for c in ("name", "_description_clean", "neighborhood_overview")])
        df = df.withColumn("clean_full_text", F.array_join(
            F.filter(parts, lambda s: F.length(s) > 0), " . "))
        return df.drop("_description_clean")
