"""Train-fitted Spark MLlib preprocessing; vectors and missing flags stored in Gold."""
from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import numeric_feature_names


class SparkTabularFeaturizer:
    def __init__(self, num_cols: list[str], cat_cols: list[str],
                 impute_strategy: str = "median", unknown_train_prob: float = 0.05,
                 seed: int = 42) -> None:
        if not 0 <= unknown_train_prob < 1:
            raise ValueError("unknown_train_prob must be in [0, 1).")
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.impute_strategy = impute_strategy
        self.unknown_train_prob = unknown_train_prob
        self.seed = seed
        self.imputed_num_cols = [f"{c}__imputed" for c in num_cols]
        self.indexed_cat_cols = [f"{c}__idx" for c in cat_cols]

    @property
    def num_feature_names(self) -> list[str]:
        return numeric_feature_names(self.num_cols)

    def build_pipeline(self) -> Pipeline:
        imputer = Imputer(inputCols=self.num_cols, outputCols=self.imputed_num_cols,
                          strategy=self.impute_strategy)
        assembler = VectorAssembler(inputCols=self.imputed_num_cols,
                                    outputCol="num_features_raw_vec", handleInvalid="error")
        scaler = StandardScaler(inputCol="num_features_raw_vec",
                                outputCol="num_features_scaled_vec", withMean=True, withStd=True)
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}__idx", handleInvalid="keep")
                    for c in self.cat_cols]
        return Pipeline(stages=[imputer, assembler, scaler, *indexers])

    def fit_and_transform(self, train_df: DataFrame, val_df: DataFrame,
                          test_df: DataFrame) -> tuple[dict[str, DataFrame], PipelineModel, list[int]]:
        # Fail with an actionable error if median/mean is undefined on the training split.
        counts = train_df.agg(*[F.count(F.col(c)).alias(c) for c in self.num_cols]).first().asDict()
        missing_all = [name for name, count in counts.items() if count == 0]
        if missing_all:
            raise ValueError("Numeric columns are entirely missing in valid Train data: "
                             f"{missing_all}. Remove them from ModelConfig.num_cols or supply data.")

        model = self.build_pipeline().fit(train_df)
        cardinalities = [len(stage.labels) + 1 for stage in model.stages if hasattr(stage, "labels")]
        if len(cardinalities) != len(self.cat_cols):
            raise RuntimeError("Unexpected number of StringIndexer stages")
        output = {
            "train": self._extract_final_columns(model.transform(train_df), cardinalities, is_train=True),
            "val": self._extract_final_columns(model.transform(val_df), cardinalities),
            "test": self._extract_final_columns(model.transform(test_df), cardinalities),
        }
        return output, model, cardinalities

    def export_preprocessor_stats(self, model: PipelineModel) -> dict:
        """Export Train-fitted imputer, scaler, and indexer stats for Spark-free inference."""
        imputer_model, _, scaler_model, *indexer_models = model.stages
        surrogate_row = imputer_model.surrogateDF.first().asDict()
        num_medians = {col: float(surrogate_row[col]) for col in self.num_cols if col in surrogate_row}
        cat_labels = {
            col: list(stage.labels)
            for col, stage in zip(self.cat_cols, indexer_models, strict=True)
        }
        return {
            "num_medians": num_medians,
            "num_means": [float(x) for x in scaler_model.mean.toArray()],
            "num_stds": [float(x) for x in scaler_model.std.toArray()],
            "cat_labels": cat_labels,
        }

    def _extract_final_columns(self, df: DataFrame, cardinalities: list[int],
                               is_train: bool = False) -> DataFrame:
        missing_flags = [F.when(F.col(c).isNull() | F.isnan(F.col(c)), 1.0)
                         .otherwise(0.0).cast("double") for c in self.num_cols]
        df = df.withColumn("x_num", F.concat(vector_to_array("num_features_scaled_vec"),
                                               F.array(*missing_flags)))
        indexed = []
        for i, c in enumerate(self.indexed_cat_cols):
            encoded = F.col(c).cast("long")
            if is_train and self.unknown_train_prob > 0:
                # Train the StringIndexer 'unknown' embedding using real training examples.
                # Stable deterministic hash, different for each feature, independent of partitions.
                fraction = F.pmod(F.xxhash64(F.col("listing_key"), F.lit(c), F.lit(self.seed)),
                                  F.lit(100000)) / F.lit(100000.0)
                encoded = F.when(fraction < self.unknown_train_prob,
                                 F.lit(cardinalities[i] - 1).cast("long")).otherwise(encoded)
            indexed.append(encoded)
        df = df.withColumn("x_cat", F.array(*indexed))
        # The production join always supplies this field. The fallback is useful
        # for standalone featurizer tests on fully observed toy inputs.
        if "available_mask" not in df.columns:
            df = df.withColumn("available_mask", F.array(F.lit(True), F.lit(True), F.lit(True)))
        return df.select("listing_key", "city", "x_num", "x_cat", "x_txt", "x_img",
                         "available_mask", F.col("log_price").alias("y"))
