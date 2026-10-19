from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import Imputer, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class SparkTabularFeaturizer:
    """Pipeline MLlib chuẩn hóa Numeric và mã hóa Categorical không gây Data Leakage."""

    def __init__(self, num_cols: list[str], cat_cols: list[str]) -> None:
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.imputed_num_cols = [f"{c}_imputed" for c in num_cols]
        self.indexed_cat_cols = [f"{c}_idx" for c in cat_cols]

    def build_pipeline(self) -> Pipeline:
        # Bước 1: Điền giá trị khuyết (Missing Values) bằng Trung vị (Median) trên phân tán
        imputer = Imputer(
            inputCols=self.num_cols,
            outputCols=self.imputed_num_cols,
            strategy="median",
        )

        # Bước 2: Gom 13 cột số đã điền khuyết thành 1 Vector để chuẩn hóa đồng loạt
        assembler = VectorAssembler(
            inputCols=self.imputed_num_cols,
            outputCol="num_features_raw_vec",
            handleInvalid="keep",
        )

        # Bước 3: Chuẩn hóa Z-score (mean=0, std=1) — bắt buộc đối với FT-Transformer
        scaler = StandardScaler(
            inputCol="num_features_raw_vec",
            outputCol="num_features_scaled_vec",
            withMean=True,
            withStd=True,
        )

        # Bước 4: Mã hóa 5 cột chữ sang chỉ số nguyên (0, 1, 2...).
        # handleInvalid="keep" tự động gán các nhãn mới (Unseen labels ở tập Test) vào 1 bucket riêng!
        indexers = [
            StringIndexer(
                inputCol=c,
                outputCol=f"{c}_idx",
                handleInvalid="keep",
            )
            for c in self.cat_cols
        ]

        return Pipeline(stages=[imputer, assembler, scaler, *indexers])

    def fit_and_transform(
        self, train_df: DataFrame, test_df: DataFrame
    ) -> tuple[DataFrame, DataFrame, PipelineModel, list[int]]:
        """Chỉ fit Pipeline trên tập Train, sau đó transform cho cả Train và Test."""
        pipeline = self.build_pipeline()
        fitted_model = pipeline.fit(train_df)

        train_transformed = self._extract_final_columns(fitted_model.transform(train_df))
        test_transformed = self._extract_final_columns(fitted_model.transform(test_df))

        # Lấy số lượng nhãn (Cardinality) của từng cột Categorical để truyền vào nn.Embedding của PyTorch
        # Cộng thêm 1 cho bucket handleInvalid="keep"
        cat_cardinalities = [
            len(stage.labels) + 1
            for stage in fitted_model.stages
            if hasattr(stage, "labels")
        ]

        return train_transformed, test_transformed, fitted_model, cat_cardinalities

    def _extract_final_columns(self, df: DataFrame) -> DataFrame:
        """Tách Vector đã chuẩn hóa thành mảng (Array) để lưu xuống Parquet gọn nhẹ."""
        df_out = df.withColumn(
            "x_num", vector_to_array(F.col("num_features_scaled_vec"))
        ).withColumn(
            "x_cat",
            F.array(*[F.col(c).cast("long") for c in self.indexed_cat_cols]),
        )

        return df_out.select(
            "id",
            "x_num",
            "x_cat",
            "clean_full_text",
            "picture_url",
            F.col("log_price").alias("y"),
        )