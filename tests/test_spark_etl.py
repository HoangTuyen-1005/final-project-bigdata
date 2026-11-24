"""Requires pyspark + Java; runs locally on a development machine."""
import pytest

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from spark_etl.cleaner import AirbnbSparkCleaner
from spark_etl.feature_pipeline import SparkTabularFeaturizer


from spark_etl.spark_session import create_spark_session


@pytest.fixture(scope="module")
def spark():
    session = create_spark_session(app_name="test_airbnb_etl")
    yield session
    session.stop()



def test_cleaner_preserves_missing_numeric_and_parses_json(spark):
    source = spark.createDataFrame([
        {
            "id": "1",
            "city": "Boston",
            "price": "$125.00",
            "description": "<b>Lovely</b> bright apartment",
            "picture_url": "https://example.org/a.jpg",
            "bathrooms_text": "1 and a half baths",
            "amenities": '["Wifi, fast", "Pool"]',
            "bedrooms": None,
        },
        {
            "id": "2",
            "city": "Boston",
            "price": "$120.00",
            "description": "   ",
            "picture_url": "https://example.org/b.jpg",
            "bathrooms_text": "1 bath",
            "amenities": "[]",
            "bedrooms": "2",
        },
    ])
    rows = AirbnbSparkCleaner.clean_and_extract_features(source).collect()
    assert len(rows) == 2  # Missing text is retained for mask-aware Gold joins
    row = rows[0]
    assert row["bathrooms"] == 1.5
    assert row["amenities_count"] == 2
    assert row["bedrooms"] is None
    assert "Lovely bright apartment" in row["clean_full_text"]
    assert row["listing_key"] == "Boston::1"


def test_train_only_imputation_missing_flags_unseen_category(spark):
    rows = [
        ("Boston::1", "Boston", 2.0, [0.1, 0.2], [0.5, 0.6], 4.5),
        ("Boston::2", "Boston", None, [0.1, 0.2], [0.5, 0.6], 4.4),
        ("Boston::3", "Boston", 4.0, [0.1, 0.2], [0.5, 0.6], 4.3),
    ]
    schema = "listing_key string, city string, bedrooms double, x_txt array<double>, x_img array<double>, log_price double"
    train = spark.createDataFrame(rows, schema=schema)
    val = spark.createDataFrame([("Rome::1", "Rome", None, [0.1, 0.2], [0.5, 0.6], 5.0)], schema)
    test = spark.createDataFrame([("Berlin::1", "Berlin", 3.0, [0.1, 0.2], [0.5, 0.6], 4.6)], schema)
    featurizer = SparkTabularFeaturizer(["bedrooms"], ["city"], unknown_train_prob=0)
    out, _, cards = featurizer.fit_and_transform(train, val, test)
    assert cards == [2]  # Boston + unseen category
    assert out["train"].filter(F.col("listing_key") == "Boston::2").first()["x_num"][1] == 1.0
    assert out["val"].first()["x_cat"] == [1]
    assert out["test"].first()["x_num"][1] == 0.0


def test_cleaner_keeps_missing_images_for_left_join(spark):
    frame = spark.createDataFrame([
        {"id": "1", "city": "Boston", "price": "$120", "description": "Nice bedroom",
         "name": "Studio", "picture_url": "not-a-url"},
        {"id": "2", "city": "Boston", "price": "$130", "description": None,
         "name": None, "picture_url": None},
    ])
    rows = AirbnbSparkCleaner.clean_and_extract_features(frame).orderBy("id").collect()
    assert len(rows) == 2
    assert rows[0]["picture_url"] is None
    assert rows[1]["clean_full_text"] == ""


def test_spark_keyed_left_join_preserves_missing_modalities(spark):
    from spark_etl.run_etl import join_modalities
    raw = spark.createDataFrame([("city::1", "city"), ("city::2", "city"),
                                 ("city::3", "city")], "listing_key string, city string")
    txt = spark.createDataFrame([("city::1", [1.0, 2.0]),
                                 ("city::2", [3.0, 4.0])],
                                "listing_key string, embedding array<double>")
    img = spark.createDataFrame([("city::1", [1.0]), ("city::3", [2.0])],
                                "listing_key string, embedding array<double>")
    result = join_modalities(raw, txt, img, 2, 1).orderBy("listing_key").collect()
    assert [r["available_mask"] for r in result] == [
        [True, True, True], [True, True, False], [True, False, True]]
    assert result[1]["x_img"] == [0.0]
    assert result[2]["x_txt"] == [0.0, 0.0]
