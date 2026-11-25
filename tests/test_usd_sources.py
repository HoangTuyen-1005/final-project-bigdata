import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession
from spark_etl.cleaner import AirbnbSparkCleaner
from spark_etl.run_etl import resolve_sources


def test_resolve_us_city_sources(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    boston = raw / "Boston" / "listings.csv.gz"
    chicago = raw / "Chicago" / "listings.csv"
    for path in (boston, chicago):
        path.parent.mkdir(parents=True)
        path.touch()
    (raw / "README.md").write_text("Input folder guide", encoding="utf-8")
    (boston.parent / "listings.csv.bak").touch()
    monkeypatch.chdir(tmp_path)
    assert resolve_sources([]) == [("Boston", str(boston)), ("Chicago", str(chicago))]
    assert resolve_sources([f"Chicago={chicago}"]) == [("Chicago", str(chicago))]
    assert resolve_sources([f"Boston={boston}", f"Chicago={chicago}"]) == [
        ("Boston", str(boston)), ("Chicago", str(chicago))
    ]
    with pytest.raises(ValueError, match="one snapshot per city"):
        resolve_sources([f"Boston={boston}", f"boston={chicago}"])
    with pytest.raises(ValueError, match="assigned to multiple cities"):
        resolve_sources([f"Boston={boston}", f"Chicago={boston}"])

    boston.with_suffix("").touch()
    with pytest.raises(ValueError, match="one snapshot per city"):
        resolve_sources([])
    assert resolve_sources([f"Boston={boston}"]) == [("Boston", str(boston))]


def test_flat_source_requires_a_city_label(tmp_path):
    path = tmp_path / "listings.csv"
    path.touch()
    with pytest.raises(FileNotFoundError, match="No labeled US listings"):
        resolve_sources([], tmp_path)
    assert resolve_sources([f"Chicago={path}"], tmp_path) == [("Chicago", str(path))]


def test_usd_prices_and_city_scoped_ids():
    spark = SparkSession.builder.master("local[1]").appName("test-usd-prices").getOrCreate()
    try:
        source = spark.createDataFrame([
            ("1", "Boston", "$1,200.50", None),
            ("1", "Chicago", "USD 150.25", "USD"),
            ("2", "Chicago", "99.99", "usd"),
            ("3", "Chicago", "EUR 90.00", None),
            ("4", "Chicago", "$90.00", "CAD"),
            ("5", "Boston", "120oops", None),
        ], "id string, city string, price string, currency string")
        rows = AirbnbSparkCleaner.clean_and_extract_features(source).select(
            "listing_key", "price_clean"
        ).collect()
        assert {row.listing_key: row.price_clean for row in rows} == {
            "Boston::1": 1200.5, "Chicago::1": 150.25, "Chicago::2": 99.99
        }
    finally:
        spark.stop()
