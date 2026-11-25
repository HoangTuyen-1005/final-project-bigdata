from pathlib import Path

from contracts.state import resolve_data_root
from contracts.storage import is_s3_uri, join_uri, parse_s3_uri, to_s3a_uri


def test_s3_uri_helpers():
    assert is_s3_uri("s3://bucket/path")
    assert is_s3_uri("s3a://bucket/path")
    assert not is_s3_uri("data/runs")
    assert to_s3a_uri("s3://bucket/a/b") == "s3a://bucket/a/b"
    assert parse_s3_uri("s3a://airbnb-lakehouse/runs/r1") == ("airbnb-lakehouse", "runs/r1")
    assert join_uri("s3a://airbnb-lakehouse/", "/demo", "runs/") == "s3a://airbnb-lakehouse/demo/runs"


def test_resolve_data_root_switches_between_local_and_s3(monkeypatch, tmp_path):
    monkeypatch.delenv("STORAGE_BACKEND", raising=False)
    monkeypatch.delenv("S3_ENDPOINT", raising=False)
    assert resolve_data_root(tmp_path) == tmp_path.resolve()

    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    monkeypatch.setenv("S3_BUCKET", "airbnb-lakehouse")
    assert resolve_data_root("/data") == "s3a://airbnb-lakehouse"
    assert resolve_data_root("/data/demo") == "s3a://airbnb-lakehouse/demo"
    assert resolve_data_root(Path("/data/demo")) == "s3a://airbnb-lakehouse/demo"
