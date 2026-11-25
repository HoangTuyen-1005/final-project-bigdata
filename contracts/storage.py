"""Unified storage abstraction supporting local filesystem and S3/MinIO (s3a://)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "local"  # "local" or "s3"
    endpoint: str = "http://minio:9000"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket: str = "airbnb-lakehouse"
    region: str = "us-east-1"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        backend = os.environ.get("STORAGE_BACKEND", "").strip().lower()
        if not backend:
            backend = "s3" if os.environ.get("S3_ENDPOINT") else "local"
        return cls(
            backend=backend,
            endpoint=os.environ.get("S3_ENDPOINT", "http://minio:9000"),
            access_key=os.environ.get("S3_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("S3_SECRET_KEY", "minioadmin"),
            bucket=os.environ.get("S3_BUCKET", "airbnb-lakehouse"),
            region=os.environ.get("S3_REGION", "us-east-1"),
        )


def is_s3_uri(path: str | Path) -> bool:
    text = str(path)
    return text.startswith("s3://") or text.startswith("s3a://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme not in ("s3", "s3a") or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def to_s3a_uri(uri: str | Path) -> str:
    text = str(uri)
    if text.startswith("s3://"):
        return "s3a://" + text[len("s3://"):]
    return text


def join_uri(base: str | Path, *parts: str) -> str:
    clean_parts = [p.strip("/") for p in parts if p and p.strip("/")]
    if is_s3_uri(base):
        root = str(base).rstrip("/")
        return "/".join([root, *clean_parts]) if clean_parts else root
    path = Path(base)
    for part in clean_parts:
        path = path / part
    return str(path)


def _s3_client(cfg: StorageConfig | None = None):
    import boto3
    from botocore.config import Config

    cfg = cfg or StorageConfig.from_env()
    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name=cfg.region,
        config=Config(s3={"addressing_style": "path"}),
    )


def uri_exists(uri: str | Path, cfg: StorageConfig | None = None) -> bool:
    if is_s3_uri(uri):
        from botocore.exceptions import ClientError

        bucket, key = parse_s3_uri(str(uri))
        client = _s3_client(cfg)
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
    return Path(uri).is_file()


def read_text(uri: str | Path, encoding: str = "utf-8", cfg: StorageConfig | None = None) -> str:
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(str(uri))
        obj = _s3_client(cfg).get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode(encoding)
    return Path(uri).read_text(encoding=encoding)


def write_text(uri: str | Path, content: str, encoding: str = "utf-8",
               cfg: StorageConfig | None = None) -> None:
    write_bytes(uri, content.encode(encoding), cfg=cfg)


def write_bytes(uri: str | Path, data: bytes, cfg: StorageConfig | None = None) -> None:
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(str(uri))
        _s3_client(cfg).put_object(Bucket=bucket, Key=key, Body=data)
        return
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def read_bytes(uri: str | Path, cfg: StorageConfig | None = None) -> bytes:
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(str(uri))
        obj = _s3_client(cfg).get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    return Path(uri).read_bytes()


def list_s3_keys(prefix_uri: str, cfg: StorageConfig | None = None) -> list[str]:
    bucket, prefix = parse_s3_uri(prefix_uri)
    client = _s3_client(cfg)
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            keys.append(f"s3a://{bucket}/{item['Key']}")
    return sorted(keys)


def has_any_objects(uri: str | Path, cfg: StorageConfig | None = None) -> bool:
    if is_s3_uri(uri):
        bucket, prefix = parse_s3_uri(str(uri))
        prefix = prefix.rstrip("/") + "/"
        resp = _s3_client(cfg).list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return resp.get("KeyCount", 0) > 0
    path = Path(uri)
    return path.exists() and any(path.iterdir())


def upload_file(local_path: str | Path, uri: str | Path, cfg: StorageConfig | None = None) -> None:
    if is_s3_uri(uri):
        bucket, key = parse_s3_uri(str(uri))
        _s3_client(cfg).upload_file(str(local_path), bucket, key)
        return
    target = Path(uri)
    target.parent.mkdir(parents=True, exist_ok=True)
    Path(local_path).replace(target)


def pyarrow_fs_and_path(uri: str | Path, cfg: StorageConfig | None = None):
    """Return (pyarrow.fs.FileSystem | None, path_string) for local or S3/MinIO URIs."""
    if not is_s3_uri(uri):
        return None, str(uri)
    from pyarrow.fs import S3FileSystem

    cfg = cfg or StorageConfig.from_env()
    parsed = urlparse(cfg.endpoint)
    endpoint = parsed.netloc or cfg.endpoint
    scheme = parsed.scheme or "http"
    fs = S3FileSystem(
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        endpoint_override=endpoint,
        scheme=scheme,
        region=cfg.region,
        force_virtual_addressing=False,
    )
    bucket, key = parse_s3_uri(str(uri))
    return fs, f"{bucket}/{key}".rstrip("/")


def read_parquet_df(uri: str | Path, columns: list[str] | None = None,
                    cfg: StorageConfig | None = None):
    import pandas as pd

    if not is_s3_uri(uri):
        return pd.read_parquet(uri, columns=columns)
    import pyarrow.dataset as ds

    fs, pa_path = pyarrow_fs_and_path(uri, cfg)
    dataset = ds.dataset(pa_path, filesystem=fs, format="parquet")
    return dataset.to_table(columns=columns).to_pandas()

