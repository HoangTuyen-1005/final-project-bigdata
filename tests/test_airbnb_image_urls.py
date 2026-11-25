from contextlib import nullcontext
from dataclasses import replace
from io import BytesIO
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image

from airbnb_image_urls import resized_airbnb_image_url
from config import ModelConfig
from extract_embedding import SafeImageDownloader, run_image_phase


@pytest.mark.parametrize("source", [
    "https://a0.muscache.com/pictures/room.jpg",
    "https://a0.muscache.com/im/pictures/room.jpg?im_w=1920",
    "https://a0.muscache.com/im/pictures/room.jpg?aki_policy=x_large",
    "https://z1.muscache.cn/im/pictures/room.jpg?im_w=1920",
])
def test_optimized_cdn_urls(source):
    url = resized_airbnb_image_url(source, 720)
    assert urlsplit(url).path == "/im/pictures/room.jpg"
    assert parse_qs(urlsplit(url).query) == {"im_w": ["720"]}


def test_existing_thumbnail_is_preserved_and_other_parameters_remain():
    source = "https://a0.muscache.com/im/pictures/room.jpg?im_w=320&im_format=avif"
    capped_url = resized_airbnb_image_url(source, 720)
    assert parse_qs(urlsplit(capped_url).query) == {
        "im_w": ["320"], "im_format": ["avif"]
    }


def test_unrecognized_cdn_paths_are_not_rewritten():
    assert resized_airbnb_image_url("https://a0.muscache.com/unknown/room.jpg", 720) is None
    assert resized_airbnb_image_url("https://example.com/pictures/room.jpg", 720) is None


def test_downloader_caps_redirect_and_reuses_cache(tmp_path, monkeypatch):
    entry = "https://a0.muscache.com/pictures/old.jpg"
    redirect = "https://a0.muscache.com/im/pictures/new.jpg?im_w=1920"
    expected = [
        "https://a0.muscache.com/im/pictures/old.jpg?im_w=720",
        "https://a0.muscache.com/im/pictures/new.jpg?im_w=720",
    ]
    payload = BytesIO()
    with Image.new("RGB", (960, 480)) as image:
        image.save(payload, format="JPEG")
    body = payload.getvalue()
    responses = iter([
        SimpleNamespace(status_code=302, headers={"Location": redirect}),
        SimpleNamespace(
            status_code=200,
            headers={"Content-Type": "image/jpeg", "Content-Length": str(len(body))},
            iter_content=lambda chunk_size: [body],
        ),
    ])
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        assert kwargs["allow_redirects"] is False
        return nullcontext(next(responses))

    downloader = SafeImageDownloader(cache_dir=tmp_path)
    monkeypatch.setattr(downloader, "_check_public_https", lambda url: None)
    monkeypatch.setattr(downloader, "_session", lambda: SimpleNamespace(get=get))
    cached_alias = expected[0].replace("im_w=720", "aki_policy=x_large")
    for url in (entry, cached_alias):
        image = downloader.download(url)
        assert image is not None
        with image:
            assert image.size == (768, 384)
    assert calls == expected


def test_downloader_does_not_retry_original_when_resized_image_fails(tmp_path, monkeypatch):
    entry = "https://a0.muscache.com/pictures/room.jpg"
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return nullcontext(SimpleNamespace(status_code=404))

    downloader = SafeImageDownloader(cache_dir=tmp_path)
    monkeypatch.setattr(downloader, "_check_public_https", lambda url: None)
    monkeypatch.setattr(downloader, "_session", lambda: SimpleNamespace(get=get))
    assert downloader.download(entry) is None
    assert calls == ["https://a0.muscache.com/im/pictures/room.jpg?im_w=720"]


def test_clip_keeps_one_embedding_per_usable_cover(tmp_path, monkeypatch):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    splits = ("train", "val", "test")
    good, broken, zero = [f"https://example.org/{name}.jpg" for name in ("good", "broken", "zero")]
    for split in splits:
        silver = tmp_path / "silver" / split
        silver.mkdir(parents=True)
        parquet.write_table(arrow.table({
            "listing_key": [f"{split}::{index}" for index in range(4)],
            "picture_url": [good, None, broken, zero],
        }), silver / "part.parquet")

    class FakeEncoder:
        def __init__(self, model_name, device):
            pass

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, images, **kwargs):
            assert kwargs["normalize_embeddings"]
            return [[3.0, 4.0] if image.getpixel((0, 0))[0] else [0.0, 0.0] for image in images]

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeEncoder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []

    def download(self, url):
        calls.append(url)
        if url == broken:
            return None
        return Image.new("RGB", (32, 32), color="white" if url == good else "black")

    monkeypatch.setattr(SafeImageDownloader, "download", download)
    cfg = replace(ModelConfig(), raw_image_dim=2)
    run_image_phase({"run_path": tmp_path, "run_id": "cover-demo"}, cfg,
                    str(tmp_path), "fake-clip", workers=1)

    assert calls == [good, broken, zero] * len(splits)
    for split in splits:
        records = parquet.read_table(tmp_path / "embeddings" / f"{split}_image.parquet").to_pylist()
        assert len(records) == 1
        assert records[0]["listing_key"] == f"{split}::0"
        assert records[0]["run_id"] == "cover-demo"
        assert records[0]["embedding"] == pytest.approx([0.6, 0.8])
