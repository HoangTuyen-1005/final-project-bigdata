from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import ModelConfig


class MultimodalFeatureExtractor:
    """
    Trích xuất Offline Embeddings cho Text (768-d) và Image (512-d) trực tiếp từ Parquet.
    Đảm bảo thứ tự dòng (Row Alignment) khớp tuyệt đối 100% với nhánh Tabular của PySpark.
    """

    def __init__(
        self,
        text_model_name: str = "intfloat/multilingual-e5-base",  # Xuất vector 768 chiều
        image_model_name: str = "clip-ViT-B-32",                 # Xuất vector 512 chiều
        batch_size: int = 128,
        max_download_workers: int = 32,
        request_timeout: float = 5.0,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.max_workers = max_download_workers
        self.timeout = request_timeout

        print(f"Khởi tạo Pre-trained Encoders trên thiết bị: [{self.device.upper()}]...")
        self.text_encoder = SentenceTransformer(text_model_name, device=self.device)
        self.image_encoder = SentenceTransformer(image_model_name, device=self.device)

        # Lấy kích thước chiều thực tế từ model để khởi tạo vector 0 khi ảnh lỗi
        self.text_dim = self.text_encoder.get_sentence_embedding_dimension() or 768
        self.img_dim = 512  # Chuẩn đầu ra của clip-ViT-B-32

        # Tái sử dụng HTTP Session để tăng tốc độ bắt tay TCP/TLS khi tải hàng chục nghìn ảnh
        self.http_session = requests.Session()
        self.http_session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )

    def _download_single_image(self, url: str) -> Image.Image | None:
        """Tải 1 ảnh từ URL về RAM. Trả về None nếu link chết (404/403) hoặc ảnh hỏng."""
        if not isinstance(url, str) or not url.startswith("http"):
            return None
        try:
            response = self.http_session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                # Loại bỏ các ảnh tracking pixel rỗng (kích thước quá nhỏ < 32x32)
                if img.width >= 32 and img.height >= 32:
                    return img
        except Exception:
            pass
        return None

    def extract_text_embeddings(self, texts: list[str]) -> np.ndarray:
        """Mã hóa danh sách văn bản thành ma trận (N, 768) đã chuẩn hóa L2."""
        print(f"-> Đang mã hóa {len(texts):,} đoạn văn bản (Text Branch)...")
        # Với dòng mô hình E5, thêm tiền tố 'passage: ' giúp tối ưu chất lượng biểu diễn
        cleaned_texts = [
            f"passage: {t}" if isinstance(t, str) and len(t.strip()) > 0 else "passage: none"
            for t in texts
        ]

        embeddings = self.text_encoder.encode(
            cleaned_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Chuẩn hóa L2 giúp FT-Transformer Gating học ổn định
        )
        return embeddings.astype(np.float32)

    def extract_image_embeddings(self, urls: list[str]) -> np.ndarray:
        """
        Tải ảnh đa luồng theo từng chunk và mã hóa bằng CLIP thành ma trận (N, 512).
        Nếu URL ảnh bị chết, tự động điền bằng vector 0 (Zero-vector fallback).
        """
        n_samples = len(urls)
        all_embeddings = np.zeros((n_samples, self.img_dim), dtype=np.float32)
        chunk_size = self.batch_size * 4  # Tải trước 512 ảnh trên RAM mỗi đợt

        valid_count = 0
        print(f"-> Đang tải đa luồng ({self.max_workers} threads) & mã hóa {n_samples:,} ảnh...")

        for start_idx in tqdm(range(0, n_samples, chunk_size), desc="Image Batches"):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_urls = urls[start_idx:end_idx]

            # 1. Tải song song toàn bộ ảnh trong chunk bằng ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                downloaded_images = list(executor.map(self._download_single_image, chunk_urls))

            # 2. Lọc ra các vị trí tải ảnh thành công để đưa vào GPU encode một lượt
            valid_images: list[Image.Image] = []
            valid_global_indices: list[int] = []

            for local_i, img in enumerate(downloaded_images):
                if img is not None:
                    valid_images.append(img)
                    valid_global_indices.append(start_idx + local_i)

            # 3. Đưa các ảnh hợp lệ qua mạng CLIP (những vị trí lỗi mặc định giữ nguyên vector 0)
            if valid_images:
                valid_count += len(valid_images)
                chunk_embs = self.image_encoder.encode(
                    valid_images,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                all_embeddings[valid_global_indices] = chunk_embs.astype(np.float32)

            # Giải phóng bộ nhớ ảnh PIL sau mỗi chunk
            for img in downloaded_images:
                if img is not None:
                    img.close()

        success_rate = (valid_count / n_samples) * 100 if n_samples > 0 else 0.0
        print(
            f"   [Thống kê Ảnh] Thành công: {valid_count:,}/{n_samples:,} ({success_rate:.1f}%). "
            f"Ảnh lỗi đã gán Zero-Vector: {n_samples - valid_count:,}."
        )
        return all_embeddings

    def process_split(self, parquet_path: Path, output_prefix: Path) -> None:
        """Đọc 1 tập Parquet (train hoặc test), trích xuất và lưu ra 2 file .npy tương ứng."""
        print(f"\n========================================================")
        print(f"Đang xử lý file Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path, columns=["clean_full_text", "picture_url"])
        n_rows = len(df)

        # 1. Trích xuất và lưu Text Embeddings
        text_emb = self.extract_text_embeddings(df["clean_full_text"].tolist())
        assert text_emb.shape == (n_rows, self.text_dim), "Sai lệch kích thước Text Embedding!"
        text_out_path = Path(f"{output_prefix}_text_emb.npy")
        np.save(text_out_path, text_emb)
        print(f"-> Đã lưu Text Embeddings {text_emb.shape} tại: {text_out_path}")

        # 2. Trích xuất và lưu Image Embeddings
        img_emb = self.extract_image_embeddings(df["picture_url"].tolist())
        assert img_emb.shape == (n_rows, self.img_dim), "Sai lệch kích thước Image Embedding!"
        img_out_path = Path(f"{output_prefix}_img_emb.npy")
        np.save(img_out_path, img_emb)
        print(f"-> Đã lưu Image Embeddings {img_emb.shape} tại: {img_out_path}")


def main() -> None:
    cfg = ModelConfig()
    gold_dir = Path("data/gold_parquet")

    extractor = MultimodalFeatureExtractor(
        batch_size=cfg.batch_size,
        max_download_workers=32,
    )

    # Chạy lần lượt cho tập Train và tập Test do PySpark sinh ra
    for split_name in ["train", "test"]:
        parquet_file = gold_dir / f"{split_name}.parquet"
        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {parquet_file}! Hãy chạy 'python -m spark_etl.run_etl' trước."
            )
        extractor.process_split(
            parquet_path=parquet_file,
            output_prefix=gold_dir / split_name,
        )

    print("\nHoàn tất trích xuất toàn bộ Embeddings! Sẵn sàng chạy 'python main.py'.")


if __name__ == "__main__":
    main()