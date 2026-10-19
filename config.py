from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    # Kích thước vector chung cho cả 3 nhánh sau khi chiếu (Projection)
    d_model: int = 128

    # Tham số cho FT-Transformer (Nhánh Tabular)
    n_heads: int = 4
    n_layers: int = 3
    d_ffn_factor: float = 1.33
    transformer_dropout: float = 0.1

    # Kích thước vector gốc từ Pre-trained Text & Image Encoders
    # (VD: Multilingual-E5-base = 768, CLIP-ViT-B/32 = 512)
    raw_text_dim: int = 768
    raw_image_dim: int = 512

    # Xác suất che mờ ngẫu nhiên 1 nhánh khi train để chống "Liệt nhánh Ảnh"
    modality_dropout_prob: float = 0.20

    # Tham số huấn luyện
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    seed: int = 42

    # Danh sách các cột Số (Numeric) và Phân loại (Categorical) từ Inside Airbnb
    num_cols: list[str] = field(
        default_factory=lambda: [
            "latitude",
            "longitude",
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "amenities_count",
            # TODO: Tạm thời bỏ qua 2 cột tỷ lệ của Host do tập dữ liệu mẫu bị khuyết 100%.
            # Không thể điền 0.0 vì gây sai lệch bản chất dữ liệu, cũng không thể drop dòng vì sẽ làm mất 100% dữ liệu.
            # Sau này sẽ thu thập dữ liệu lớn hơn hoặc dùng mô hình Imputation (MICE/KNN) để xử lý sau.
            # "host_response_rate",
            # "host_acceptance_rate",
            "calculated_host_listings_count",
            "minimum_nights",
            "number_of_reviews",
            "review_scores_rating",
        ]
    )

    cat_cols: list[str] = field(
        default_factory=lambda: [
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "host_is_superhost",
            "instant_bookable",
        ]
    )