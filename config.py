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
    # (Multilingual-E5-base = 768, CLIP-ViT-B/32 = 512)
    raw_text_dim: int = 768
    raw_image_dim: int = 512

    # Xác suất che mờ ngẫu nhiên 1 nhánh khi train để chống "Liệt nhánh Ảnh/Văn bản"
    modality_dropout_prob: float = 0.20

    # Tham số huấn luyện
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    seed: int = 42

    # Giới hạn số lượng mẫu phân tầng cho mỗi thành phố (Stratified Sampling)
    # Giúp cân bằng đại diện giữa các thành phố và tối ưu hóa thời gian tải ảnh/trích xuất embedding
    max_samples_per_city: int = 4000

    # Danh sách các cột Số (Numeric):
    # - minimum_nights_capped: Đã giới hạn tại 30 đêm nhằm triệt tiêu outliers (365, 1125 đêm)
    # - is_long_term_stay: Cờ nhị phân (1.0 nếu >= 30 đêm, 0.0 nếu < 30 đêm) đại diện cho quy định thuê dài hạn
    num_cols: list[str] = field(
        default_factory=lambda: [
            "latitude",
            "longitude",
            "accommodates",
            "bedrooms",
            "beds",
            "bathrooms",
            "amenities_count",
            "calculated_host_listings_count",
            "minimum_nights_capped",
            "is_long_term_stay",
            "number_of_reviews",
            "review_scores_rating",
        ]
    )

    # Danh sách các cột Phân loại (Categorical):
    # - city_name: Tự động trích xuất từ tên file để mô hình học context địa lý và chính sách riêng biệt từng thành phố
    cat_cols: list[str] = field(
        default_factory=lambda: [
            "neighbourhood_cleansed",
            "property_type",
            "room_type",
            "host_is_superhost",
            "instant_bookable",
            "city_name",
        ]
    )
