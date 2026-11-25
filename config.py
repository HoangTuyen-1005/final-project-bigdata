"""Shared configuration; CLI flags and environment settings override deployment details."""
from dataclasses import dataclass, field


MODALITY_NAMES = ("tabular", "text", "image")
PRICE_CURRENCY = "USD"


def numeric_feature_names(columns: list[str]) -> list[str]:
    """Match Gold vector order: scaled values followed by missing indicators."""
    return [*columns, *[f"{column}__missing" for column in columns]]


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ffn_factor: float = 1.33
    transformer_dropout: float = 0.1
    raw_text_dim: int = 768
    raw_image_dim: int = 512
    modality_dropout_prob: float = 0.20  # symmetric masking across all available channels
    auxiliary_loss_weight: float = 0.0  # optional per-modality supervision; no forced gate weights
    batch_size: int = 32  # conservative default for 6 GB laptop GPUs
    text_batch_size: int = 16
    image_batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    early_stopping_patience: int = 5
    seed: int = 42
    price_min: float = 10.0
    price_max: float = 5000.0
    missing_numeric_strategy: str = "median"  # or mean; fitted on Train only
    unknown_category_train_prob: float = 0.05
    num_cols: list[str] = field(default_factory=lambda: [
        "latitude", "longitude", "accommodates", "bedrooms", "beds",
        "bathrooms", "amenities_count", "calculated_host_listings_count",
        "minimum_nights", "number_of_reviews", "review_scores_rating",
    ])
    cat_cols: list[str] = field(default_factory=lambda: [
        "city", "neighbourhood_cleansed", "property_type", "room_type",
        "host_is_superhost", "instant_bookable",
    ])

    @property
    def num_feature_names(self) -> list[str]:
        return numeric_feature_names(self.num_cols)
