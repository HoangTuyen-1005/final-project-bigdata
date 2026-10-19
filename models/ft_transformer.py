import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Biến đổi từng cột Numeric và Categorical thành các Token Embeddings."""

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list[int],
        d_model: int,
    ) -> None:
        super().__init__()
        self.num_continuous = num_continuous

        # Trọng số tuyến tính riêng cho từng cột Numeric: x_i * W_i + b_i
        if num_continuous > 0:
            self.num_weights = nn.Parameter(torch.empty(num_continuous, d_model))
            self.num_biases = nn.Parameter(torch.empty(num_continuous, d_model))
            nn.init.xavier_uniform_(self.num_weights)
            nn.init.zeros_(self.num_biases)

        # Bảng Lookup Embedding riêng cho từng cột Categorical
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_model) for cardinality in cat_cardinalities]
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens: list[torch.Tensor] = []

        if self.num_continuous > 0:
            # x_num: (B, N_num) -> (B, N_num, 1) * (1, N_num, D) -> (B, N_num, D)
            num_tokens = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)
            tokens.append(num_tokens)

        if len(self.cat_embeddings) > 0:
            cat_tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)],
                dim=1,
            )  # (B, N_cat, D)
            tokens.append(cat_tokens)

        return torch.cat(tokens, dim=1)  # (B, N_total_cols, D)


class FTTransformerEncoder(nn.Module):
    """Bộ mã hóa FT-Transformer xuất ra vector đại diện [CLS] cho nhánh Tabular."""

    def __init__(
        self,
        num_continuous: int,
        cat_cardinalities: list[int],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ffn_factor: float = 1.33,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_continuous, cat_cardinalities, d_model)

        # Token [CLS] học thông tin tổng hợp của toàn bộ 18 cột Tabular
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * d_ffn_factor),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm giúp hội tụ ổn định hơn
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.size(0)
        feature_tokens = self.tokenizer(x_num, x_cat)  # (B, N_cols, D)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, feature_tokens], dim=1)  # (B, 1 + N_cols, D)

        x = self.transformer(x)
        cls_out = self.norm(x[:, 0, :])  # Chỉ lấy đầu ra của token [CLS]: (B, D)
        return cls_out