import torch
import torch.nn as nn

from config import ModelConfig
from models.ft_transformer import FTTransformerEncoder


class GatedMultimodalFusion(nn.Module):
    """Hợp nhất 3 nhánh bằng cơ chế Gating có trọng số tự động + Modality Dropout."""

    def __init__(self, d_model: int, modality_dropout_prob: float = 0.2) -> None:
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob

        # Mạng tính điểm quan trọng (Gate logits) cho 3 nhánh dựa trên ngữ cảnh chung
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def _apply_modality_dropout(
        self, h_tab: torch.Tensor, h_txt: torch.Tensor, h_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Che mờ ngẫu nhiên nhánh Tabular hoặc Text khi train để bắt mô hình học Ảnh."""
        if not self.training or self.modality_dropout_prob <= 0.0:
            return h_tab, h_txt, h_img

        batch_size = h_tab.size(0)
        device = h_tab.device

        # Tạo mặt nạ giữ lại (1 = giữ, 0 = che mờ) cho từng sample trong batch
        mask_tab = (torch.rand(batch_size, 1, device=device) > self.modality_dropout_prob).float()
        mask_txt = (torch.rand(batch_size, 1, device=device) > self.modality_dropout_prob).float()

        return h_tab * mask_tab, h_txt * mask_txt, h_img

    def forward(
        self, h_tab: torch.Tensor, h_txt: torch.Tensor, h_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Áp dụng Modality Dropout (chỉ kích hoạt khi model.train())
        h_tab, h_txt, h_img = self._apply_modality_dropout(h_tab, h_txt, h_img)

        # 2. Tính trọng số cổng alpha_tab, alpha_txt, alpha_img (tổng = 1.0)
        concat_features = torch.cat([h_tab, h_txt, h_img], dim=-1)  # (B, 3*D)
        gate_logits = self.gate_net(concat_features)  # (B, 3)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, 3)

        # 3. Nhân trọng số cổng với từng nhánh và cộng gộp
        stacked = torch.stack([h_tab, h_txt, h_img], dim=1)  # (B, 3, D)
        fused = torch.sum(stacked * gate_weights.unsqueeze(-1), dim=1)  # (B, D)

        return fused, gate_weights


class AirbnbMultimodalRegressor(nn.Module):
    """Mô hình End-to-End kết hợp FT-Transformer + Text/Image Projections + Gated Fusion."""

    def __init__(self, cfg: ModelConfig, cat_cardinalities: list[int]) -> None:
        super().__init__()
        self.tabular_encoder = FTTransformerEncoder(
            num_continuous=len(cfg.num_cols),
            cat_cardinalities=cat_cardinalities,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ffn_factor=cfg.d_ffn_factor,
            dropout=cfg.transformer_dropout,
        )

        # Chiếu vector Text (768) và Image (512) về cùng không gian d_model (128)
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.raw_text_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(cfg.raw_image_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

        self.fusion = GatedMultimodalFusion(
            d_model=cfg.d_model,
            modality_dropout_prob=cfg.modality_dropout_prob,
        )

        self.regressor_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_txt: torch.Tensor,
        x_img: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_tab = self.tabular_encoder(x_num, x_cat)
        h_txt = self.text_proj(x_txt)
        h_img = self.image_proj(x_img)

        fused, gate_weights = self.fusion(h_tab, h_txt, h_img)
        pred = self.regressor_head(fused).squeeze(-1)
        return pred, gate_weights