"""FT-Transformer, CLIP/E5 projections, and correctly masked gated fusion."""
import torch
import torch.nn as nn
from config import ModelConfig
from models.ft_transformer import FTTransformerEncoder


class GatedMultimodalFusion(nn.Module):
    def __init__(self, d_model: int, modality_dropout_prob: float = 0.2) -> None:
        super().__init__()
        self.modality_dropout_prob = modality_dropout_prob
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.GELU(), nn.Linear(d_model, 3))

    def forward(self, h_tab: torch.Tensor, h_txt: torch.Tensor,
                h_img: torch.Tensor, available_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = h_tab.size(0)
        if available_mask is None:
            mask = torch.ones((batch, 3), dtype=torch.bool, device=h_tab.device)
        else:
            mask = available_mask.to(device=h_tab.device, dtype=torch.bool)
            if mask.shape != (batch, 3):
                raise ValueError(f"available_mask must be {(batch, 3)}, got {tuple(mask.shape)}")
        if not mask.any(dim=1).all():
            raise ValueError("Every sample must retain at least one modality")
        # Symmetric dropout across all available branches. When none survives,
        # restore ONE randomly selected originally available branch per row.
        # In particular, Tabular+Text without Image is now reachable in training.
        if self.training and self.modality_dropout_prob > 0:
            keep = mask & (torch.rand((batch, 3), device=h_tab.device)
                           >= self.modality_dropout_prob)
            empty = ~keep.any(dim=1)
            if empty.any():
                scores = torch.rand((int(empty.sum()), 3), device=h_tab.device)
                scores = scores.masked_fill(~mask[empty], -1.0)
                chosen = scores.argmax(dim=1)
                keep[empty] = torch.nn.functional.one_hot(chosen, 3).bool()
            mask = keep
        embeddings = torch.stack((h_tab, h_txt, h_img), dim=1)
        embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        logits = self.gate_net(embeddings.flatten(start_dim=1))
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=1)
        fused = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return fused, weights


class AirbnbMultimodalRegressor(nn.Module):
    def __init__(self, cfg: ModelConfig, cat_cardinalities: list[int]) -> None:
        super().__init__()
        self.tabular_encoder = FTTransformerEncoder(
            num_continuous=len(cfg.num_feature_names), cat_cardinalities=cat_cardinalities,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            d_ffn_factor=cfg.d_ffn_factor, dropout=cfg.transformer_dropout)
        self.text_proj = nn.Sequential(nn.Linear(cfg.raw_text_dim, cfg.d_model),
                                       nn.LayerNorm(cfg.d_model), nn.GELU())
        self.image_proj = nn.Sequential(nn.Linear(cfg.raw_image_dim, cfg.d_model),
                                        nn.LayerNorm(cfg.d_model), nn.GELU())
        self.fusion = GatedMultimodalFusion(cfg.d_model, cfg.modality_dropout_prob)
        self.regressor_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2),
                                            nn.GELU(), nn.Linear(cfg.d_model // 2, 1))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, x_txt: torch.Tensor,
                x_img: torch.Tensor, available_mask: torch.Tensor | None = None,
                return_aux: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_tab = self.tabular_encoder(x_num, x_cat)
        h_txt = self.text_proj(x_txt)
        h_img = self.image_proj(x_img)
        fused, weights = self.fusion(h_tab, h_txt, h_img, available_mask)
        prediction = self.regressor_head(fused).squeeze(-1)
        if return_aux:
            # Same prediction head supervises each representation independently.
            # These are TRAINING losses, not independent evaluation baselines.
            branches = torch.stack((h_tab, h_txt, h_img), dim=1)
            branch_predictions = self.regressor_head(branches).squeeze(-1)
            return prediction, weights, branch_predictions
        return prediction, weights
