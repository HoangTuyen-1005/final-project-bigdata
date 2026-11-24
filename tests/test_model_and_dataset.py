"""CPU-only regression tests. Run: python -m pytest tests -q"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from config import ModelConfig
from dataset import AirbnbMultimodalDataset
from evaluator import MultimodalEvaluator
from main import create_mock_airbnb_data, train_model, ridge_tabular_baseline
from models.multimodal_net import AirbnbMultimodalRegressor, GatedMultimodalFusion
from torch.utils.data import DataLoader


def test_masked_softmax_zeros_invalid_branches():
    gate = GatedMultimodalFusion(d_model=8, modality_dropout_prob=0.0).eval()
    vectors = [torch.randn(3, 8) for _ in range(3)]
    available = torch.tensor([[True, True, False],
                              [True, False, False], [False, True, True]])
    fused, weights = gate(*vectors, available_mask=available)
    assert fused.shape == (3, 8)
    assert torch.all(weights[~available] == 0)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
    assert weights[1, 0] == 1


def test_dropout_masks_zero_gate_mass():
    torch.manual_seed(5)
    gate = GatedMultimodalFusion(d_model=8, modality_dropout_prob=1.0).train()
    _, weights = gate(*(torch.randn(4, 8) for _ in range(3)))
    # Symmetric p=1 forces exactly ONE randomly retained, originally valid
    # branch per example. No modality is permanently guaranteed by the model.
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))
    assert torch.all((weights == 0) | (weights == 1))
    assert (weights.argmax(dim=1).unique().numel() > 1)


def test_all_modalities_masked_is_rejected():
    gate = GatedMultimodalFusion(8, 0).eval()
    with pytest.raises(ValueError):
        gate(*(torch.randn(2, 8) for _ in range(3)),
             available_mask=torch.zeros((2, 3), dtype=torch.bool))


def test_dataset_rejects_duplicate_ids_and_invalid_embeddings():
    n = 3
    args = [np.zeros((n, 2)), np.zeros((n, 2), dtype=np.int64),
            np.zeros((n, 4)), np.zeros((n, 3)), np.zeros(n)]
    with pytest.raises(ValueError, match="duplicate"):
        AirbnbMultimodalDataset(*args, keys=["city::1", "city::1", "city::3"])
    args[3][1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        AirbnbMultimodalDataset(*args)


def test_smoke_training_evaluation(tmp_path):
    torch.manual_seed(1)
    cfg = replace(ModelConfig(), d_model=16, n_heads=4, n_layers=1,
                  raw_text_dim=12, raw_image_dim=8, batch_size=8,
                  epochs=1, early_stopping_patience=1)
    train, val, test, cards = create_mock_airbnb_data(cfg, n_samples=48)
    model = AirbnbMultimodalRegressor(cfg, cards)
    # Reduce CPU thread oversubscription in constrained CI runners.
    torch.set_num_threads(2)
    info = train_model(model, DataLoader(train, 8, shuffle=True),
                       DataLoader(val, 8), cfg, torch.device("cpu"), tmp_path / "best.pt")
    assert info["best_epoch"] == 1
    evaluator = MultimodalEvaluator(model, test, cfg.num_feature_names,
                                    cfg.cat_cols, device=torch.device("cpu"), batch_size=4)
    metrics, gates, cities = evaluator.evaluate(by_city=True)
    assert all(np.isfinite(v) for v in metrics.values())
    assert abs(sum(gates.values()) - 100.0) < 0.1
    assert len(cities) == 1
    assert len(evaluator.evaluate_ablation()) == 4
    assert len(evaluator.measure_modality_importance(n_repeats=1)) == 3
    assert len(evaluator.measure_field_importance(n_repeats=1)) == len(cfg.num_feature_names) + len(cfg.cat_cols)
    ridge = ridge_tabular_baseline(train, test)
    assert ridge["RMSE_log"] >= 0


def test_subsets_preserve_real_availability_and_complete_case():
    n = 5
    available = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [1, 1, 1]])
    ds = AirbnbMultimodalDataset(np.ones((n, 3)), np.zeros((n, 1), dtype=np.int64),
                                np.ones((n, 4)), np.ones((n, 2)), np.ones(n),
                                keys=[f"city::{i}" for i in range(n)], available_mask=available)
    selected = ds.select_modalities((True, True, False), "complete")
    assert len(selected) == 2
    assert selected.available_mask.tolist() == [[True, True, False]] * 2
    eligible = ds.select_modalities((False, False, True), "eligible")
    assert len(eligible) == 3
    assert eligible.available_mask.tolist() == [[False, False, True]] * 3
    single = AirbnbMultimodalDataset(np.ones((1, 3)), np.zeros((1, 1), dtype=np.int64),
                                    np.zeros((1, 4)), np.zeros((1, 2)), np.ones(1),
                                    available_mask=np.array([[1, 0, 0]]))
    with pytest.raises(ValueError, match="No complete"):
        single.select_modalities((False, True, False), "complete")


def test_masked_fusion_mixed_availability_and_aux_loss_gradient():
    torch.manual_seed(9)
    cfg = replace(ModelConfig(), d_model=16, n_heads=4, n_layers=1,
                  raw_text_dim=12, raw_image_dim=8)
    model = AirbnbMultimodalRegressor(cfg, [3] * len(cfg.cat_cols))
    model.train()
    mask = torch.tensor([[True, True, False], [True, False, False],
                         [True, True, True]], dtype=torch.bool)
    tensors = (torch.randn(3, len(cfg.num_feature_names)),
               torch.zeros(3, len(cfg.cat_cols), dtype=torch.int64),
               torch.randn(3, cfg.raw_text_dim), torch.randn(3, cfg.raw_image_dim))
    pred, gate, aux = model(*tensors, available_mask=mask, return_aux=True)
    assert pred.shape == (3,) and aux.shape == (3, 3)
    assert (gate[~mask] == 0).all()
    criterion = torch.nn.functional.huber_loss(aux, torch.ones_like(aux), reduction="none")
    loss = ((criterion * mask).sum() / mask.sum()) + pred.square().mean()
    loss.backward()
    assert model.text_proj[0].weight.grad is not None
    assert model.image_proj[0].weight.grad is not None


def test_dropout_supports_all_nonempty_modality_subsets():
    torch.manual_seed(123)
    fusion = GatedMultimodalFusion(8, 0.5).train()
    _, weights = fusion(*(torch.randn(1024, 8) for _ in range(3)))
    observed = {tuple(row) for row in (weights.detach().numpy() > 0).tolist()}
    expected = {(a,b,c) for a in (False, True) for b in (False, True)
                for c in (False, True) if a or b or c}
    assert observed == expected
