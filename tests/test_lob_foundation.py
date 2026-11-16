import numpy as np
import torch
from src.evaluate_economics import evaluate_economic_performance
from src.model import LOBFoundationTransformer, LOBTransformerConfig


def test_model_output_shapes():
    cfg = LOBTransformerConfig(feature_dim=23, patch_size=4, d_model=64, n_heads=4, n_layers=2)
    model = LOBFoundationTransformer(cfg)
    x = torch.randn(8, 16, 23 * 4)
    pred_patch, pred_ret = model(x)
    assert pred_patch.shape == (8, 16, 23 * 4)
    assert pred_ret.shape == (8, 16, len(cfg.horizons))


def test_strict_causal_invariance():
    """Modifying future patch t+1 must not alter outputs at timesteps <= t."""
    torch.manual_seed(0)
    cfg = LOBTransformerConfig(feature_dim=23, patch_size=4, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    model = LOBFoundationTransformer(cfg).eval()

    x1 = torch.randn(2, 12, 23 * 4)
    x2 = x1.clone()
    x2[:, 6:, :] += 10.0  # Perturb future patches from index 6 onward

    with torch.no_grad():
        p1, r1 = model(x1)
        p2, r2 = model(x2)

    assert torch.allclose(p1[:, :6, :], p2[:, :6, :], atol=1e-6)
    assert torch.allclose(r1[:, :6, :], r2[:, :6, :], atol=1e-6)


def test_economic_evaluation_deducts_spread_costs():
    preds = np.array([2.0, 2.0, -2.0, -2.0])
    realized = np.array([1.0, 1.0, -1.0, -1.0])
    zero_cost = evaluate_economic_performance(preds, realized, np.zeros(4), threshold_bps=0.5)
    with_cost = evaluate_economic_performance(preds, realized, np.full(4, 0.5), threshold_bps=0.5)
    assert with_cost["total_net_pnl_bps"] < zero_cost["total_net_pnl_bps"]
