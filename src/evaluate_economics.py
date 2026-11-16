import time
import numpy as np
import torch


def evaluate_economic_performance(
    preds_bps: np.ndarray,
    realized_ret_bps: np.ndarray,
    half_spread_bps: np.ndarray,
    threshold_bps: float = 0.5,
) -> dict[str, float]:
    """Compute directional IC, gross/net markouts, turnover, and post-cost Sharpe."""
    ic = float(np.corrcoef(preds_bps, realized_ret_bps)[0, 1]) if np.std(preds_bps) > 1e-9 else 0.0

    # Position signal: +1 long, -1 short, 0 flat when expected edge < threshold
    pos = np.where(preds_bps > threshold_bps, 1.0, np.where(preds_bps < -threshold_bps, -1.0, 0.0))
    turnover = np.abs(np.diff(pos, prepend=0.0))

    gross_pnl_bps = pos * realized_ret_bps
    cost_bps = turnover * half_spread_bps
    net_pnl_bps = gross_pnl_bps - cost_bps

    active_trades = float(np.sum(turnover > 0))
    mean_markout_bps = float(np.mean(gross_pnl_bps[pos != 0])) if np.any(pos != 0) else 0.0
    sharpe = float(np.mean(net_pnl_bps) / (np.std(net_pnl_bps) + 1e-8) * np.sqrt(252 * 390))

    return {
        "ic": round(ic, 4),
        "gross_markout_bps": round(mean_markout_bps, 4),
        "total_net_pnl_bps": round(float(np.sum(net_pnl_bps)), 2),
        "net_sharpe": round(sharpe, 3),
        "trade_count": active_trades,
    }


@torch.no_grad()
def benchmark_inference_latency(
    model: torch.nn.Module,
    sample_batch: torch.Tensor,
    n_runs: int = 100,
) -> dict[str, float]:
    """Measure p50 and p99 single-batch inference latency in milliseconds."""
    model.eval()
    device = next(model.parameters()).device
    sample_batch = sample_batch.to(device)

    for _ in range(10):
        _ = model(sample_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model(sample_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }
