import numpy as np
import torch
from data.lob_simulator import generate_lob_series
from src.evaluate_economics import benchmark_inference_latency, evaluate_economic_performance
from src.model import LOBTransformerConfig
from src.pretrain import create_patch_sequences, train_foundation_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running LOB Foundation Scaling & Regime Benchmark on: {device}")

    configs = {
        "Tiny (0.3M)": LOBTransformerConfig(d_model=96, n_heads=4, n_layers=2),
        "Base (1.8M)": LOBTransformerConfig(d_model=192, n_heads=6, n_layers=4),
    }

    for name, cfg in configs.items():
        model, losses = train_foundation_model(
            cfg, regime="calm", n_steps=30_000, epochs=4, batch_size=64, checkpoint_path=f"checkpoints/{name.split()[0].lower()}.pt"
        )
        n_params = model.count_parameters()
        lat = benchmark_inference_latency(model, torch.randn(1, 32, cfg.feature_dim * cfg.patch_size))
        print(f"\n=== {name} | Params: {n_params:,} | Final Loss: {losses[-1]:.4f} | p99 Latency: {lat['p99_ms']} ms ===")

        model.eval()
        for eval_regime in ("calm", "volatile", "liquidity_shock"):
            sim = generate_lob_series(n_steps=12_000, regime=eval_regime, seed=99)
            X_ev, _, Y_ret = create_patch_sequences(sim["features"], sim["mid_price"], cfg.patch_size, 32, cfg.horizons)
            with torch.no_grad():
                _, pred_ret = model(X_ev.to(device))
            # Evaluate 5-tick horizon (index 1) at final patch of each sequence
            preds_bps = pred_ret[:, -1, 1].cpu().numpy()
            realized_bps = Y_ret[:, -1, 1].numpy()
            spread_bps = np.full_like(realized_bps, 0.5 if eval_regime == "calm" else 1.75)
            metrics = evaluate_economic_performance(preds_bps, realized_bps, spread_bps, threshold_bps=0.4)
            print(f"  Regime [{eval_regime:15s}] -> IC: {metrics['ic']:+.4f} | Net PnL: {metrics['total_net_pnl_bps']:+8.2f} bps | Net Sharpe: {metrics['net_sharpe']:+6.3f}")


if __name__ == "__main__":
    main()