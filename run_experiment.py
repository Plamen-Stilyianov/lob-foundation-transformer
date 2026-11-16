import torch
from data.alpaca_loader import CROSS_ASSET_SYMBOLS, load_cross_asset_market_data
from src.evaluate_economics import benchmark_inference_latency, evaluate_economic_performance
from src.model import LOBTransformerConfig
from src.pretrain import create_patch_sequences, train_foundation_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_symbols = ("SPY", "QQQ", "TLT", "GLD", "USO")
    unseen_symbols = ("IEF", "SLV")
    print(f"Pre-training LOB Foundation Transformer on real cross-asset data ({device})...")

    configs = {
        "Tiny (0.3M)": LOBTransformerConfig(d_model=96, n_heads=4, n_layers=2),
        "Base (1.8M)": LOBTransformerConfig(d_model=192, n_heads=6, n_layers=4),
    }

    for name, cfg in configs.items():
        model, losses, oos_splits = train_foundation_model(
            cfg,
            symbols=train_symbols,
            days_back=21,
            epochs=8,
            batch_size=64,
            checkpoint_path=f"checkpoints/{name.split()[0].lower()}.pt",
        )
        n_params = model.count_parameters()
        lat = benchmark_inference_latency(model, torch.randn(1, 32, cfg.feature_dim * cfg.patch_size))
        print(f"\n=== {name} | Params: {n_params:,} | Final Loss: {losses[-1]:.4f} | p99 Latency: {lat['p99_ms']} ms ===")

        model.eval()
        # 1. Chronological Out-of-Sample Evaluation on Pre-Trained Asset Classes
        for sym, split in oos_splits.items():
            X_ev, _, Y_ret = create_patch_sequences(split["features"], split["mid_price"], cfg.patch_size, 32, cfg.horizons)
            with torch.no_grad():
                _, pred_ret = model(X_ev.to(device))
            spread_bps = (split["half_spread"][-len(X_ev):] / split["mid_price"][-len(X_ev):]) * 1e4
            m = evaluate_economic_performance(pred_ret[:, -1, 1].cpu().numpy(), Y_ret[:, -1, 1].numpy(), spread_bps, threshold_bps=0.25)
            print(f"  OOS [{sym:3s} | {CROSS_ASSET_SYMBOLS.get(sym, 'Asset'):9s}] -> IC: {m['ic']:+.4f} | Net PnL: {m['total_net_pnl_bps']:+8.2f} bps | Sharpe: {m['net_sharpe']:+6.3f}")

        # 2. Zero-Shot Cross-Instrument Transfer on Unseen Assets
        for sym in unseen_symbols:
            raw = load_cross_asset_market_data(symbol=sym, days_back=21)
            X_ev, _, Y_ret = create_patch_sequences(raw["features"], raw["mid_price"], cfg.patch_size, 32, cfg.horizons)
            with torch.no_grad():
                _, pred_ret = model(X_ev.to(device))
            spread_bps = (raw["half_spread"][-len(X_ev):] / raw["mid_price"][-len(X_ev):]) * 1e4
            m = evaluate_economic_performance(pred_ret[:, -1, 1].cpu().numpy(), Y_ret[:, -1, 1].numpy(), spread_bps, threshold_bps=0.25)
            print(f"  Zero-Shot [{sym:3s}]         -> IC: {m['ic']:+.4f} | Net PnL: {m['total_net_pnl_bps']:+8.2f} bps | Sharpe: {m['net_sharpe']:+6.3f}")


if __name__ == "__main__":
    main()
