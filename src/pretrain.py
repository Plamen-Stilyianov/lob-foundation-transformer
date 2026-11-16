import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from data.alpaca_loader import load_cross_asset_market_data
from src.model import LOBFoundationTransformer, LOBTransformerConfig


def create_patch_sequences(
    features: np.ndarray,
    mid_price: np.ndarray,
    patch_size: int = 4,
    seq_patches: int = 32,
    horizons: tuple[int, ...] = (1, 5, 15),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice market features into non-overlapping patch sequences and forward return targets (bps)."""
    max_h = max(horizons)
    n_ticks = len(features) - max_h
    n_total_patches = n_ticks // patch_size
    if n_total_patches <= seq_patches:
        raise ValueError(f"Insufficient ticks ({len(features)}) for seq_patches={seq_patches} and patch_size={patch_size}.")

    trimmed_ticks = n_total_patches * patch_size
    patches = features[:trimmed_ticks].reshape(n_total_patches, patch_size * features.shape[1])

    patch_end_idx = np.arange(patch_size - 1, trimmed_ticks, patch_size)
    p0 = mid_price[patch_end_idx]
    targets_bps = np.column_stack(
        [((mid_price[patch_end_idx + h] - p0) / (p0 + 1e-8)) * 1e4 for h in horizons]
    ).astype(np.float32)

    X_list, Y_patch_list, Y_ret_list = [], [], []
    for i in range(n_total_patches - seq_patches):
        X_list.append(patches[i : i + seq_patches])
        Y_patch_list.append(patches[i + 1 : i + seq_patches + 1])
        Y_ret_list.append(targets_bps[i : i + seq_patches])

    return (
        torch.from_numpy(np.asarray(X_list, dtype=np.float32)),
        torch.from_numpy(np.asarray(Y_patch_list, dtype=np.float32)),
        torch.from_numpy(np.asarray(Y_ret_list, dtype=np.float32)),
    )


def train_foundation_model(
    cfg: LOBTransformerConfig,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "TLT", "GLD", "USO"),
    days_back: int = 21,
    train_split: float = 0.75,
    seq_patches: int = 32,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 42,
    checkpoint_path: str | None = "checkpoints/lob_foundation.pt",
) -> tuple[LOBFoundationTransformer, list[float], dict[str, dict[str, np.ndarray]]]:
    """Pre-train LOBFoundationTransformer from scratch on real multi-asset market data (bfloat16 on CUDA)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_parts, Yp_parts, Yr_parts = [], [], []
    oos_splits: dict[str, dict[str, np.ndarray]] = {}

    for idx, sym in enumerate(symbols):
        data = load_cross_asset_market_data(symbol=sym, days_back=days_back, seed=seed + idx)
        split_idx = int(len(data["features"]) * train_split)

        X_tr, Yp_tr, Yr_tr = create_patch_sequences(
            data["features"][:split_idx], data["mid_price"][:split_idx], cfg.patch_size, seq_patches, cfg.horizons
        )
        X_parts.append(X_tr)
        Yp_parts.append(Yp_tr)
        Yr_parts.append(Yr_tr)

        oos_splits[sym] = {
            "features": data["features"][split_idx:],
            "mid_price": data["mid_price"][split_idx:],
            "half_spread": data["half_spread"][split_idx:],
        }

    loader = DataLoader(
        TensorDataset(torch.cat(X_parts), torch.cat(Yp_parts), torch.cat(Yr_parts)),
        batch_size=batch_size,
        shuffle=True,
    )

    model = LOBFoundationTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    total_steps = max(1, epochs * len(loader))
    warmup_steps = max(1, int(0.1 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"

    epoch_losses: list[float] = []
    model.train()
    for _ in range(epochs):
        running_loss = 0.0
        for bx, by_patch, by_ret in loader:
            bx, by_patch, by_ret = bx.to(device), by_patch.to(device), by_ret.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pred_patch, pred_ret = model(bx)
                loss = F.mse_loss(pred_patch, by_patch) + 0.25 * F.huber_loss(pred_ret, by_ret, delta=2.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())

        epoch_losses.append(running_loss / max(1, len(loader)))

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": cfg, "state_dict": model.state_dict(), "losses": epoch_losses}, ckpt)

    return model, epoch_losses, oos_splits
