from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    volatility: float
    mean_reversion: float
    jump_prob: float
    spread_bps: float


REGIMES = {
    "calm": RegimeConfig("calm", volatility=0.0004, mean_reversion=0.05, jump_prob=0.002, spread_bps=1.0),
    "volatile": RegimeConfig("volatile", volatility=0.0015, mean_reversion=0.02, jump_prob=0.015, spread_bps=3.5),
    "liquidity_shock": RegimeConfig("liquidity_shock", volatility=0.0030, mean_reversion=0.01, jump_prob=0.040, spread_bps=8.0),
}


def generate_lob_series(
    n_steps: int = 50_000,
    n_levels: int = 10,
    regime: str = "calm",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulate L2 Limit Order Book snapshots and forward mid-price markouts."""
    cfg = REGIMES[regime]
    rng = np.random.default_rng(seed)

    # Latent order-flow imbalance (OFI) with AR(1) persistence + jumps
    ofi = np.zeros(n_steps, dtype=np.float32)
    jumps = (rng.random(n_steps) < cfg.jump_prob) * rng.normal(0, 3.0, size=n_steps)
    eps = rng.normal(0, 1.0, size=n_steps)
    for t in range(1, n_steps):
        ofi[t] = 0.85 * ofi[t - 1] + 0.35 * eps[t] + jumps[t]

    # Mid-price dynamics driven by lagged OFI + microstructure noise
    ret = 0.00025 * np.tanh(ofi) + rng.normal(0, cfg.volatility, size=n_steps)
    mid_price = 100.0 * np.exp(np.cumsum(ret)).astype(np.float32)

    half_spread = (mid_price * (cfg.spread_bps * 1e-4) * 0.5).astype(np.float32)
    levels = np.arange(1, n_levels + 1, dtype=np.float32)[None, :]
    tick = (mid_price * 1e-4)[:, None]

    bid_prices = mid_price[:, None] - half_spread[:, None] - (levels - 1.0) * tick
    ask_prices = mid_price[:, None] + half_spread[:, None] + (levels - 1.0) * tick

    # Queue sizes skew with order-flow imbalance
    skew = np.clip(0.25 * np.tanh(ofi), -0.45, 0.45)[:, None]
    base_depth = rng.lognormal(mean=4.5, sigma=0.4, size=(n_steps, n_levels)).astype(np.float32)
    bid_sizes = base_depth * (1.0 + skew)
    ask_sizes = base_depth * (1.0 - skew)

    # Features per tick: [log_ret, rel_spread, l1_ofi, bid_sizes(10), ask_sizes(10)] -> 23 dims
    log_ret = np.diff(np.log(mid_price), prepend=np.log(mid_price[0])).astype(np.float32)
    rel_spread = ((ask_prices[:, 0] - bid_prices[:, 0]) / mid_price).astype(np.float32)
    l1_ofi = ((bid_sizes[:, 0] - ask_sizes[:, 0]) / (bid_sizes[:, 0] + ask_sizes[:, 0] + 1e-6)).astype(np.float32)

    features = np.column_stack([log_ret, rel_spread, l1_ofi, np.log1p(bid_sizes), np.log1p(ask_sizes)]).astype(np.float32)

    return {
        "features": features,
        "mid_price": mid_price,
        "half_spread": half_spread,
        "regime": regime,
    }
