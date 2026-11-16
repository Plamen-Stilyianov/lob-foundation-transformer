import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import numpy as np
import pandas as pd

from data.lob_simulator import generate_lob_series

load_dotenv(override=False)

CROSS_ASSET_SYMBOLS: dict[str, str] = {
    "SPY": "Equity",
    "QQQ": "Equity",
    "TLT": "Rates",
    "IEF": "Rates",
    "GLD": "Commodity",
    "USO": "Commodity",
    "FXE": "FX",
    "FXY": "FX",
    "UUP": "FX",
}


def bars_to_microstructure_features(df: pd.DataFrame, target_dim: int = 23) -> dict[str, np.ndarray]:
    """Convert OHLCV + VWAP bars into a 23-dim float32 feature matrix aligned with LOBFoundationTransformer."""
    close = df["close"].to_numpy(dtype=np.float32)
    high = df["high"].to_numpy(dtype=np.float32)
    low = df["low"].to_numpy(dtype=np.float32)
    open_p = df["open"].to_numpy(dtype=np.float32)
    volume = df["volume"].to_numpy(dtype=np.float32)
    vwap = df["vwap"].to_numpy(dtype=np.float32) if "vwap" in df.columns else close

    log_ret = np.diff(np.log(close + 1e-8), prepend=np.log(close[0] + 1e-8)).astype(np.float32)
    hl_spread = ((high - low) / (close + 1e-8)).astype(np.float32)
    bar_ofi = ((close - open_p) / (high - low + 1e-6)).astype(np.float32)
    vwap_dev = ((close - vwap) / (vwap + 1e-8)).astype(np.float32)
    log_vol = np.log1p(volume).astype(np.float32)

    base_cols = [log_ret, hl_spread, bar_ofi, vwap_dev, log_vol]
    s = pd.Series(log_ret)
    for w in (3, 5, 10, 20, 40, 60):
        base_cols.append(s.rolling(w, min_periods=1).mean().to_numpy(dtype=np.float32))
        base_cols.append(s.rolling(w, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float32))

    v_s = pd.Series(log_vol)
    for w in (3, 5, 10, 20, 40, 60):
        base_cols.append((v_s - v_s.rolling(w, min_periods=1).mean()).to_numpy(dtype=np.float32))

    features = np.column_stack(base_cols[:target_dim]).astype(np.float32)
    half_spread = (close * np.maximum(hl_spread * 0.25, 1e-4)).astype(np.float32)

    return {
        "features": features,
        "mid_price": close,
        "half_spread": half_spread,
        "regime": "real_market_alpaca",
    }


def load_cross_asset_market_data(
    symbol: str = "SPY",
    days_back: int = 14,
    min_bars: int = 512,
    n_fallback_steps: int = 12_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Fetch real intraday 1-minute IEX bars via alpaca-py when API keys are set; otherwise fall back cleanly."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        return generate_lob_series(n_steps=n_fallback_steps, regime="calm", seed=seed)

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX,
    )
    bars_df = client.get_stock_bars(req).df.reset_index()
    if len(bars_df) < min_bars:
        return generate_lob_series(n_steps=n_fallback_steps, regime="calm", seed=seed)

    return bars_to_microstructure_features(bars_df)
