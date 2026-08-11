from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"date", "open", "high", "low", "close", "volume"}


def load_market_frame(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported market file: {p.suffix}; use CSV or Parquet")
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"timestamp": "date", "datetime": "date", "time": "date", "adj_close": "close"}
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Market data missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")
    for c in REQUIRED - {"date"}:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(REQUIRED)).sort_values("date").drop_duplicates("date", keep="last")
    if len(df) < 32:
        raise ValueError("Market dataset is too short; at least 32 aligned observations are required")
    return df.reset_index(drop=True)


def build_market_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    x = df.copy()
    close = x["close"].astype(float)
    log_close = np.log(close.clip(lower=1e-12))
    logret = log_close.diff()
    ret = close.pct_change()
    high_low = (x["high"] - x["low"]) / close.replace(0, np.nan)
    oc = (x["close"] - x["open"]) / x["open"].replace(0, np.nan)
    vol_log = np.log1p(x["volume"].clip(lower=0))
    feats = pd.DataFrame({
        "return_1": ret,
        "log_return_1": logret,
        "range_pct": high_low,
        "body_pct": oc,
        "volume_log": vol_log,
        "momentum_5": close.pct_change(5),
        "momentum_20": close.pct_change(20),
        "drawdown_20": close / close.rolling(20).max() - 1.0,
        "volatility_5": logret.rolling(5).std(),
        "volatility_20": logret.rolling(20).std(),
        "volume_z20": (vol_log - vol_log.rolling(20).mean()) / vol_log.rolling(20).std().replace(0, np.nan),
        "trend_20": (close.rolling(5).mean() / close.rolling(20).mean()) - 1.0,
    })
    # Forward fill is point-in-time safe; early unavailable rolling windows become zero.
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    raw_return = feats["return_1"].to_numpy(dtype=np.float32)
    arr = feats.to_numpy(dtype=np.float32)
    med = np.nanmedian(arr, axis=0)
    mad = np.nanmedian(np.abs(arr - med), axis=0) + 1e-6
    arr = np.clip((arr - med) / (1.4826 * mad), -8.0, 8.0).astype(np.float32)
    arr[:, 0] = raw_return
    return arr, list(feats.columns)
