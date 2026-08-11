from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_market_csv(path: str | Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    aliases = {c.lower(): c for c in df.columns}
    if timestamp_col not in df.columns:
        for candidate in ("datetime", "date", "time", "timestamp"):
            if candidate in aliases:
                timestamp_col = aliases[candidate]
                break
    required = {"open", "high", "low", "close"}
    lower = {c.lower() for c in df.columns}
    missing = required - lower
    if missing:
        raise ValueError(f"market data missing columns: {sorted(missing)}")
    rename = {aliases[x]: x for x in required if x in aliases}
    df = df.rename(columns=rename)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values(timestamp_col).drop_duplicates(timestamp_col).reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def market_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> tuple[np.ndarray, list[str]]:
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    logp = np.log(c.clip(lower=1e-12))
    ret = logp.diff().fillna(0.0)
    mom5 = logp.diff(5).fillna(0.0)
    mom20 = logp.diff(20).fillna(0.0)
    vol20 = ret.rolling(20, min_periods=2).std().fillna(0.0)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=2).mean().fillna(tr)
    dd = c / c.cummax().replace(0, np.nan) - 1.0
    vol_mean = v.rolling(20, min_periods=2).mean()
    vol_std = v.rolling(20, min_periods=2).std().replace(0, np.nan)
    vol_z = ((v - vol_mean) / vol_std).fillna(0.0)
    delta = c.diff().fillna(0.0)
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = up.rolling(14, min_periods=2).mean() / down.rolling(14, min_periods=2).mean().replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50.0) / 100.0
    x = np.column_stack([ret, mom5, mom20, vol20, atr14 / c.clip(lower=1e-12), dd, vol_z, rsi])
    names = ["ret_1", "mom_5", "mom_20", "vol_20", "atr14_rel", "drawdown", "volume_z20", "rsi14"]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names
