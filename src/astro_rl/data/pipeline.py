from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .canonical import CanonicalDataset, DatasetSpec
from .ephemeris import EphemerisConfig, SwissEphemerisProvider
from .features import FeatureConfig, FinancialAstroFeatureBuilder
from .market import load_market_csv, market_features


def _fit_scale(x: np.ndarray, end: int) -> tuple[np.ndarray, dict]:
    end = max(2, min(end, len(x)))
    mu = x[:end].mean(axis=0)
    sd = x[:end].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32), {"mean": mu.tolist(), "std": sd.tolist(), "fit_end": int(end)}


def build_canonical_dataset(
    market_path: str | Path,
    *,
    timestamp_col: str = "timestamp",
    ephemeris: EphemerisConfig | None = None,
    features: FeatureConfig | None = None,
    train_fraction: float = 0.70,
    start: str | None = None,
    end: str | None = None,
) -> CanonicalDataset:
    df = load_market_csv(market_path, timestamp_col=timestamp_col)
    if start:
        df = df[df[timestamp_col] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df[timestamp_col] <= pd.Timestamp(end, tz="UTC")]
    df = df.reset_index(drop=True)
    if len(df) < 256:
        raise ValueError("at least 256 aligned market observations are required")

    market, market_names = market_features(df, timestamp_col=timestamp_col)
    eph_cfg = ephemeris or EphemerisConfig()
    provider = SwissEphemerisProvider(eph_cfg)
    eph, eph_names = provider.matrix(df[timestamp_col])
    astro_builder = FinancialAstroFeatureBuilder(eph_names, features)
    astro_extra, astro_extra_names = astro_builder.build(eph)
    astro = np.hstack([eph, astro_extra]).astype(np.float32)

    split = int(len(df) * train_fraction)
    market, market_scaler = _fit_scale(market, split)
    astro, astro_scaler = _fit_scale(astro, split)
    timestamps = df[timestamp_col].to_numpy(dtype="datetime64[ns]")

    ds = CanonicalDataset(
        timestamps=timestamps,
        market=market,
        astro=astro,
        feature_names=market_names + eph_names + astro_extra_names,
        metadata={
            "schema": DatasetSpec(frequency="market-row", ephemeris_backend=eph_cfg.backend).__dict__,
            "source_market": str(market_path),
            "n_rows": int(len(df)),
            "market_dim": int(market.shape[1]),
            "astro_dim": int(astro.shape[1]),
            "market_scaler": market_scaler,
            "astro_scaler": astro_scaler,
            "financial_astrology": {
                "classical_aspects": list((features or FeatureConfig()).classical_aspects),
                "magi_geometry": list((features or FeatureConfig()).magi_geometry),
                "harmonics": list((features or FeatureConfig()).harmonics),
                "note": "Hypothesis features; empirical validity must be established by out-of-sample testing.",
            },
        },
    )
    ds.validate()
    return ds


def save_dataset_npz(ds: CanonicalDataset, path: str | Path) -> None:
    ds.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, market=ds.market, astro=ds.astro, timestamps=ds.timestamps.astype("datetime64[ns]"))
    path.with_suffix(".json").write_text(json.dumps({"feature_names": ds.feature_names, **ds.metadata}, indent=2, default=str), encoding="utf-8")
