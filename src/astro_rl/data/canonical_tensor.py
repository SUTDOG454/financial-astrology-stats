from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .ephemeris import EphemerisConfig, SwissEphemerisProvider
from .market import build_market_features, load_market_frame

CORE_BODIES = ["sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
EXTRA_BODIES = ["chiron", "ceres", "pallas", "juno", "vesta", "sedna"]
ASPECTS = [(0.0, "conjunction"), (60.0, "sextile"), (90.0, "square"), (120.0, "trine"), (180.0, "opposition")]
FINANCIAL_PAIRS = [
    ("sun", "jupiter"), ("sun", "saturn"), ("jupiter", "saturn"), ("jupiter", "uranus"),
    ("jupiter", "pluto"), ("venus", "jupiter"), ("venus", "saturn"), ("venus", "pluto"),
    ("mars", "uranus"), ("saturn", "uranus"), ("saturn", "pluto"), ("uranus", "pluto"),
]

@dataclass(frozen=True)
class FinancialAstroFeatureConfig:
    backend: str = "swisseph"
    ephe_path: str | None = None
    jpl_file: str | None = None
    include_heliocentric: bool = True
    include_extra_bodies: bool = True
    aspect_orb_deg: float = 3.0
    station_speed_threshold: float = 0.03
    standardize: bool = False  # fit-on-train standardization belongs in the split pipeline, not here

def angular_distance(a: float, b: float) -> float:
    return float(abs((a - b + 180.0) % 360.0 - 180.0))

def aspect_residual(a: float, b: float, exact: float) -> float:
    return abs(angular_distance(a, b) - exact)

def _safe(v: float) -> float:
    return 0.0 if not np.isfinite(v) else float(v)

class CanonicalFinancialAstroTensorBuilder:
    """Market x Swiss Ephemeris/JPL tensor with point-in-time-safe alignment."""
    def __init__(self, config: FinancialAstroFeatureConfig | None = None):
        self.config = config or FinancialAstroFeatureConfig()
        bodies = {
            "sun": 0, "moon": 1, "mercury": 2, "venus": 3, "mars": 4,
            "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8, "pluto": 9,
            "chiron": -1, "ceres": 1, "pallas": 2, "juno": 3, "vesta": 4, "sedna": 90377,
        }
        common = dict(backend=self.config.backend, ephe_path=self.config.ephe_path,
                      jpl_file=self.config.jpl_file, include_extra_bodies=self.config.include_extra_bodies)
        self.geo = SwissEphemerisProvider(EphemerisConfig(**common, heliocentric=False), bodies=bodies)
        self.helio = (SwissEphemerisProvider(EphemerisConfig(**common, heliocentric=True), bodies=bodies)
                      if self.config.include_heliocentric else None)

    def _body_features(self, state, prefix):
        values, names = [], []
        bodies = CORE_BODIES + (EXTRA_BODIES if self.config.include_extra_bodies else [])
        for body in bodies:
            s = state.get(body, {})
            lon, lat, decl = _safe(s.get("longitude", 0.0)), _safe(s.get("latitude", 0.0)), _safe(s.get("declination", 0.0))
            speed, dist, retro = _safe(s.get("speed_longitude", 0.0)), _safe(s.get("distance_au", 0.0)), _safe(s.get("retrograde", 0.0))
            phase = math.radians(lon)
            values.extend([math.sin(phase), math.cos(phase), math.sin(math.radians(lat)),
                           math.sin(math.radians(decl)), math.cos(math.radians(decl)), speed / 2.0,
                           np.tanh(dist), retro])
            names.extend([f"{prefix}.{body}.lon_sin", f"{prefix}.{body}.lon_cos", f"{prefix}.{body}.lat_sin",
                          f"{prefix}.{body}.decl_sin", f"{prefix}.{body}.decl_cos", f"{prefix}.{body}.speed_norm",
                          f"{prefix}.{body}.distance_norm", f"{prefix}.{body}.retrograde"])
        return values, names

    def _aspect_features(self, state, prefix):
        vals, names = [], []
        available = [b for b in CORE_BODIES if b in state]
        for a, b in combinations(available, 2):
            for exact, label in ASPECTS:
                residual = aspect_residual(state[a]["longitude"], state[b]["longitude"], exact)
                vals.append(max(0.0, 1.0 - residual / self.config.aspect_orb_deg))
                names.append(f"{prefix}.aspect.{a}_{b}.{label}")
        return vals, names

    def _financial_geometry(self, state, prefix):
        vals, names = [], []
        for a, b in FINANCIAL_PAIRS:
            if a not in state or b not in state:
                continue
            sep = angular_distance(state[a]["longitude"], state[b]["longitude"])
            for exact, label in ASPECTS:
                vals.append(max(0.0, 1.0 - abs(sep - exact) / self.config.aspect_orb_deg))
                names.append(f"{prefix}.financial_geometry.{a}_{b}.{label}")
        return vals, names

    def _harmonics(self, state, prefix):
        vals, names = [], []
        available = [b for b in CORE_BODIES if b in state]
        for h in range(1, 25):
            scores = [math.cos(h * math.radians(state[a]["longitude"] - state[b]["longitude"]))
                      for a, b in combinations(available, 2)]
            vals.append(float(np.mean(scores)) if scores else 0.0)
            names.append(f"{prefix}.harmonic.H{h}.mean_cosine_resonance")
        return vals, names

    def _global_features(self, geo, helio):
        vals, names = [], []
        if "sun" in geo and "moon" in geo:
            phase = angular_distance(geo["sun"]["longitude"], geo["moon"]["longitude"])
            vals += [math.sin(math.radians(phase)), math.cos(math.radians(phase)), phase / 180.0]
            names += ["geo.lunar_phase.sin", "geo.lunar_phase.cos", "geo.lunar_phase.normalized"]
        for body in CORE_BODIES:
            vals.append(float(abs(_safe(geo.get(body, {}).get("speed_longitude", 0.0))) < self.config.station_speed_threshold))
            names.append(f"geo.{body}.station_proximity")
        for a, b in FINANCIAL_PAIRS:
            if a in geo and b in geo:
                da, db = geo[a]["declination"], geo[b]["declination"]
                vals.extend([max(0.0, 1.0 - abs(da - db) / 1.0), max(0.0, 1.0 - abs(da + db) / 1.0)])
                names.extend([f"geo.declination.{a}_{b}.parallel", f"geo.declination.{a}_{b}.contraparallel"])
        if helio is not None:
            lats = [_safe(helio.get(b, {}).get("latitude", 0.0)) for b in CORE_BODIES]
            vals += [float(np.mean(lats)), float(np.std(lats))]
            names += ["helio.latitude.mean", "helio.latitude.std"]
        return vals, names

    def _row(self, timestamp: pd.Timestamp):
        geo = self.geo.state(timestamp.to_pydatetime())
        helio = self.helio.state(timestamp.to_pydatetime()) if self.helio else None
        vals, names = [], []
        for fn in (lambda: self._body_features(geo, "geo"), lambda: self._aspect_features(geo, "geo"),
                   lambda: self._financial_geometry(geo, "geo"), lambda: self._harmonics(geo, "geo"),
                   lambda: self._global_features(geo, helio)):
            v, n = fn(); vals.extend(v); names.extend(n)
        if helio is not None:
            for fn in (lambda: self._body_features(helio, "helio"), lambda: self._aspect_features(helio, "helio"),
                       lambda: self._financial_geometry(helio, "helio")):
                v, n = fn(); vals.extend(v); names.extend(n)
        return vals, names

    def build(self, market_df: pd.DataFrame):
        market, market_names = build_market_features(market_df)
        rows, names = [], None
        for ts in market_df["date"]:
            row, row_names = self._row(pd.Timestamp(ts))
            rows.append(row)
            names = names or row_names
        astro = np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if astro.ndim != 2 or astro.shape[0] != market.shape[0]:
            raise RuntimeError("Market and ephemeris tensors are not row-aligned")
        if self.config.standardize:
            raise ValueError("standardize=True is disabled in the canonical builder; fit scalers on training-only data")
        return market.astype(np.float32), astro.astype(np.float32), market_names + (names or [])


def build_aligned_tensor(market_path: str | Path, output_path: str | Path, config: FinancialAstroFeatureConfig | None = None) -> dict:
    df = load_market_frame(market_path)
    builder = CanonicalFinancialAstroTensorBuilder(config)
    market, astro, feature_names = builder.build(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    market_feature_count = len(build_market_features(df)[1])
    np.savez_compressed(output_path, market=market, astro=astro,
                        dates=df["date"].astype("int64").to_numpy(),
                        feature_names=np.asarray(feature_names, dtype=str),
                        market_feature_count=np.asarray([market_feature_count], dtype=np.int64))
    meta = {"rows": int(len(df)), "market_dim": int(market.shape[1]), "astro_dim": int(astro.shape[1]),
            "start": str(df["date"].iloc[0]), "end": str(df["date"].iloc[-1]),
            "backend": (config or FinancialAstroFeatureConfig()).backend, "feature_names": feature_names}
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
