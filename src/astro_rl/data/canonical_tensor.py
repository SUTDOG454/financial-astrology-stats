from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
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
    standardize: bool = True


def angular_distance(a: float, b: float) -> float:
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(d)


def aspect_residual(a: float, b: float, exact: float) -> float:
    return abs(angular_distance(a, b) - exact)


def _safe(v: float) -> float:
    return 0.0 if not np.isfinite(v) else float(v)


class CanonicalFinancialAstroTensorBuilder:
    """Market x ephemeris feature tensor with deterministic, point-in-time alignment.

    Row t contains only information available at market timestamp t. The PPO
    environment uses market return at t+1 as the next realized reward, preventing
    future-return leakage into the feature tensor.
    """

    def __init__(self, config: FinancialAstroFeatureConfig | None = None):
        self.config = config or FinancialAstroFeatureConfig()
        bodies = {
            "sun": 0, "moon": 1, "mercury": 2, "venus": 3, "mars": 4,
            "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8, "pluto": 9,
            "chiron": -1, "ceres": 1, "pallas": 2, "juno": 3, "vesta": 4, "sedna": 90377,
        }
        self.geo = SwissEphemerisProvider(EphemerisConfig(
            backend=self.config.backend, ephe_path=self.config.ephe_path,
            jpl_file=self.config.jpl_file, include_extra_bodies=self.config.include_extra_bodies,
            heliocentric=False,
        ), bodies=bodies)
        self.helio = None
        if self.config.include_heliocentric:
            self.helio = SwissEphemerisProvider(EphemerisConfig(
                backend=self.config.backend, ephe_path=self.config.ephe_path,
                jpl_file=self.config.jpl_file, include_extra_bodies=self.config.include_extra_bodies,
                heliocentric=True,
            ), bodies=bodies)

    def _body_features(self, state: dict[str, dict[str, float]], prefix: str) -> tuple[list[float], list[str]]:
        values, names = [], []
        for body in CORE_BODIES + (EXTRA_BODIES if self.config.include_extra_bodies else []):
            s = state.get(body, {})
            lon = _safe(s.get("longitude", 0.0))
            lat = _safe(s.get("latitude", 0.0))
            decl = _safe(s.get("declination", 0.0))
            speed = _safe(s.get("speed_longitude", 0.0))
            dist = _safe(s.get("distance_au", 0.0))
            retro = _safe(s.get("retrograde", 0.0))
            phase = math.radians(lon)
            values.extend([math.sin(phase), math.cos(phase), math.sin(math.radians(lat)),
                           math.sin(math.radians(decl)), math.cos(math.radians(decl)),
                           speed / 2.0, np.tanh(dist), retro])
            names.extend([f"{prefix}.{body}.lon_sin", f"{prefix}.{body}.lon_cos",
                          f"{prefix}.{body}.lat_sin", f"{prefix}.{body}.decl_sin",
                          f"{prefix}.{body}.decl_cos", f"{prefix}.{body}.speed_norm",
                          f"{prefix}.{body}.distance_norm", f"{prefix}.{body}.retrograde"])
        return values, names

    def _aspect_features(self, state: dict[str, dict[str, float]], prefix: str) -> tuple[list[float], list[str]]:
        vals, names = [], []
        available = [b for b in CORE_BODIES if b in state]
        for a, b in combinations(available, 2):
            sep = angular_distance(state[a]["longitude"], state[b]["longitude"])
            for exact, label in ASPECTS:
                residual = aspect_residual(state[a]["longitude"], state[b]["longitude"], exact)
                strength = max(0.0, 1.0 - residual / self.config.aspect_orb_deg)
                vals.append(strength)
                names.append(f"{prefix}.aspect.{a}_{b}.{label}")
        return vals, names

    def _financial_geometry(self, state: dict[str, dict[str, float]], prefix: str) -> tuple[list[float], list[str]]:
        vals, names = [], []
        for a, b in FINANCIAL_PAIRS:
            if a not in state or b not in state:
                continue
            sep = angular_distance(state[a]["longitude"], state[b]["longitude"])
            for exact, label in ASPECTS:
                residual = abs(sep - exact)
                vals.append(max(0.0, 1.0 - residual / self.config.aspect_orb_deg))
                names.append(f"{prefix}.financial_geometry.{a}_{b}.{label}")
        return vals, names

    def _harmonics(self, state: dict[str, dict[str, float]], prefix: str) -> tuple[list[float], list[str]]:
        vals, names = [], []
        available = [b for b in CORE_BODIES if b in state]
        for h in range(1, 25):
            # Harmonic resonance = mean cos(n * phase difference) across core pairs.
            pair_scores = []
            for a, b in combinations(available, 2):
                d = math.radians(state[a]["longitude"] - state[b]["longitude"])
                pair_scores.append(math.cos(h * d))
            vals.append(float(np.mean(pair_scores)) if pair_scores else 0.0)
            names.append(f"{prefix}.harmonic.H{h}.mean_cosine_resonance")
        return vals, names

    def _global_features(self, geo: dict[str, dict[str, float]], helio: dict[str, dict[str, float]] | None) -> tuple[list[float], list[str]]:
        vals, names = [], []
        if "sun" in geo and "moon" in geo:
            phase = angular_distance(geo["sun"]["longitude"], geo["moon"]["longitude"])
            vals += [math.sin(math.radians(phase)), math.cos(math.radians(phase)), phase / 180.0]
            names += ["geo.lunar_phase.sin", "geo.lunar_phase.cos", "geo.lunar_phase.normalized"]
        for body in CORE_BODIES:
            s = geo.get(body, {})
            speed = abs(_safe(s.get("speed_longitude", 0.0)))
            vals.append(float(speed < self.config.station_speed_threshold))
            names.append(f"geo.{body}.station_proximity")
        # Declination geometry: parallel/contra-parallel proximity for major bodies.
        for a, b in FINANCIAL_PAIRS:
            if a in geo and b in geo:
                da, db = geo[a]["declination"], geo[b]["declination"]
                vals.extend([max(0.0, 1.0 - abs(da - db) / 1.0), max(0.0, 1.0 - abs(da + db) / 1.0)])
                names.extend([f"geo.declination.{a}_{b}.parallel", f"geo.declination.{a}_{b}.contraparallel"])
        if helio is not None:
            vals += [np.mean([_safe(helio.get(b, {}).get("latitude", 0.0)) for b in CORE_BODIES]),
                     np.std([_safe(helio.get(b, {}).get("latitude", 0.0)) for b in CORE_BODIES])]
            names += ["helio.latitude.mean", "helio.latitude.std"]
        return vals, names

    def _row(self, timestamp: pd.Timestamp) -> tuple[list[float], list[str]]:
        geo = self.geo.state(timestamp.to_pydatetime())
        helio = self.helio.state(timestamp.to_pydatetime()) if self.helio else None
        vals, names = [], []
        for fn in (lambda: self._body_features(geo, "geo"),
                   lambda: self._aspect_features(geo, "geo"),
                   lambda: self._financial_geometry(geo, "geo"),
                   lambda: self._harmonics(geo, "geo"),
                   lambda: self._global_features(geo, helio)):
            v, n = fn(); vals.extend(v); names.extend(n)
        if helio is not None:
            for fn in (lambda: self._body_features(helio, "helio"),
                       lambda: self._aspect_features(helio, "helio"),
                       lambda: self._financial_geometry(helio, "helio")):
                v, n = fn(); vals.extend(v); names.extend(n)
        # Fixed ordering is part of the dataset contract.
        return vals, names

    def build(self, market_df: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
        market, market_names = build_market_features(market_df)
        astro_rows, astro_names = [], None
        for ts in market_df["date"]:
            row, names = self._row(pd.Timestamp(ts))
            astro_rows.append(row)
            if astro_names is None:
                astro_names = names
        astro = np.asarray(astro_rows, dtype=np.float32)
        if astro.ndim != 2 or astro.shape[0] != market.shape[0]:
            raise RuntimeError("Market and ephemeris tensors are not row-aligned")
        astro = np.nan_to_num(astro, nan=0.0, posinf=0.0, neginf=0.0)
        if self.config.standardize:
            med = np.median(astro, axis=0)
            mad = np.median(np.abs(astro - med), axis=0) + 1e-6
            astro = np.clip((astro - med) / (1.4826 * mad), -8.0, 8.0).astype(np.float32)
        return market.astype(np.float32), astro.astype(np.float32), market_names + astro_names


def build_aligned_tensor(market_path: str | Path, output_path: str | Path, config: FinancialAstroFeatureConfig | None = None) -> dict:
    df = load_market_frame(market_path)
    builder = CanonicalFinancialAstroTensorBuilder(config)
    market, astro, feature_names = builder.build(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, market=market, astro=astro,
                        dates=df["date"].astype("int64").to_numpy(),
                        feature_names=np.asarray(feature_names, dtype=str),
                        market_feature_count=np.asarray([len(build_market_features(df)[1])], dtype=np.int64))
    meta = {
        "rows": int(len(df)), "market_dim": int(market.shape[1]), "astro_dim": int(astro.shape[1]),
        "start": str(df["date"].iloc[0]), "end": str(df["date"].iloc[-1]),
        "backend": (config or FinancialAstroFeatureConfig()).backend,
        "feature_names": feature_names,
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
