from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import numpy as np

try:
    import swisseph as swe
except ImportError:  # pragma: no cover
    swe = None


@dataclass(frozen=True)
class EphemerisConfig:
    backend: str = "swiss"  # swiss | jpl
    ephe_path: str | None = None
    jpl_file: str | None = None
    planets: tuple[str, ...] = (
        "sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn",
        "uranus", "neptune", "pluto", "chiron", "ceres", "true_node",
    )


_PLANETS = {
    "sun": "SE_SUN", "moon": "SE_MOON", "mercury": "SE_MERCURY", "venus": "SE_VENUS",
    "mars": "SE_MARS", "jupiter": "SE_JUPITER", "saturn": "SE_SATURN", "uranus": "SE_URANUS",
    "neptune": "SE_NEPTUNE", "pluto": "SE_PLUTO", "chiron": "SE_CHIRON", "ceres": "SE_CERES",
    "true_node": "SE_TRUE_NODE",
}


class SwissEphemerisProvider:
    """Deterministic UTC -> geocentric ecliptic ephemeris adapter.

    Swiss Ephemeris is the normal backend. When a JPL DE file is configured, the
    adapter requests JPL calculations through Swiss Ephemeris. The feature schema
    is identical across backends, allowing backend-ablation and cross-validation.
    """

    def __init__(self, cfg: EphemerisConfig):
        if swe is None:
            raise ImportError("pyswisseph is required for ephemeris ingestion")
        self.cfg = cfg
        if cfg.ephe_path:
            swe.set_ephe_path(cfg.ephe_path)
        if cfg.backend == "jpl":
            if not cfg.jpl_file:
                raise ValueError("backend='jpl' requires jpl_file")
            if not hasattr(swe, "set_jpl_file"):
                raise RuntimeError("installed pyswisseph does not expose set_jpl_file")
            swe.set_jpl_file(cfg.jpl_file)

    def _flags(self):
        return (swe.FLG_JPLEPH if self.cfg.backend == "jpl" else swe.FLG_SWIEPH) | swe.FLG_SPEED

    @staticmethod
    def _jd(dt: datetime) -> float:
        u = (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
        hour = u.hour + u.minute / 60.0 + u.second / 3600.0 + u.microsecond / 3.6e9
        return float(swe.julday(u.year, u.month, u.day, hour, swe.GREG_CAL))

    def row(self, dt: datetime) -> tuple[np.ndarray, list[str]]:
        jd = self._jd(dt)
        flags = self._flags()
        vals: list[float] = []
        names: list[str] = []
        for name in self.cfg.planets:
            body = getattr(swe, _PLANETS[name])
            xx, _ = swe.calc_ut(jd, body, flags)
            lon, lat, dist, lon_speed, lat_speed, dist_speed = map(float, xx[:6])
            r = math.radians(lon)
            vals.extend([
                math.sin(r), math.cos(r), math.sin(math.radians(lat)), math.cos(math.radians(lat)),
                lon_speed / 2.0, lat_speed / 2.0, math.log1p(max(dist, 0.0)), float(lon_speed < 0),
            ])
            names.extend([
                f"{name}_lon_sin", f"{name}_lon_cos", f"{name}_lat_sin", f"{name}_lat_cos",
                f"{name}_lon_speed", f"{name}_lat_speed", f"{name}_log_distance", f"{name}_retrograde",
            ])
        return np.asarray(vals, dtype=np.float32), names

    def matrix(self, timestamps) -> tuple[np.ndarray, list[str]]:
        rows, names = [], []
        for ts in timestamps:
            row, names = self.row(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
            rows.append(row)
        return np.vstack(rows).astype(np.float32), names
