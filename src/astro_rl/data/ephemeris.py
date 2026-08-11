from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable

import numpy as np

try:
    import swisseph as swe
except ImportError as exc:  # pragma: no cover
    swe = None
    _SWE_IMPORT_ERROR = exc


DEFAULT_BODIES = {
    "sun": 0, "moon": 1, "mercury": 2, "venus": 3, "mars": 4,
    "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8, "pluto": 9,
    "chiron": 9990, "ceres": 10001, "sedna": 10010,
}


@dataclass(frozen=True)
class EphemerisConfig:
    backend: str = "swisseph"  # swisseph or jpl
    ephe_path: str | None = None
    jpl_file: str | None = None
    include_extra_bodies: bool = True
    heliocentric: bool = False


class SwissEphemerisProvider:
    """Deterministic date -> planetary state provider.

    Swiss Ephemeris exposes `calc_ut` for UT-based calculations and can select
    Swiss, Moshier, or JPL ephemerides through calculation flags. JPL mode is
    activated with `set_jpl_file`; the caller supplies the local kernel path.
    """

    def __init__(self, config: EphemerisConfig | None = None, bodies: dict[str, int] | None = None):
        if swe is None:
            raise ImportError("pyswisseph is required; install pyswisseph") from _SWE_IMPORT_ERROR
        self.config = config or EphemerisConfig()
        if self.config.ephe_path:
            swe.set_ephe_path(self.config.ephe_path)
        if self.config.backend.lower() == "jpl":
            if not self.config.jpl_file:
                raise ValueError("jpl backend requires EphemerisConfig.jpl_file")
            swe.set_jpl_file(self.config.jpl_file)
        elif self.config.backend.lower() != "swisseph":
            raise ValueError("backend must be 'swisseph' or 'jpl'")
        self.bodies = dict(DEFAULT_BODIES if bodies is None else bodies)
        if not self.config.include_extra_bodies:
            self.bodies = {k: v for k, v in self.bodies.items() if k in DEFAULT_BODIES and v < 10}

    @staticmethod
    def _jd(ts: pd.Timestamp) -> float:
        return float(swe.julday(ts.year, ts.month, ts.day, ts.hour + ts.minute / 60.0 + ts.second / 3600.0))

    def _flags(self) -> int:
        flags = swe.FLG_SPEED
        flags |= swe.FLG_HELCTR if self.config.heliocentric else swe.FLG_SWIEPH
        return flags

    def state(self, timestamp: datetime) -> dict[str, dict[str, float]]:
        import pandas as pd
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        jd = self._jd(ts)
        flags = self._flags()
        result: dict[str, dict[str, float]] = {}
        for name, body in self.bodies.items():
            # Chiron/Sedna/Ceres use Swiss asteroid offsets.
            ipl = body if body < 10 else swe.AST_OFFSET + (body - 10000 if body >= 10000 else 2060)
            if name == "chiron":
                ipl = swe.CHIRON
            try:
                xx, retflags = swe.calc_ut(jd, ipl, flags)
            except Exception:
                continue
            lon, lat, dist, speed_lon, speed_lat, _ = xx
            # Equatorial coordinates are requested separately for declination.
            eq_flags = flags | swe.FLG_EQUATORIAL
            try:
                eq, _ = swe.calc_ut(jd, ipl, eq_flags)
                decl = float(eq[1])
            except Exception:
                decl = float("nan")
            result[name] = {
                "longitude": float(lon % 360.0),
                "latitude": float(lat),
                "distance_au": float(dist),
                "speed_longitude": float(speed_lon),
                "speed_latitude": float(speed_lat),
                "declination": decl,
                "retrograde": float(speed_lon < 0.0),
                "ephemeris_flags": float(retflags),
            }
        return result

    def frame(self, timestamps: Iterable[datetime]):
        import pandas as pd
        rows = []
        for ts in timestamps:
            row = {"date": pd.Timestamp(ts)}
            state = self.state(ts)
            for body, vals in state.items():
                for key, value in vals.items():
                    row[f"{body}.{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)
