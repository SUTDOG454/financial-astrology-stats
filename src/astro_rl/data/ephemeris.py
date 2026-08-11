from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd

try:
    import swisseph as swe
except ImportError as exc:  # pragma: no cover
    swe = None
    _SWE_IMPORT_ERROR = exc


DEFAULT_BODIES = {
    "sun": 0, "moon": 1, "mercury": 2, "venus": 3, "mars": 4,
    "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8, "pluto": 9,
    "chiron": -1, "ceres": 1, "sedna": 90377,
}


@dataclass(frozen=True)
class EphemerisConfig:
    backend: str = "swisseph"  # swisseph or jpl
    ephe_path: str | None = None
    jpl_file: str | None = None
    include_extra_bodies: bool = True
    heliocentric: bool = False


class SwissEphemerisProvider:
    """Deterministic UTC timestamp -> planetary state provider."""

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
            self.bodies = {k: v for k, v in self.bodies.items() if k in {
                "sun", "moon", "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"
            }}

    @staticmethod
    def _jd(ts: pd.Timestamp) -> float:
        hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0 + ts.microsecond / 3.6e9
        return float(swe.julday(ts.year, ts.month, ts.day, hour))

    def _flags(self) -> int:
        flags = swe.FLG_SPEED
        flags |= swe.FLG_HELCTR if self.config.heliocentric else swe.FLG_SWIEPH
        return flags

    def _planet_id(self, name: str, body: int) -> int:
        if name == "chiron":
            return swe.CHIRON
        return body if body < 10 else swe.AST_OFFSET + body

    def state(self, timestamp: datetime) -> dict[str, dict[str, float]]:
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        jd = self._jd(ts)
        flags = self._flags()
        result: dict[str, dict[str, float]] = {}
        for name, body in self.bodies.items():
            ipl = self._planet_id(name, body)
            try:
                xx, retflags = swe.calc_ut(jd, ipl, flags)
            except Exception:
                continue
            lon, lat, dist, speed_lon, speed_lat, _ = xx
            try:
                eq, _ = swe.calc_ut(jd, ipl, flags | swe.FLG_EQUATORIAL)
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
        rows = []
        for ts in timestamps:
            row = {"date": pd.Timestamp(ts)}
            for body, vals in self.state(ts).items():
                for key, value in vals.items():
                    row[f"{body}.{key}"] = value
            rows.append(row)
        return pd.DataFrame(rows)
