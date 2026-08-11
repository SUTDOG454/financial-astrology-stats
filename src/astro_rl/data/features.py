from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    aspect_orb_deg: float = 3.0
    harmonic_orb_deg: float = 2.0
    harmonics: tuple[int, ...] = tuple(range(1, 25))
    financial_pairs: tuple[tuple[str, str], ...] = (
        ("jupiter", "saturn"), ("jupiter", "uranus"), ("saturn", "uranus"),
        ("venus", "jupiter"), ("venus", "uranus"), ("mars", "jupiter"),
        ("sun", "jupiter"), ("sun", "saturn"), ("mercury", "jupiter"),
        ("mercury", "uranus"), ("saturn", "neptune"), ("uranus", "neptune"),
    )
    classical_aspects: tuple[float, ...] = (0.0, 60.0, 90.0, 120.0, 180.0)
    magi_geometry: tuple[float, ...] = (0.0, 36.0, 72.0, 108.0, 144.0, 180.0)


def _circular_distance(x: np.ndarray, target: float) -> np.ndarray:
    d = np.abs(((x - target + 180.0) % 360.0) - 180.0)
    return d


def _resonance(delta: np.ndarray, angle: float, orb: float) -> np.ndarray:
    return np.exp(-0.5 * (_circular_distance(delta, angle) / max(orb, 1e-6)) ** 2).astype(np.float32)


class FinancialAstroFeatureBuilder:
    """Builds testable financial-astrology features without future leakage.

    The feature registry deliberately separates raw ephemeris, classical aspects,
    harmonic structure and Magi-style geometry. It does not assign empirical
    truth to any astrological hypothesis; every feature can be ablated and tested.
    """

    def __init__(self, ephemeris_names: list[str], cfg: FeatureConfig | None = None):
        self.names = ephemeris_names
        self.cfg = cfg or FeatureConfig()
        self.lon_idx = {}
        self.speed_idx = {}
        for i, name in enumerate(ephemeris_names):
            if name.endswith("_lon_sin"):
                self.lon_idx[name[:-8]] = (i, i + 1)
            elif name.endswith("_lon_speed"):
                self.speed_idx[name[:-10]] = i

    def _longitudes(self, e: np.ndarray) -> dict[str, np.ndarray]:
        return {p: np.degrees(np.arctan2(e[:, a], e[:, b])) % 360.0 for p, (a, b) in self.lon_idx.items()}

    def build(self, e: np.ndarray) -> tuple[np.ndarray, list[str]]:
        lon = self._longitudes(e)
        out, names = [], []
        if "sun" in lon and "moon" in lon:
            phase = (lon["moon"] - lon["sun"]) % 360.0
            out += [np.sin(np.radians(phase)), np.cos(np.radians(phase)), phase / 180.0 - 1.0]
            names += ["lunar_phase_sin", "lunar_phase_cos", "lunar_phase_norm"]

        for planet in sorted(lon):
            if planet in self.speed_idx:
                s = e[:, self.speed_idx[planet]]
                out += [np.abs(s), np.exp(-np.abs(s) / 0.05)]
                names += [f"{planet}_speed_abs", f"{planet}_station_proximity"]

        for a, b in self.cfg.financial_pairs:
            if a not in lon or b not in lon:
                continue
            delta = (lon[a] - lon[b]) % 360.0
            for angle in self.cfg.classical_aspects:
                out.append(_resonance(delta, angle, self.cfg.aspect_orb_deg))
                names.append(f"aspect_{a}_{b}_{int(angle)}")
            for angle in self.cfg.magi_geometry:
                out.append(_resonance(delta, angle, self.cfg.aspect_orb_deg))
                names.append(f"magi_geometry_{a}_{b}_{int(angle)}")
            # Directional phase: useful for applying/separating experiments.
            if a in self.speed_idx and b in self.speed_idx:
                rel_speed = e[:, self.speed_idx[a]] - e[:, self.speed_idx[b]]
                out.append(np.tanh(rel_speed / 0.5))
                names.append(f"relative_speed_{a}_{b}")

        # Harmonic resonance density: a compact H1-H24 representation.
        for h in self.cfg.harmonics:
            density = np.zeros(e.shape[0], dtype=np.float32)
            count = 0
            for a, b in self.cfg.financial_pairs:
                if a not in lon or b not in lon:
                    continue
                delta = (lon[a] - lon[b]) % 360.0
                # Harmonic H is strongest when H*delta is close to 0 mod 360.
                folded = (h * delta) % 360.0
                density += _resonance(folded, 0.0, self.cfg.harmonic_orb_deg)
                count += 1
            if count:
                density /= count
            out.append(density)
            names.append(f"harmonic_h{h}_density")

        x = np.column_stack(out).astype(np.float32) if out else np.empty((e.shape[0], 0), dtype=np.float32)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), names
