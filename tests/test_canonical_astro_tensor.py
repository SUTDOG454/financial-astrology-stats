import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astro_rl.data.features import FinancialAstroFeatureBuilder, FeatureConfig


def _ephemeris_row(planets):
    names, values = [], []
    for p, lon, speed in planets:
        r = np.radians(lon)
        values.extend([np.sin(r), np.cos(r), 0.0, 1.0, speed / 2.0, 0.0, 0.5, float(speed < 0)])
        names.extend([f"{p}_lon_sin", f"{p}_lon_cos", f"{p}_lat_sin", f"{p}_lat_cos", f"{p}_lon_speed", f"{p}_lat_speed", f"{p}_log_distance", f"{p}_retrograde"])
    return np.asarray(values, dtype=np.float32), names


def test_feature_tensor_is_finite_and_deterministic():
    row, names = _ephemeris_row([
        ("sun", 0.0, 1.0), ("moon", 90.0, 13.0), ("jupiter", 60.0, 0.08),
        ("saturn", 120.0, 0.03), ("uranus", 132.0, 0.01), ("venus", 30.0, 1.2),
        ("mars", 210.0, 0.5), ("mercury", 15.0, 1.0), ("neptune", 300.0, 0.006),
    ])
    e = np.repeat(row[None, :], 8, axis=0)
    x1, n1 = FinancialAstroFeatureBuilder(names, FeatureConfig()).build(e)
    x2, n2 = FinancialAstroFeatureBuilder(names, FeatureConfig()).build(e)
    assert x1.shape == x2.shape
    assert n1 == n2
    assert np.isfinite(x1).all()
    assert np.allclose(x1, x2)
    assert any(n.startswith("magi_geometry_") for n in n1)
    assert any(n == "harmonic_h11_density" for n in n1)
