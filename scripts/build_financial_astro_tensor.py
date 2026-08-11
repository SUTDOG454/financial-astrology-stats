from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astro_rl.data import FinancialAstroFeatureConfig, build_aligned_tensor


def main():
    p = argparse.ArgumentParser(description="Build canonical market x Swiss Ephemeris/JPL tensor")
    p.add_argument("--market", required=True, help="CSV/Parquet: date,open,high,low,close,volume")
    p.add_argument("--out", default="artifacts/canonical/market_astro.npz")
    p.add_argument("--backend", choices=["swisseph", "jpl"], default="swisseph")
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--jpl-file", default=None)
    p.add_argument("--no-heliocentric", action="store_true")
    p.add_argument("--no-extra-bodies", action="store_true")
    args = p.parse_args()
    cfg = FinancialAstroFeatureConfig(
        backend=args.backend, ephe_path=args.ephe_path, jpl_file=args.jpl_file,
        include_heliocentric=not args.no_heliocentric,
        include_extra_bodies=not args.no_extra_bodies,
    )
    meta = build_aligned_tensor(args.market, args.out, cfg)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
