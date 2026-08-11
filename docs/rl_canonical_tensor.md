# Canonical Market × Ephemeris × Financial-Astrology PPO Tensor

The regime-switching PPO trainer no longer needs the synthetic `build_demo_dataset()` path. Production training accepts either a materialized canonical NPZ or an OHLCV CSV/Parquet file and constructs the aligned tensor in one deterministic pipeline.

## Data contract

For each market timestamp `t`:

`market[t] = [r1, mom5, mom20, vol20, atr14_rel, drawdown, volume_z20, rsi14]`

`astro[t] = [raw ephemeris encodings, station/retrograde features, lunar phase, classical aspect resonances, Magi-style geometry resonances, relative speeds, H1-H24 harmonic density]`

All features are computed from information available at `t`; no future return is used by the feature generator. Standardization parameters are fitted on the first training fraction and then applied to the complete tensor, so the preprocessing itself is time-ordered.

## Ephemeris backends

`--ephemeris-backend swiss` uses Swiss Ephemeris. `--ephemeris-backend jpl --jpl-file <DE440 file>` requests JPL mode through Swiss Ephemeris. DE440 covers approximately 1550–2650, while DE441 is intended for a much longer historical span; select the file according to the research interval.

## Financial-astrology registry

The feature layer is deliberately hypothesis-oriented. It includes configurable classical aspects, Magi-style 36/72/108/144-degree geometry, lunar phase, planetary stations/retrograde state, relative planetary speed, selected financial pair interactions, and H1-H24 harmonic density. These are candidate explanatory variables, not assumed causal predictors. Every group can be removed in ablation experiments.

## Training

```bash
python train/train_regime_switching_ppo.py \
  --market data/SPX.csv \
  --timestamp-col timestamp \
  --ephemeris-backend swiss \
  --device cuda \
  --materialize artifacts/spx_canonical.npz
```

For JPL DE440:

```bash
python train/train_regime_switching_ppo.py \
  --market data/SPX.csv \
  --ephemeris-backend jpl \
  --jpl-file /path/to/de440.eph \
  --device cuda
```

The existing `--npz` path remains supported so feature construction can be decoupled from PPO training and reused for walk-forward experiments.
