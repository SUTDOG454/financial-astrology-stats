from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from astro_rl.config import PPOConfig
from astro_rl.data import FinancialAstroFeatureConfig, build_aligned_tensor, load_market_frame, CanonicalFinancialAstroTensorBuilder
from astro_rl.env import BatchedMarketEnv
from astro_rl.models import HierarchicalPPO
from astro_rl.ppo import PPOTrainer, RolloutBuffer, regime_target_from_market


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: str, device: torch.device):
    z = np.load(path, allow_pickle=False)
    market = torch.from_numpy(z["market"]).float().to(device)
    astro = torch.from_numpy(z["astro"]).float().to(device)
    if market.ndim != 2 or astro.ndim != 2 or market.shape[0] != astro.shape[0]:
        raise ValueError("NPZ must contain aligned 2-D arrays: market and astro")
    if market.shape[1] < 1:
        raise ValueError("market tensor must retain return_1 as column 0 for reward accounting")
    return market, astro


def build_from_market(args, device: torch.device):
    cfg = FinancialAstroFeatureConfig(
        backend=args.ephemeris_backend,
        ephe_path=args.ephe_path,
        jpl_file=args.jpl_file,
        include_heliocentric=not args.no_heliocentric,
        include_extra_bodies=not args.no_extra_bodies,
    )
    df = load_market_frame(args.market)
    builder = CanonicalFinancialAstroTensorBuilder(cfg)
    market_np, astro_np, feature_names = builder.build(df)
    out_npz = Path(args.tensor_out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, market=market_np, astro=astro_np,
                        dates=df["date"].astype("int64").to_numpy(),
                        feature_names=np.asarray(feature_names, dtype=str),
                        market_feature_count=np.asarray([market_np.shape[1]], dtype=np.int64))
    meta = {"rows": len(df), "market_dim": market_np.shape[1], "astro_dim": astro_np.shape[1],
            "start": str(df.date.iloc[0]), "end": str(df.date.iloc[-1]),
            "ephemeris_backend": args.ephemeris_backend, "tensor": str(out_npz)}
    out_npz.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return torch.from_numpy(market_np).float().to(device), torch.from_numpy(astro_np).float().to(device), meta


def main():
    p = argparse.ArgumentParser(description="Canonical market x Swiss Ephemeris/JPL hierarchical PPO")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--market", help="CSV/Parquet with date, open, high, low, close, volume")
    source.add_argument("--npz", help="Previously materialized aligned market/astro tensor")
    p.add_argument("--tensor-out", default="artifacts/canonical/market_astro.npz")
    p.add_argument("--ephemeris-backend", choices=["swisseph", "jpl"], default="swisseph")
    p.add_argument("--ephe-path", default=None)
    p.add_argument("--jpl-file", default=None, help="Local JPL binary ephemeris accepted by Swiss Ephemeris")
    p.add_argument("--no-heliocentric", action="store_true")
    p.add_argument("--no-extra-bodies", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--updates", type=int, default=200)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/regime_ppo")
    args = p.parse_args()

    cfg = PPOConfig(updates=args.updates, num_envs=args.envs, rollout_steps=args.rollout, seed=args.seed)
    seed_everything(cfg.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    if args.npz:
        market, astro = load_npz(args.npz, device)
        tensor_meta = {"tensor": args.npz, "market_dim": market.shape[1], "astro_dim": astro.shape[1]}
    else:
        market, astro, tensor_meta = build_from_market(args, device)

    cfg.market_dim = int(market.shape[1])
    cfg.astro_dim = int(astro.shape[1])

    env = BatchedMarketEnv(
        market, astro, num_envs=cfg.num_envs, episode_len=cfg.rollout_steps,
        transaction_cost_bps=cfg.transaction_cost_bps,
        slippage_bps=cfg.slippage_bps, device=device, seed=cfg.seed,
    )
    market_dim, astro_dim = env.state_dim
    model = HierarchicalPPO(market_dim, astro_dim, cfg.hidden_dim, cfg.regime_dim, cfg.position_std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    trainer = PPOTrainer(model, optimizer, cfg)

    state = env.reset()
    best_equity = -float("inf")
    history = []

    for update in range(1, cfg.updates + 1):
        buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, market_dim, astro_dim, device)
        episode_count = 0
        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                action, regime, logp, _, value, _ = model.act(state.market, state.astro)
                target = regime_target_from_market(state.market, cfg.regime_dim)
            next_state, reward, done = env.step(action)
            buffer.add(state.market, state.astro, action, regime, reward, done, value, logp, target)
            episode_count += int(done.sum())
            state = next_state

        with torch.no_grad():
            _, _, _, _, last_value, _ = model.act(state.market, state.astro, deterministic=True)
        advantages, returns = buffer.finish(last_value, cfg.gamma, cfg.gae_lambda)
        batch = buffer.flatten(advantages, returns)
        metrics = trainer.update(batch, advantages, returns)
        metrics.update(update=update, mean_return_target=float(returns.mean()), mean_advantage=float(advantages.mean()),
                       episodes=episode_count, device=str(device))
        history.append(metrics)

        with torch.no_grad():
            eval_state = env.reset()
            eval_equity = torch.ones(cfg.num_envs, device=device)
            for _ in range(min(cfg.rollout_steps, 128)):
                action, _, _, _, _, _ = model.act(eval_state.market, eval_state.astro, deterministic=True)
                eval_state, reward, _ = env.step(action)
                eval_equity *= torch.exp(reward.clamp(-0.2, 0.2))
            mean_equity = float(eval_equity.mean())

        if mean_equity > best_equity:
            best_equity = mean_equity
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": vars(cfg),
                        "tensor_meta": tensor_meta, "update": update, "mean_equity": mean_equity},
                       os.path.join(args.out, "best.pt"))
        if update == 1 or update % 10 == 0:
            print(json.dumps({**metrics, "eval_equity": mean_equity}, sort_keys=True))

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": vars(cfg),
                "tensor_meta": tensor_meta, "history": history}, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
