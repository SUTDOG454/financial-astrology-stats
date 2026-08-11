from __future__ import annotations

import argparse
import json
import os
import random
import numpy as np
import torch

from astro_rl.config import PPOConfig
from astro_rl.env import BatchedMarketEnv, build_demo_dataset
from astro_rl.models import HierarchicalPPO
from astro_rl.ppo import PPOTrainer, RolloutBuffer, regime_target_from_market


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: str, device: torch.device):
    z = np.load(path)
    market = torch.from_numpy(z["market"]).float().to(device)
    astro = torch.from_numpy(z["astro"]).float().to(device)
    if market.ndim != 2 or astro.ndim != 2 or market.shape[0] != astro.shape[0]:
        raise ValueError("NPZ must contain aligned 2-D arrays: market and astro")
    return market, astro


def main():
    p = argparse.ArgumentParser(description="End-to-end GPU-vectorized regime-switching PPO")
    p.add_argument("--npz", default=None, help="Aligned dataset with market and astro arrays")
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
    else:
        market, astro = build_demo_dataset(device=device if False else "cpu", seed=cfg.seed)
        market, astro = market.to(device), astro.to(device)

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
        episode_rewards = torch.zeros(cfg.num_envs, device=device)
        episode_count = 0

        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                action, regime, logp, _, value, _ = model.act(state.market, state.astro)
                target = regime_target_from_market(state.market, cfg.regime_dim)
            next_state, reward, done = env.step(action)
            buffer.add(state.market, state.astro, action, regime, reward, done, value, logp, target)
            episode_rewards += reward
            episode_count += int(done.sum())
            state = next_state

        with torch.no_grad():
            _, _, _, _, last_value, _ = model.act(state.market, state.astro, deterministic=True)
        advantages, returns = buffer.finish(last_value, cfg.gamma, cfg.gae_lambda)
        batch = buffer.flatten(advantages, returns)
        metrics = trainer.update(batch, advantages, returns)
        mean_reward = float(batch["returns"].mean())
        mean_adv = float(advantages.mean())
        metrics.update(update=update, mean_return_target=mean_reward, mean_advantage=mean_adv,
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
                        "update": update, "mean_equity": mean_equity}, os.path.join(args.out, "best.pt"))

        if update == 1 or update % 10 == 0:
            print(json.dumps({**metrics, "eval_equity": mean_equity}, sort_keys=True))

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": vars(cfg),
                "history": history}, os.path.join(args.out, "last.pt"))
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
