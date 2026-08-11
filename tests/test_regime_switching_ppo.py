import torch

from astro_rl.config import PPOConfig
from astro_rl.env import BatchedMarketEnv, build_demo_dataset
from astro_rl.models import HierarchicalPPO
from astro_rl.ppo import PPOTrainer, RolloutBuffer, regime_target_from_market


def test_end_to_end_rollout_and_update():
    cfg = PPOConfig(num_envs=4, rollout_steps=8, epochs=1, minibatch_size=16, hidden_dim=32)
    market, astro = build_demo_dataset(n=256, market_dim=cfg.market_dim, astro_dim=cfg.astro_dim, seed=cfg.seed)
    env = BatchedMarketEnv(market, astro, cfg.num_envs, cfg.rollout_steps, device="cpu", seed=cfg.seed)
    model = HierarchicalPPO(cfg.market_dim, cfg.astro_dim, cfg.hidden_dim, cfg.regime_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    trainer = PPOTrainer(model, optimizer, cfg)
    state = env.reset()
    buf = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, cfg.market_dim, cfg.astro_dim, "cpu")

    for _ in range(cfg.rollout_steps):
        with torch.no_grad():
            action, regime, logp, _, value, _ = model.act(state.market, state.astro)
            target = regime_target_from_market(state.market, cfg.regime_dim)
        nxt, reward, done = env.step(action)
        buf.add(state.market, state.astro, action, regime, reward, done, value, logp, target)
        state = nxt

    with torch.no_grad():
        _, _, _, _, last_value, _ = model.act(state.market, state.astro, deterministic=True)
    adv, returns = buf.finish(last_value, cfg.gamma, cfg.gae_lambda)
    batch = buf.flatten(adv, returns)
    metrics = trainer.update(batch, adv, returns)
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert batch["market"].shape[0] == cfg.num_envs * cfg.rollout_steps
