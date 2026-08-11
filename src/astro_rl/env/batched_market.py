from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass
class BatchState:
    market: torch.Tensor
    astro: torch.Tensor


def build_demo_dataset(n: int = 12000, market_dim: int = 8, astro_dim: int = 16, seed: int = 42):
    """Deterministic synthetic data for smoke tests; replace with aligned market/ephemeris arrays."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n, dtype=torch.float32)
    cyc = torch.stack([torch.sin(t / p) for p in (17, 29, 61, 127)], dim=1)
    astro = torch.randn(n, astro_dim, generator=g) * 0.2
    astro[:, :4] += cyc
    returns = 0.0002 * torch.randn(n, generator=g)
    returns += 0.0015 * torch.tanh(astro[:, 0] - 0.5 * astro[:, 1])
    vol = torch.zeros(n)
    vol[0] = 0.01
    for i in range(1, n):
        vol[i] = 0.94 * vol[i - 1] + 0.06 * (abs(returns[i - 1]) + 1e-4)
    market = torch.zeros(n, market_dim)
    market[:, 0] = returns
    market[:, 1] = torch.cumsum(returns, 0)
    market[:, 2] = vol
    market[:, 3] = torch.sin(t / 20)
    market[:, 4] = torch.cos(t / 20)
    market[:, 5:] = torch.randn(n, market_dim - 5, generator=g) * 0.01
    return market, astro


class BatchedMarketEnv:
    """GPU-native vectorized market environment.

    Each environment samples a contiguous segment from the same time-aligned dataset.
    The action is a target position in [-1, 1]. Reward is net return after turnover costs.
    """

    def __init__(self, market: torch.Tensor, astro: torch.Tensor, num_envs: int = 64,
                 episode_len: int = 256, transaction_cost_bps: float = 2.0,
                 slippage_bps: float = 1.0, device: str | torch.device = "cpu", seed: int = 42):
        assert market.ndim == 2 and astro.ndim == 2 and market.shape[0] == astro.shape[0]
        self.market = market.to(device=device, dtype=torch.float32)
        self.astro = astro.to(device=device, dtype=torch.float32)
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.device = torch.device(device)
        self.cost = (transaction_cost_bps + slippage_bps) / 10_000.0
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.max_start = max(1, self.market.shape[0] - episode_len - 2)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.start = torch.zeros_like(self.t)
        self.position = torch.zeros(num_envs, device=self.device)
        self.equity = torch.ones(num_envs, device=self.device)
        self.steps = torch.zeros_like(self.t)

    @property
    def state_dim(self):
        return self.market.shape[1], self.astro.shape[1]

    def reset(self):
        self.start = torch.randint(0, self.max_start, (self.num_envs,), generator=self.gen, device=self.device)
        self.t = self.start.clone()
        self.position.zero_()
        self.equity.fill_(1.0)
        self.steps.zero_()
        return self._state()

    def _state(self):
        return BatchState(self.market[self.t], self.astro[self.t])

    def step(self, action: torch.Tensor):
        action = action.squeeze(-1).clamp(-1.0, 1.0)
        nxt = (self.t + 1).clamp_max(self.market.shape[0] - 1)
        asset_return = self.market[nxt, 0]
        turnover = (action - self.position).abs()
        net_return = action * asset_return - turnover * self.cost
        self.equity *= torch.exp(net_return.clamp(-0.2, 0.2))
        self.position = action
        self.t = nxt
        self.steps += 1
        done = self.steps >= self.episode_len
        reward = net_return
        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            self.start[idx] = torch.randint(0, self.max_start, (idx.numel(),), generator=self.gen, device=self.device)
            self.t[idx] = self.start[idx]
            self.position[idx] = 0.0
            self.equity[idx] = 1.0
            self.steps[idx] = 0
        return self._state(), reward, done
