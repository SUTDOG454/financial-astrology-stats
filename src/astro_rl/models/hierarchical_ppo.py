from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Normal


class HierarchicalPPO(nn.Module):
    """Two-level actor-critic: latent regime selector + regime-conditioned position policy."""

    def __init__(self, market_dim: int, astro_dim: int, hidden_dim: int = 128,
                 regime_dim: int = 5, position_std: float = 0.35):
        super().__init__()
        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.astro_encoder = nn.Sequential(
            nn.Linear(astro_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()
        )
        self.regime_head = nn.Linear(hidden_dim, regime_dim)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + regime_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.value_head = nn.Linear(hidden_dim + regime_dim, 1)
        self.log_std = nn.Parameter(torch.tensor(float(position_std)).log())
        self.regime_dim = regime_dim

    def encode(self, market, astro):
        return self.fusion(torch.cat([self.market_encoder(market), self.astro_encoder(astro)], dim=-1))

    def _conditioned_heads(self, z, regime):
        one_hot = torch.nn.functional.one_hot(regime, self.regime_dim).float()
        h = torch.cat([z, one_hot], dim=-1)
        mean = torch.tanh(self.policy_head(h))
        value = self.value_head(h).squeeze(-1)
        return mean, value

    def distributions(self, market, astro, regime=None):
        z = self.encode(market, astro)
        regime_logits = self.regime_head(z)
        regime_dist = Categorical(logits=regime_logits)
        if regime is None:
            regime = regime_dist.sample()
        mean, value = self._conditioned_heads(z, regime)
        std = self.log_std.exp().clamp(0.03, 1.0)
        action_dist = Normal(mean, std)
        return regime_dist, action_dist, value, regime_logits

    def act(self, market, astro, deterministic=False):
        z = self.encode(market, astro)
        regime_logits = self.regime_head(z)
        regime_dist = Categorical(logits=regime_logits)
        regime = regime_dist.probs.argmax(-1) if deterministic else regime_dist.sample()
        mean, value = self._conditioned_heads(z, regime)
        action_dist = Normal(mean, self.log_std.exp().clamp(0.03, 1.0))
        # Keep the sampled Gaussian action for PPO's exact log-probability ratio.
        # The environment clips the executable target position to [-1, 1].
        action = mean if deterministic else action_dist.rsample()
        logp = regime_dist.log_prob(regime) + action_dist.log_prob(action).sum(-1)
        entropy = regime_dist.entropy() + action_dist.entropy().sum(-1)
        return action, regime, logp, entropy, value, regime_logits

    def evaluate(self, market, astro, regime, action):
        regime_dist, action_dist, value, regime_logits = self.distributions(market, astro, regime)
        logp = regime_dist.log_prob(regime) + action_dist.log_prob(action).sum(-1)
        entropy = regime_dist.entropy() + action_dist.entropy().sum(-1)
        return logp, entropy, value, regime_logits
