from __future__ import annotations

import math
import torch
from torch import nn


class RolloutBuffer:
    def __init__(self, steps, envs, market_dim, astro_dim, device):
        shape = (steps, envs)
        self.market = torch.empty((*shape, market_dim), device=device)
        self.astro = torch.empty((*shape, astro_dim), device=device)
        self.actions = torch.empty((*shape, 1), device=device)
        self.regimes = torch.empty(shape, dtype=torch.long, device=device)
        self.rewards = torch.empty(shape, device=device)
        self.dones = torch.empty(shape, device=device)
        self.values = torch.empty(shape, device=device)
        self.logp = torch.empty(shape, device=device)
        self.regime_targets = torch.empty(shape, dtype=torch.long, device=device)
        self.ptr = 0

    def add(self, market, astro, action, regime, reward, done, value, logp, regime_target):
        i = self.ptr
        self.market[i], self.astro[i] = market, astro
        self.actions[i], self.regimes[i] = action, regime
        self.rewards[i], self.dones[i] = reward, done.float()
        self.values[i], self.logp[i] = value, logp
        self.regime_targets[i] = regime_target
        self.ptr += 1

    def finish(self, last_value, gamma, gae_lambda):
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.rewards.shape[1], device=self.rewards.device)
        for t in reversed(range(self.ptr)):
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            adv[t] = gae
        return adv, adv + self.values

    def flatten(self, adv, returns):
        n = self.ptr * self.market.shape[1]
        return {
            "market": self.market[:self.ptr].reshape(n, -1), "astro": self.astro[:self.ptr].reshape(n, -1),
            "actions": self.actions[:self.ptr].reshape(n, -1), "regimes": self.regimes[:self.ptr].reshape(n),
            "old_logp": self.logp[:self.ptr].reshape(n), "old_values": self.values[:self.ptr].reshape(n),
            "advantages": adv.reshape(n), "returns": returns.reshape(n),
            "regime_targets": self.regime_targets[:self.ptr].reshape(n),
        }


def regime_target_from_market(market: torch.Tensor, regime_dim: int) -> torch.Tensor:
    """Contemporaneous pseudo-labels; no future return information is used.

    Canonical market layout: 0 return_1, 8 volatility_5, 9 volatility_20,
    11 trend_20. The legacy 8-column demo layout is retained for compatibility.
    """
    if market.shape[1] >= 12:
        vol = 0.5 * (market[:, 8].abs() + market[:, 9].abs())
        trend = market[:, 11].abs()
    else:
        vol = market[:, 2].abs()
        trend = market[:, 1].abs()
    score = torch.sigmoid(6.0 * vol) + 0.5 * torch.sigmoid(6.0 * trend)
    bins = torch.linspace(0.0, 1.5, regime_dim + 1, device=market.device)[1:-1]
    return torch.bucketize(score, bins).clamp_max(regime_dim - 1)


class PPOTrainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, cfg):
        self.model, self.optimizer, self.cfg = model, optimizer, cfg

    def update(self, batch, advantages, returns):
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = adv.numel()
        total = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0, "regime": 0.0}
        for _ in range(self.cfg.epochs):
            perm = torch.randperm(n, device=adv.device)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = perm[start:start + self.cfg.minibatch_size]
                logp, entropy, value, regime_logits = self.model.evaluate(
                    batch["market"][idx], batch["astro"][idx], batch["regimes"][idx], batch["actions"][idx]
                )
                ratio = (logp - batch["old_logp"][idx]).exp()
                policy_loss = -torch.min(ratio * adv[idx], ratio.clamp(1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv[idx]).mean()
                value_old = batch["old_values"][idx]
                value_clipped = value_old + (value - value_old).clamp(-self.cfg.value_clip_eps, self.cfg.value_clip_eps)
                value_loss = 0.5 * torch.max((value - returns[idx]).pow(2), (value_clipped - returns[idx]).pow(2)).mean()
                entropy_loss = entropy.mean()
                regime_loss = torch.nn.functional.cross_entropy(regime_logits, batch["regime_targets"][idx])
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss + self.cfg.regime_coef * regime_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                for k, v in (("loss", loss), ("policy", policy_loss), ("value", value_loss), ("entropy", entropy_loss), ("regime", regime_loss)):
                    total[k] += float(v.detach())
        denom = self.cfg.epochs * math.ceil(n / self.cfg.minibatch_size)
        return {k: v / denom for k, v in total.items()}
