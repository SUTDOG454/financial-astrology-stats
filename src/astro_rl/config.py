from dataclasses import dataclass


@dataclass
class PPOConfig:
    market_dim: int = 8
    astro_dim: int = 16
    hidden_dim: int = 128
    regime_dim: int = 5
    num_envs: int = 64
    rollout_steps: int = 256
    updates: int = 200
    minibatch_size: int = 2048
    epochs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    regime_coef: float = 0.10
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    position_std: float = 0.35
    initial_equity: float = 1.0
    transaction_cost_bps: float = 2.0
    slippage_bps: float = 1.0
    seed: int = 42
