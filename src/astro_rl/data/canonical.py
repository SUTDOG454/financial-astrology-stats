from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json
import numpy as np
import torch


@dataclass(frozen=True)
class DatasetSpec:
    schema_version: str = "astro-market-tensor.v1"
    frequency: str = "1D"
    price_field: str = "close"
    feature_scaling: str = "fit-on-train-only"
    ephemeris_backend: str = "swiss"


@dataclass
class CanonicalDataset:
    timestamps: np.ndarray
    market: np.ndarray
    astro: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any]
    regime_labels: np.ndarray | None = None

    def validate(self) -> None:
        n = len(self.timestamps)
        if self.market.ndim != 2 or self.astro.ndim != 2:
            raise ValueError("market and astro must be rank-2 arrays")
        if self.market.shape[0] != n or self.astro.shape[0] != n:
            raise ValueError("timestamps, market and astro must be time-aligned")
        if len(self.feature_names) != self.market.shape[1] + self.astro.shape[1]:
            raise ValueError("feature_names length does not match tensor width")
        if not np.isfinite(self.market).all() or not np.isfinite(self.astro).all():
            raise ValueError("feature tensors contain NaN/Inf")

    def to_torch(self, device: str | torch.device = "cpu"):
        self.validate()
        return (
            torch.as_tensor(self.market, dtype=torch.float32, device=device),
            torch.as_tensor(self.astro, dtype=torch.float32, device=device),
        )

    def metadata_json(self) -> str:
        payload = dict(self.metadata)
        payload["schema"] = asdict(DatasetSpec(**payload["schema"])) if isinstance(payload.get("schema"), dict) else payload.get("schema")
        return json.dumps(payload, sort_keys=True, default=str)
