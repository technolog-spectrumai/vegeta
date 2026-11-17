"""A running mean and standard deviation of the observations (Welford, batched), applied as a normalising layer;
saved in the checkpoint and folded into the ONNX export."""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RunningMeanStd"]


class RunningMeanStd(nn.Module):
    def __init__(self, shape: int, epsilon: float = 1e-4, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(float(epsilon)))
        self.clip = clip
        self.frozen = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.frozen:
            return
        x = x.reshape(-1, x.shape[-1])
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), float(x.shape[0])
        delta = bm - self.mean
        tot = self.count + bc
        new_mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        new_var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)
