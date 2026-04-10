"""Shared rotary time embedding components for RoTE."""

from __future__ import annotations

import torch
import torch.nn as nn

from modules.fusion import (
    DAY_BASE,
    DAY_WEIGHT,
    MONTH_BASE,
    MONTH_WEIGHT,
    YEAR_BASE,
    YEAR_WEIGHT,
    coarse_to_fine_fusion,
)


class RotaryTimeEmbedding(nn.Module):
    """Standard RoPE driven by timestamps instead of token positions."""

    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bs,d->bsd", positions.float(), self.inv_freq)
        return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)

    def precompute(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._cos_sin(positions)

    @staticmethod
    def apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_q, sin_q = self._cos_sin(q_pos)
        cos_k, sin_k = self._cos_sin(k_pos)
        return self.apply(q, cos_q, sin_q), self.apply(k, cos_k, sin_k)


class MultiLevelRoTE(nn.Module):
    """Year-month-day rotary time embedding used by the shared RoTE modules."""

    def __init__(
        self,
        dim: int,
        year_base: float = YEAR_BASE,
        month_base: float = MONTH_BASE,
        day_base: float = DAY_BASE,
        year_weight: float = YEAR_WEIGHT,
        month_weight: float = MONTH_WEIGHT,
        day_weight: float = DAY_WEIGHT,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoTE requires an even head dimension.")
        idx = torch.arange(0, dim, 2).float() / dim
        self.register_buffer("inv_year", 1.0 / (year_base ** idx), persistent=False)
        self.register_buffer("inv_month", 1.0 / (month_base ** idx), persistent=False)
        self.register_buffer("inv_day", 1.0 / (day_base ** idx), persistent=False)
        self.year_weight = year_weight
        self.month_weight = month_weight
        self.day_weight = day_weight

    def _theta(
        self,
        year_ids: torch.Tensor,
        month_ids: torch.Tensor,
        day_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta_year = torch.einsum("bs,d->bsd", year_ids.float(), self.inv_year)
        theta_month = torch.einsum("bs,d->bsd", month_ids.float(), self.inv_month)
        theta_day = torch.einsum("bs,d->bsd", day_ids.float(), self.inv_day)
        return theta_year, theta_month, theta_day

    @staticmethod
    def _cos_sin(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return theta.cos().unsqueeze(1), theta.sin().unsqueeze(1)

    def precompute(
        self,
        ymd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        theta_year, theta_month, theta_day = self._theta(*ymd)
        return (
            self._cos_sin(theta_year),
            self._cos_sin(theta_month),
            self._cos_sin(theta_day),
        )

    def apply(
        self,
        x: torch.Tensor,
        cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> torch.Tensor:
        (cos_year, sin_year), (cos_month, sin_month), (cos_day, sin_day) = cache
        year_state = RotaryTimeEmbedding.apply(x, cos_year, sin_year)
        month_state = RotaryTimeEmbedding.apply(x, cos_month, sin_month)
        day_state = RotaryTimeEmbedding.apply(x, cos_day, sin_day)
        return coarse_to_fine_fusion(
            year_state,
            month_state,
            day_state,
            year_weight=self.year_weight,
            month_weight=self.month_weight,
            day_weight=self.day_weight,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_ymd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        k_ymd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_cache = self.precompute(q_ymd)
        k_cache = q_cache if q_ymd is k_ymd else self.precompute(k_ymd)
        return self.apply(q, q_cache), self.apply(k, k_cache)
