"""Shared RoTE constants and fusion utilities."""

from __future__ import annotations


YEAR_BASE = 1_000_000.0
MONTH_BASE = 10_000.0
DAY_BASE = 100.0

YEAR_WEIGHT = 1.5
MONTH_WEIGHT = 1.0
DAY_WEIGHT = 0.5


def coarse_to_fine_fusion(
    year_state,
    month_state,
    day_state,
    year_weight: float = YEAR_WEIGHT,
    month_weight: float = MONTH_WEIGHT,
    day_weight: float = DAY_WEIGHT,
):
    """Fuse year/month/day rotary states using the repository defaults."""

    return year_weight * year_state + month_weight * month_state + day_weight * day_state
