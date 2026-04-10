"""Shared temporal decomposition helpers for RoTE."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch


SECONDS_PER_DAY = 86_400.0
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = 30.4375


def decompose_unix_timestamp(
    timestamps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate coarse-to-fine time decomposition used during training."""

    days = timestamps.float() / SECONDS_PER_DAY
    years = days / DAYS_PER_YEAR
    months = days / DAYS_PER_MONTH
    return years, months, days


def decompose_unix_timestamp_exact(
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact calendar decomposition used for inspection and validation."""

    epoch = datetime(1970, 1, 1)
    shape = timestamps.shape
    flat_ts = timestamps.flatten()

    years = np.zeros_like(flat_ts, dtype=np.float32)
    months = np.zeros_like(flat_ts, dtype=np.float32)
    days = np.zeros_like(flat_ts, dtype=np.float32)

    for idx, ts in enumerate(flat_ts):
        if ts == 0:
            continue
        try:
            dt = datetime.fromtimestamp(ts)
        except (OSError, OverflowError, ValueError):
            continue
        years[idx] = dt.year - 1970
        months[idx] = (dt.year - 1970) * 12 + (dt.month - 1)
        days[idx] = (dt - epoch).days

    return years.reshape(shape), months.reshape(shape), days.reshape(shape)
