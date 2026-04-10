"""Shared RoTE modules used by all backbones."""

from .fusion import (
    DAY_BASE,
    DAY_WEIGHT,
    MONTH_BASE,
    MONTH_WEIGHT,
    YEAR_BASE,
    YEAR_WEIGHT,
    coarse_to_fine_fusion,
)

try:  # pragma: no cover - allows config-only flows without torch installed
    from .rotary_time_embedding import MultiLevelRoTE, RotaryTimeEmbedding
    from .temporal_decomposition import (
        decompose_unix_timestamp,
        decompose_unix_timestamp_exact,
    )
except Exception:
    MultiLevelRoTE = None
    RotaryTimeEmbedding = None
    decompose_unix_timestamp = None
    decompose_unix_timestamp_exact = None

__all__ = [
    "YEAR_BASE",
    "MONTH_BASE",
    "DAY_BASE",
    "YEAR_WEIGHT",
    "MONTH_WEIGHT",
    "DAY_WEIGHT",
    "coarse_to_fine_fusion",
    "RotaryTimeEmbedding",
    "MultiLevelRoTE",
    "decompose_unix_timestamp",
    "decompose_unix_timestamp_exact",
]
