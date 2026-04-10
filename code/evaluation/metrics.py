"""Shared ranking metrics used by the RoTE codebase."""

from __future__ import annotations

import torch


def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    return pos_index[:, :k].sum(dim=1).float()


def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    ranks = torch.arange(1, pos_index.shape[-1] + 1, device=pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    return torch.where(pos_index, dcg, 0)[:, :k].sum(dim=1).float()

