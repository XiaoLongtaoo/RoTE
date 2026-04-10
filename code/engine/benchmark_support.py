"""Reusable sample-batch builders for public benchmarking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from datasets.sequential_txt import WarpSampler, WarpSamplerWithTime
from engine.common import build_accelerator
from engine.rpg_runner import RPGRunner
from engine.sasrec_runner import SASRecRunner


@dataclass
class ForwardCase:
    model: torch.nn.Module
    forward: Callable[[], Any]
    predict: Callable[[], Any] | None
    cleanup: Callable[[], None]
    batch_size: int


def _prepare_sasrec_case(config: dict, batch_size: int, num_workers: int) -> ForwardCase:
    local_config = config.copy()
    local_config["batch_size"] = batch_size

    model, bundle, _ = SASRecRunner._build_model(local_config)
    sampler_workers = max(1, num_workers) if num_workers is not None else max(1, int(local_config.get("num_workers", 1)))
    if local_config["variant"] == "rote":
        sampler = WarpSamplerWithTime(
            bundle["train"],
            bundle["train_time"],
            bundle["usernum"],
            bundle["itemnum"],
            batch_size=batch_size,
            maxlen=local_config["maxlen"],
            n_workers=sampler_workers,
        )
        user, seq, time_seq, pos, neg = [np.array(x) for x in sampler.next_batch()]

        def forward():
            return model(user, seq, time_seq, pos, neg)

        def predict():
            item_indices = np.arange(1, min(bundle["itemnum"], 5) + 1, dtype=np.int64)
            item_indices = np.tile(item_indices, (batch_size, 1))
            return model.predict(user, seq, time_seq, item_indices)

    else:
        sampler = WarpSampler(
            bundle["train"],
            bundle["usernum"],
            bundle["itemnum"],
            batch_size=batch_size,
            maxlen=local_config["maxlen"],
            n_workers=sampler_workers,
        )
        user, seq, pos, neg = [np.array(x) for x in sampler.next_batch()]

        def forward():
            return model(user, seq, pos, neg)

        def predict():
            item_indices = np.arange(1, min(bundle["itemnum"], 5) + 1, dtype=np.int64)
            item_indices = np.tile(item_indices, (batch_size, 1))
            return model.predict(user, seq, item_indices)

    return ForwardCase(
        model=model,
        forward=forward,
        predict=predict,
        cleanup=sampler.close,
        batch_size=batch_size,
    )


def _prepare_rpg_case(config: dict, batch_size: int, num_workers: int) -> ForwardCase:
    local_config = config.copy()
    local_config["train_batch_size"] = batch_size
    local_config["eval_batch_size"] = batch_size
    local_config["num_workers"] = num_workers
    local_config["accelerator"] = build_accelerator(local_config)

    _, _, _, model, train_loader, _, test_loader = RPGRunner._build_pipeline(local_config)
    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    def forward():
        return model(train_batch)

    def predict():
        return model.generate(test_batch, n_return_sequences=min(2, batch_size))

    return ForwardCase(
        model=model,
        forward=forward,
        predict=predict,
        cleanup=lambda: None,
        batch_size=batch_size,
    )


def build_forward_case(config: dict, batch_size: int = 2, num_workers: int = 0) -> ForwardCase:
    if config["backbone"] == "sasrec":
        return _prepare_sasrec_case(config, batch_size=batch_size, num_workers=num_workers)
    if config["backbone"] == "rpg":
        return _prepare_rpg_case(config, batch_size=batch_size, num_workers=num_workers)
    raise ValueError(f"Unsupported backbone for benchmark helpers: {config['backbone']}")
