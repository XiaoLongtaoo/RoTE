"""Shared config, logging, and utility helpers for the RoTE codebase."""

from __future__ import annotations

import datetime
import hashlib
import html
import logging
import os
import random
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import requests
import yaml

from modules.fusion import (
    DAY_BASE,
    DAY_WEIGHT,
    MONTH_BASE,
    MONTH_WEIGHT,
    YEAR_BASE,
    YEAR_WEIGHT,
)


ROOT_DIR = Path(__file__).resolve().parents[2]

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency in research environments
    tiktoken = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for config-only flows
    torch = None


class SimpleAccelerator:
    """A small single-process fallback for environments without `accelerate`."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device) if torch is not None else device
        self.is_main_process = True
        self.num_processes = 1

    def prepare(self, *objects):
        if len(objects) == 1:
            return objects[0]
        return objects

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, model):
        return model

    def gather_for_metrics(self, values):
        return values

    @contextmanager
    def main_process_first(self):
        yield

    def init_trackers(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None

    def end_training(self):
        return None


def parse_command_line_args(unparsed: list[str]) -> dict:
    args = {}
    for text_arg in unparsed:
        if "=" not in text_arg:
            raise ValueError(f"Invalid command line argument: {text_arg}. Expected --key=value.")
        key, value = text_arg.split("=", 1)
        key = key[2:] if key.startswith("--") else key
        try:
            value = eval(value)
        except Exception:
            pass
        args[key] = value
    return args


def convert_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(config.items()):
        if not isinstance(value, str):
            continue
        try:
            new_value = eval(value)
            if new_value is not None and not isinstance(new_value, (str, int, float, bool, list, dict, tuple)):
                new_value = value
        except Exception:
            lowered = value.lower()
            if lowered in {"true", "false"}:
                new_value = lowered == "true"
            else:
                new_value = value
        config[key] = new_value
    return config


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text()) or {}
    if overrides:
        config.update(overrides)
    config = convert_config_dict(config)

    config.setdefault("seed", 2024)
    config.setdefault("reproducibility", True)
    config.setdefault("topk", [5, 10])
    config.setdefault("metrics", ["ndcg", "recall"])
    config.setdefault("val_metric", "ndcg@10")
    config.setdefault("eval_item_batch", 4096)
    config.setdefault("year_base", YEAR_BASE)
    config.setdefault("month_base", MONTH_BASE)
    config.setdefault("day_base", DAY_BASE)
    config.setdefault("year_weight", YEAR_WEIGHT)
    config.setdefault("month_weight", MONTH_WEIGHT)
    config.setdefault("day_weight", DAY_WEIGHT)
    config.setdefault("data_dir", str(ROOT_DIR / "data"))
    config.setdefault("cache_dir", str(ROOT_DIR / "cache"))
    config.setdefault("runs_dir", str(ROOT_DIR / "runs"))
    config.setdefault("device", "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
    config["config_path"] = str(config_path)
    return config


def make_run_name(config: dict[str, Any]) -> str:
    payload = f"{config.get('backbone','na')}-{config.get('variant','na')}-{config.get('dataset_name', config.get('category','na'))}"
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()[:6]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{payload}-{timestamp}-{digest}"


def get_dataset_id(config: dict[str, Any]) -> str:
    return str(config.get("dataset_name") or config.get("category") or "default")


def get_run_family_dir(config: dict[str, Any]) -> Path:
    return Path(config["runs_dir"]) / config["backbone"] / config["variant"] / get_dataset_id(config)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def _set_run_artifact_paths(
    config: dict[str, Any],
    run_dir: Path,
    *,
    log_filename: str,
    create_dirs: bool,
) -> dict[str, Any]:
    if create_dirs:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tensorboard").mkdir(exist_ok=True)
        (run_dir / "ckpt").mkdir(exist_ok=True)
    config["run_dir"] = str(run_dir)
    config["tensorboard_dir"] = str(run_dir / "tensorboard")
    config["ckpt_dir"] = str(run_dir / "ckpt")
    config["log_path"] = str(run_dir / log_filename)
    return config


def prepare_run_dirs(config: dict[str, Any], create_dirs: bool = True) -> dict[str, Any]:
    run_name = config.get("run_name") or make_run_name(config)
    config["run_name"] = run_name
    run_dir = get_run_family_dir(config) / run_name
    return _set_run_artifact_paths(config, run_dir, log_filename="train.log", create_dirs=create_dirs)


def resolve_checkpoint_path(config: dict[str, Any]) -> Path:
    state_dict_path = config.get("state_dict_path")
    if state_dict_path:
        path = resolve_repo_path(state_dict_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    run_root = get_run_family_dir(config)
    run_name = config.get("run_name")
    pattern = f"{run_name}/ckpt/*.pth" if run_name else "*/ckpt/*.pth"
    candidates = [path.resolve() for path in run_root.glob(pattern) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Train a model first or pass --state-dict-path to evaluate a specific checkpoint."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def prepare_evaluation_run(config: dict[str, Any], create_dirs: bool = True) -> dict[str, Any]:
    ckpt_path = resolve_checkpoint_path(config)
    source_ckpt_dir = ckpt_path.parent.resolve()
    source_run_dir = source_ckpt_dir.parent.resolve() if source_ckpt_dir.name == "ckpt" else source_ckpt_dir
    eval_name = config.get("eval_name") or f"evaluate-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    config["state_dict_path"] = str(ckpt_path)
    config["source_ckpt_dir"] = str(source_ckpt_dir)
    config["source_run_dir"] = str(source_run_dir)
    config["resolved_run_name"] = source_run_dir.name
    config["eval_name"] = eval_name

    run_dir = source_run_dir / "eval" / eval_name
    return _set_run_artifact_paths(config, run_dir, log_filename="evaluate.log", create_dirs=create_dirs)


def configure_logger(config: dict[str, Any], name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if "log_path" in config:
        file_handler = logging.FileHandler(config["log_path"])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def log_message(logger: logging.Logger, message: str, level: str = "info") -> None:
    getattr(logger, level)(message)


def set_random_seed(seed: int, reproducibility: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def as_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def build_accelerator(config: dict[str, Any]):
    try:
        from accelerate import Accelerator

        return Accelerator()
    except Exception:
        return SimpleAccelerator(config.get("device", "cpu"))


def dump_config(config: dict[str, Any]) -> str:
    lines = [f"{key}: {config[key]}" for key in sorted(config.keys()) if key != "accelerator"]
    return "\n".join(lines)


def get_total_steps(config: dict[str, Any], train_dataloader) -> int:
    if config.get("steps") is not None:
        return int(config["steps"])
    return len(train_dataloader) * int(config["epochs"])


def config_for_log(config: dict[str, Any]) -> dict[str, Any]:
    copied = config.copy()
    copied.pop("accelerator", None)
    for key, value in list(copied.items()):
        if isinstance(value, list):
            copied[key] = str(value)
    return copied


def download_file(url: str, path: str) -> None:
    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as handle:
        handle.write(response.content)


def list_to_str(value, remove_blank: bool = False) -> str:
    result = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
    return result.replace(" ", "") if remove_blank else result


def clean_text(raw_text: str) -> str:
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[\n\t]", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    return text


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    if tiktoken is None:
        return len(string.split())
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))
