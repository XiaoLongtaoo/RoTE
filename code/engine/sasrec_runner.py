"""Runner for SASRec and RoTE-SASRec under the unified `code/` entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from datasets.sequential_txt import WarpSampler, WarpSamplerWithTime, partition_by_leave_one_out
from engine.common import (
    as_namespace,
    configure_logger,
    dump_config,
    prepare_evaluation_run,
    prepare_run_dirs,
    resolve_repo_path,
    set_random_seed,
)
from evaluation.sasrec import evaluate, evaluate_valid, evaluate_valid_with_time, evaluate_with_time
from models.sasrec.baseline import SASRec
from models.sasrec.rote import RoTESASRec


class SASRecRunner:
    @staticmethod
    def _resolve_data_path(config: dict) -> Path:
        path = Path(config["data_path"])
        if path.is_absolute():
            return path.resolve()

        repo_path = resolve_repo_path(path)
        if repo_path.exists():
            return repo_path

        return (Path(config["data_dir"]) / path).resolve()

    @staticmethod
    def _build_model(config: dict):
        data_path = SASRecRunner._resolve_data_path(config)
        if not data_path.exists():
            raise FileNotFoundError(f"SASRec dataset not found: {data_path}")

        bundle = partition_by_leave_one_out(data_path)
        args = as_namespace(config)
        if config["variant"] == "rote":
            model = RoTESASRec(bundle["usernum"], bundle["itemnum"], args).to(config["device"])
        else:
            model = SASRec(bundle["usernum"], bundle["itemnum"], args).to(config["device"])

        for _, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except Exception:
                pass
        model.item_emb.weight.data[0, :] = 0
        return model, bundle, args

    @staticmethod
    def _save_checkpoint(model: torch.nn.Module, config: dict, epoch: int) -> Path:
        ckpt_path = Path(config["ckpt_dir"]) / f"{config['backbone']}_{config['variant']}_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        return ckpt_path

    @staticmethod
    def _resolve_checkpoint(config: dict) -> Path:
        state_dict_path = config.get("state_dict_path")
        if state_dict_path:
            path = Path(state_dict_path)
            if not path.is_absolute():
                path = Path(config["run_dir"]) / path
            return path.resolve()
        ckpts = sorted(Path(config["ckpt_dir"]).glob("*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found under {config['ckpt_dir']}")
        return ckpts[-1]

    @staticmethod
    def train(config: dict):
        config = config.copy()
        if "run_dir" not in config:
            config = prepare_run_dirs(config)
        logger = configure_logger(config, "rote.sasrec")
        set_random_seed(config["seed"], config["reproducibility"])
        logger.info("Resolved config\n%s", dump_config(config))
        Path(config["run_dir"]).mkdir(parents=True, exist_ok=True)
        with open(Path(config["run_dir"]) / "config.json", "w") as handle:
            json.dump(config, handle, indent=2, default=str)

        model, bundle, args = SASRecRunner._build_model(config)
        logger.info("Loaded dataset with %s users and %s items", bundle["usernum"], bundle["itemnum"])

        sampler = None
        if config["variant"] == "rote":
            sampler = WarpSamplerWithTime(
                bundle["train"],
                bundle["train_time"],
                bundle["usernum"],
                bundle["itemnum"],
                batch_size=config["batch_size"],
                maxlen=config["maxlen"],
                n_workers=config.get("num_workers", 1),
            )
        else:
            sampler = WarpSampler(
                bundle["train"],
                bundle["usernum"],
                bundle["itemnum"],
                batch_size=config["batch_size"],
                maxlen=config["maxlen"],
                n_workers=config.get("num_workers", 1),
            )

        num_batch = (len(bundle["train"]) - 1) // config["batch_size"] + 1
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98))

        best_valid = float("-inf")
        best_epoch = 0

        try:
            for epoch in range(1, int(config["num_epochs"]) + 1):
                model.train()
                epoch_loss = 0.0

                for step in range(num_batch):
                    batch = sampler.next_batch()
                    optimizer.zero_grad()
                    if config["variant"] == "rote":
                        user, seq, time_seq, pos, neg = [np.array(x) for x in batch]
                        pos_logits, neg_logits = model(user, seq, time_seq, pos, neg)
                    else:
                        user, seq, pos, neg = [np.array(x) for x in batch]
                        pos_logits, neg_logits = model(user, seq, pos, neg)

                    pos_labels = torch.ones(pos_logits.shape, device=config["device"])
                    neg_labels = torch.zeros(neg_logits.shape, device=config["device"])
                    indices = np.where(pos != 0)

                    loss = criterion(pos_logits[indices], pos_labels[indices])
                    loss += criterion(neg_logits[indices], neg_labels[indices])
                    for param in model.item_emb.parameters():
                        loss += config["l2_emb"] * torch.sum(param ** 2)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                logger.info("Epoch %s train_loss=%.6f", epoch, epoch_loss / max(1, num_batch))

                if epoch % int(config.get("eval_interval", 20)) != 0:
                    continue

                model.eval()
                with torch.no_grad():
                    if config["variant"] == "rote":
                        valid_metrics = evaluate_valid_with_time(model, bundle, args)
                        test_metrics = evaluate_with_time(model, bundle, args)
                    else:
                        valid_metrics = evaluate_valid(model, bundle, args)
                        test_metrics = evaluate(model, bundle, args)
                logger.info("Epoch %s valid=%s", epoch, valid_metrics)
                logger.info("Epoch %s test=%s", epoch, test_metrics)

                valid_key = config.get("val_metric", "ndcg@10")
                current_valid = float(valid_metrics.get(valid_key, 0.0))
                if current_valid > best_valid:
                    best_valid = current_valid
                    best_epoch = epoch
                    ckpt_path = SASRecRunner._save_checkpoint(model, config, epoch)
                    logger.info("Saved best checkpoint to %s", ckpt_path)

            if best_epoch == 0:
                ckpt_path = SASRecRunner._save_checkpoint(model, config, int(config["num_epochs"]))
                logger.info("Saved final checkpoint to %s", ckpt_path)
        finally:
            if sampler is not None:
                sampler.close()

    @staticmethod
    def evaluate(config: dict):
        config = config.copy()
        if "run_dir" not in config or "state_dict_path" not in config:
            config = prepare_evaluation_run(config)
        logger = configure_logger(config, "rote.sasrec.eval")
        set_random_seed(config["seed"], config["reproducibility"])

        model, bundle, args = SASRecRunner._build_model(config)
        ckpt_path = SASRecRunner._resolve_checkpoint(config)
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(config["device"])))
        model.eval()

        with torch.no_grad():
            if config["variant"] == "rote":
                metrics = evaluate_with_time(model, bundle, args)
            else:
                metrics = evaluate(model, bundle, args)
        logger.info("Loaded checkpoint %s", ckpt_path)
        logger.info("Evaluation metrics: %s", metrics)
        return metrics
