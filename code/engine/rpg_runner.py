"""Runner for RPG and RoTE-RPG under the unified `code/` entrypoint."""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler
from tqdm import tqdm

from datasets.amazon_reviews import AmazonReviews2014Time
from engine.common import (
    build_accelerator,
    config_for_log,
    configure_logger,
    dump_config,
    get_total_steps,
    prepare_evaluation_run,
    prepare_run_dirs,
    set_random_seed,
)
from evaluation.rpg import Evaluator
from models.rpg.tokenizer import RPGTokenizer


class RPGTrainer:
    def __init__(self, config: dict, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = config["accelerator"]
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = configure_logger(config, f"rote.rpg.{config['variant']}")
        self.saved_model_ckpt = Path(config["ckpt_dir"]) / f"{config['backbone']}_{config['variant']}.pth"

    def fit(self, train_dataloader, val_dataloader):
        optimizer = AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        total_steps = get_total_steps(self.config, train_dataloader)
        if total_steps == 0:
            self.logger.info("No training steps resolved; skipping.")
            return None, None

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=total_steps,
        )

        prepared = self.accelerator.prepare(self.model, optimizer, train_dataloader, val_dataloader, scheduler)
        self.model, optimizer, train_dataloader, val_dataloader, scheduler = prepared

        best_epoch = 0
        best_score = float("-inf")
        n_epochs = int(np.ceil(total_steps / max(1, len(train_dataloader))))

        for epoch in range(n_epochs):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training epoch {epoch + 1}")
            for batch in progress:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config["max_grad_norm"] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            self.logger.info("Epoch %s train_loss=%.6f", epoch + 1, total_loss / max(1, len(train_dataloader)))

            if (epoch + 1) % int(self.config["eval_interval"]) != 0:
                continue

            val_results = self.evaluate(val_dataloader, split="val")
            self.logger.info("Epoch %s val=%s", epoch + 1, val_results)
            score = float(val_results[self.config["val_metric"]])
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                torch.save(self.accelerator.unwrap_model(self.model).state_dict(), self.saved_model_ckpt)
                self.logger.info("Saved checkpoint to %s", self.saved_model_ckpt)

            if self.config["patience"] is not None and epoch + 1 - best_epoch >= int(self.config["patience"]):
                self.logger.info("Early stopping at epoch %s", epoch + 1)
                break

        return best_epoch, best_score

    def evaluate(self, dataloader, split="test"):
        self.model.eval()
        all_results = defaultdict(list)
        progress = tqdm(dataloader, total=len(dataloader), desc=f"Eval {split}")
        for batch in progress:
            with torch.no_grad():
                batch = {key: value.to(self.accelerator.device) for key, value in batch.items()}
                preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                results = self.evaluator.calculate_metrics(preds, batch["labels"])
                for key, value in results.items():
                    all_results[key].append(value)

        output = OrderedDict()
        for metric in self.config["metrics"]:
            for k in self.config["topk"]:
                key = f"{metric}@{k}"
                output[key] = torch.cat(all_results[key]).mean().item()
        output["n_visited_items"] = torch.cat(all_results["n_visited_items"]).mean().item()
        return output


class RPGRunner:
    @staticmethod
    def _build_pipeline(config: dict):
        config = config.copy()
        config["accelerator"] = build_accelerator(config)
        config["use_ddp"] = getattr(config["accelerator"], "num_processes", 1) > 1

        dataset = AmazonReviews2014Time(config)
        datasets = dataset.split()
        tokenizer = RPGTokenizer(config, dataset)
        tokenized = tokenizer.tokenize(datasets)

        train_loader = DataLoader(
            tokenized["train"],
            batch_size=config["train_batch_size"],
            shuffle=True,
            num_workers=min(int(config.get("num_workers", 0)), 4),
            collate_fn=tokenizer.collate_fn["train"],
        )
        val_loader = DataLoader(
            tokenized["val"],
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=min(int(config.get("num_workers", 0)), 4),
            collate_fn=tokenizer.collate_fn["val"],
        )
        test_loader = DataLoader(
            tokenized["test"],
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=min(int(config.get("num_workers", 0)), 4),
            collate_fn=tokenizer.collate_fn["test"],
        )

        if config["variant"] == "rote":
            from models.rpg.rote import RoTERPG

            model = RoTERPG(config, dataset, tokenizer)
        else:
            from models.rpg.baseline import RPG

            model = RPG(config, dataset, tokenizer)

        return config, dataset, tokenizer, model, train_loader, val_loader, test_loader

    @staticmethod
    def train(config: dict):
        config = config.copy()
        if "run_dir" not in config:
            config = prepare_run_dirs(config)
        logger = configure_logger(config, "rote.rpg")
        set_random_seed(config["seed"], config["reproducibility"])
        logger.info("Resolved config\n%s", dump_config(config))
        with open(Path(config["run_dir"]) / "config.json", "w") as handle:
            json.dump(config_for_log(config), handle, indent=2, default=str)

        config, dataset, tokenizer, model, train_loader, val_loader, test_loader = RPGRunner._build_pipeline(config)
        logger.info("Dataset stats: users=%s items=%s interactions=%s", dataset.n_users, dataset.n_items, dataset.n_interactions)

        trainer = RPGTrainer(config, model, tokenizer)
        trainer.fit(train_loader, val_loader)
        test_results = trainer.evaluate(test_loader, split="test")
        logger.info("Test results: %s", test_results)
        return test_results

    @staticmethod
    def evaluate(config: dict):
        config = config.copy()
        if "run_dir" not in config or "state_dict_path" not in config:
            config = prepare_evaluation_run(config)
        logger = configure_logger(config, "rote.rpg.eval")
        set_random_seed(config["seed"], config["reproducibility"])
        config, dataset, tokenizer, model, _, _, test_loader = RPGRunner._build_pipeline(config)

        state_dict_path = config.get("state_dict_path")
        ckpt_path = Path(state_dict_path) if state_dict_path else Path(config["ckpt_dir"]) / f"{config['backbone']}_{config['variant']}.pth"
        if not ckpt_path.is_absolute():
            ckpt_path = ckpt_path.resolve()
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(config["device"])))

        trainer = RPGTrainer(config, model, tokenizer)
        results = trainer.evaluate(test_loader, split="test")
        logger.info("Loaded checkpoint %s", ckpt_path)
        logger.info("Evaluation results: %s", results)
        return results
