"""SASRec evaluation logic extracted from the original codebase."""

from __future__ import annotations

import copy
import random
import sys

import numpy as np
import torch


def _evaluate_split(model, dataset: dict, args, split: str):
    bundle = copy.deepcopy(dataset)
    train = bundle["train"]
    valid = bundle["valid"]
    test = bundle["test"]
    usernum = bundle["usernum"]
    itemnum = bundle["itemnum"]

    ks = tuple(args.topk)
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics.update({f"ndcg@{k}": 0.0 for k in ks})
    metrics.update({f"hr@{k}": 0.0 for k in ks})

    users = random.sample(range(1, usernum + 1), min(usernum, 10000)) if usernum > 10000 else range(1, usernum + 1)
    valid_user = 0.0

    for user in users:
        target_list = test if split == "test" else valid
        if len(train[user]) < 1 or len(target_list[user]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if split == "test" and len(valid[user]) > 0:
            seq[idx] = valid[user][0]
            idx -= 1

        for item in reversed(train[user]):
            seq[idx] = item
            idx -= 1
            if idx == -1:
                break

        target = target_list[user][0]
        rated = set(train[user]) | set(valid[user]) | set(test[user]) | {0}
        rated.discard(target)

        rank = _rank_target_full(model, seq, target, rated, itemnum, args)
        valid_user += 1
        for k in ks:
            if rank < k:
                gain = 1 / np.log2(rank + 2)
                metrics[f"recall@{k}"] += 1
                metrics[f"hr@{k}"] += 1
                metrics[f"ndcg@{k}"] += gain
        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    for k in ks:
        metrics[f"recall@{k}"] /= valid_user
        metrics[f"hr@{k}"] /= valid_user
        metrics[f"ndcg@{k}"] /= valid_user
    return metrics


def _evaluate_split_with_time(model, dataset: dict, args, split: str):
    bundle = copy.deepcopy(dataset)
    train = bundle["train"]
    valid = bundle["valid"]
    test = bundle["test"]
    train_time = bundle["train_time"]
    valid_time = bundle["valid_time"]
    test_time = bundle["test_time"]
    usernum = bundle["usernum"]
    itemnum = bundle["itemnum"]

    ks = tuple(args.topk)
    metrics = {f"recall@{k}": 0.0 for k in ks}
    metrics.update({f"ndcg@{k}": 0.0 for k in ks})
    metrics.update({f"hr@{k}": 0.0 for k in ks})

    users = random.sample(range(1, usernum + 1), min(usernum, 10000)) if usernum > 10000 else range(1, usernum + 1)
    valid_user = 0.0

    for user in users:
        target_list = test if split == "test" else valid
        if len(train[user]) < 1 or len(target_list[user]) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int64)
        idx = args.maxlen - 1

        if split == "test" and len(valid[user]) > 0:
            seq[idx] = valid[user][0]
            time_seq[idx] = valid_time[user][0] if len(valid_time[user]) > 0 else 0
            idx -= 1

        for item, timestamp in zip(reversed(train[user]), reversed(train_time[user])):
            seq[idx] = item
            time_seq[idx] = timestamp
            idx -= 1
            if idx == -1:
                break

        target = target_list[user][0]
        rated = set(train[user]) | set(valid[user]) | set(test[user]) | {0}
        rated.discard(target)

        rank = _rank_target_full_with_time(model, seq, time_seq, target, rated, itemnum, args)
        valid_user += 1
        for k in ks:
            if rank < k:
                gain = 1 / np.log2(rank + 2)
                metrics[f"recall@{k}"] += 1
                metrics[f"hr@{k}"] += 1
                metrics[f"ndcg@{k}"] += gain
        if valid_user % 100 == 0:
            print(".", end="")
            sys.stdout.flush()

    for k in ks:
        metrics[f"recall@{k}"] /= valid_user
        metrics[f"hr@{k}"] /= valid_user
        metrics[f"ndcg@{k}"] /= valid_user
    return metrics


def evaluate(model, dataset: dict, args):
    return _evaluate_split(model, dataset, args, "test")


def evaluate_valid(model, dataset: dict, args):
    return _evaluate_split(model, dataset, args, "valid")


def evaluate_with_time(model, dataset: dict, args):
    return _evaluate_split_with_time(model, dataset, args, "test")


def evaluate_valid_with_time(model, dataset: dict, args):
    return _evaluate_split_with_time(model, dataset, args, "valid")


def _rank_target_full(model, seq, target, rated, itemnum, args):
    step = getattr(args, "eval_item_batch", 4096)
    with torch.no_grad():
        final_feat = model.log2feats(np.array([seq]))[:, -1, :].squeeze(0)
        scores = torch.full((itemnum + 1,), float("-inf"), device=args.device)
        for start in range(1, itemnum + 1, step):
            end = min(itemnum + 1, start + step)
            item_ids = torch.arange(start, end, device=args.device)
            logits = model.item_emb(item_ids).matmul(final_feat)
            scores[start:end] = logits
        if rated:
            rated_tensor = torch.tensor(list(rated), device=args.device, dtype=torch.long)
            scores[rated_tensor] = float("-inf")
        target_score = scores[target]
        return (scores > target_score).sum().item()


def _rank_target_full_with_time(model, seq, time_seq, target, rated, itemnum, args):
    step = getattr(args, "eval_item_batch", 4096)
    with torch.no_grad():
        final_feat = model.log2feats(np.array([seq]), np.array([time_seq]))[:, -1, :].squeeze(0)
        scores = torch.full((itemnum + 1,), float("-inf"), device=args.device)
        for start in range(1, itemnum + 1, step):
            end = min(itemnum + 1, start + step)
            item_ids = torch.arange(start, end, device=args.device)
            logits = model.item_emb(item_ids).matmul(final_feat)
            scores[start:end] = logits
        if rated:
            rated_tensor = torch.tensor(list(rated), device=args.device, dtype=torch.long)
            scores[rated_tensor] = float("-inf")
        target_score = scores[target]
        return (scores > target_score).sum().item()
