"""RPG evaluation utilities extracted from the original codebase."""

from __future__ import annotations

import torch

from evaluation.metrics import ndcg_at_k, recall_at_k


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            "recall": recall_at_k,
            "ndcg": ndcg_at_k,
        }
        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config["topk"])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for idx in range(preds.shape[0]):
            cur_label = labels[idx].tolist()
            if self.eos_token in cur_label:
                cur_label = cur_label[:cur_label.index(self.eos_token)]
            for rank in range(self.maxk):
                if preds[idx, rank].tolist() == cur_label:
                    pos_index[idx, rank] = True
                    break
        return pos_index

    def calculate_metrics(self, preds, labels):
        if isinstance(preds, tuple):
            preds, n_visited_items = preds
        else:
            n_visited_items = torch.FloatTensor([len(self.tokenizer.item2tokens)] * preds.shape[0])

        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config["metrics"]:
            for k in self.config["topk"]:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        results["n_visited_items"] = n_visited_items
        return results
