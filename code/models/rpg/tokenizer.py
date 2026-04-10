# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import datetime
import json
import math
import os
from logging import getLogger

import numpy as np
from torch.utils.data.dataloader import default_collate

from datasets.amazon_reviews import AbstractDataset, ExampleDataset


class AbstractTokenizer:
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.config = config
        self.dataset = dataset
        self.logger = getLogger()
        self.eos_token = None
        self.collate_fn = {"train": None, "val": None, "test": None}

    def _init_tokenizer(self):
        raise NotImplementedError

    def tokenize(self, datasets):
        raise NotImplementedError

    @property
    def vocab_size(self):
        raise NotImplementedError

    @property
    def padding_token(self):
        return 0

    @property
    def max_token_seq_len(self):
        raise NotImplementedError

    def log(self, message, level="info"):
        if getattr(self.config["accelerator"], "is_main_process", True):
            getattr(self.logger, level)(message)


class RPGTokenizer(AbstractTokenizer):
    """
    An example when "codebook_size == 256, n_codebook == 32":
        0: padding
        1-256: digit 1
        257-512: digit 2
        ...
        7937-8192: digit 32
        8193: eos
    """

    def __init__(self, config: dict, dataset: AbstractDataset):
        self.n_codebook_bits = self._get_codebook_bits(config["codebook_size"])
        self.index_factory = f'OPQ{config["n_codebook"]},IVF1,PQ{config["n_codebook"]}x{self.n_codebook_bits}'

        super().__init__(config, dataset)
        self.time_seqs = dataset.time_seqs
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping["id2item"]
        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100
        self.collate_fn = {
            "train": self._collate,
            "val": self._collate,
            "test": self._collate,
        }

    def _collate(self, features: list[dict]):
        batch = default_collate(features)
        mode = self.config.get("rote_mode", "unix")
        if mode == "ymd":
            missing = [key for key in ("year_ids", "month_ids", "day_ids") if key not in batch]
            if missing:
                raise ValueError(f"rote_mode=ymd but batch is missing keys {missing}.")
        elif "time_ids" not in batch:
            raise ValueError("rote_mode=unix but batch is missing key 'time_ids'.")
        return batch

    @property
    def n_digit(self):
        return self.config["n_codebook"]

    @property
    def codebook_size(self):
        return self.config["codebook_size"]

    @property
    def max_token_seq_len(self) -> int:
        return self.config["max_item_seq_len"]

    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        if not (x.is_integer() and x >= 0):
            raise AssertionError("Invalid value for n_codebook")
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        assert self.config["metadata"] == "sentence", "RPGTokenizer only supports sentence metadata."

        meta_sentences = []
        for index in range(1, dataset.n_items):
            item = dataset.id_mapping["id2item"][index]
            meta_sentences.append(dataset.item2meta[item])

        if "sentence-transformers" not in self.config["sent_emb_model"]:
            raise NotImplementedError("Only sentence-transformers encoders are currently supported.")

        from sentence_transformers import SentenceTransformer

        sent_emb_model = SentenceTransformer(self.config["sent_emb_model"]).to(self.config["device"])
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config["sent_emb_batch_size"],
            show_progress_bar=True,
            device=self.config["device"],
        )
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        items_for_training = set()
        for row in dataset.split_data["train"]:
            for item in row["item_seq"]:
                items_for_training.add(item)
        self.log(f"[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}")
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        import faiss

        if self.config["opq_use_gpu"]:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config["faiss_omp_num_threads"])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT,
        )
        self.log("[TOKENIZER] Training index...")
        if self.config["opq_use_gpu"]:
            index = faiss.index_cpu_to_gpu(res, self.config["opq_gpu_id"], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config["opq_use_gpu"]:
            index = faiss.index_gpu_to_cpu(index)

        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        list_size = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), list_size * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for _ in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for index in range(pq_codes.shape[0]):
            item = self.id2item[index + 1]
            item2sem_ids[item] = tuple(pq_codes[index].tolist())
        self.log(f"[TOKENIZER] Saving semantic IDs to {sem_ids_path}...")
        with open(sem_ids_path, "w") as handle:
            json.dump(item2sem_ids, handle)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        sem_ids_path = os.path.join(
            dataset.cache_dir,
            "processed",
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids',
        )

        if not os.path.exists(sem_ids_path):
            sent_emb_path = os.path.join(
                dataset.cache_dir,
                "processed",
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb',
            )
            if os.path.exists(sent_emb_path):
                self.log(f"[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...")
                sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config["sent_emb_dim"])
            else:
                self.log("[TOKENIZER] Encoding sentence embeddings...")
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

            if self.config["sent_emb_pca"] > 0:
                self.log("[TOKENIZER] Applying PCA to sentence embeddings...")
                from sklearn.decomposition import PCA

                pca = PCA(n_components=self.config["sent_emb_pca"], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(f"[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}")

            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)

        self.log(f"[TOKENIZER] Loading semantic IDs from {sem_ids_path}...")
        with open(sem_ids_path, "r") as handle:
            item2sem_ids = json.load(handle)
        return self._sem_ids_to_tokens(item2sem_ids)

    def _get_aligned_time_seq(self, user: str, item_seq: list) -> list:
        if self.time_seqs is None or user not in self.time_seqs:
            return [0] * len(item_seq)
        time_seq = [int(ts) for ts in self.time_seqs[user]]
        if len(time_seq) < len(item_seq):
            time_seq = time_seq + [0] * (len(item_seq) - len(time_seq))
        return time_seq[:len(item_seq)]

    def _time_ids_to_ymd_ids(self, time_ids: list) -> tuple[list, list, list]:
        year_ids = []
        month_ids = []
        day_ids = []
        epoch_date = datetime.date(1970, 1, 1)
        for timestamp in time_ids:
            timestamp = int(timestamp)
            if timestamp <= 0:
                year_ids.append(0)
                month_ids.append(0)
                day_ids.append(0)
                continue
            dt = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
            year_ids.append(dt.year - 1970)
            month_ids.append((dt.year - 1970) * 12 + (dt.month - 1))
            day_ids.append((dt.date() - epoch_date).days)
        return year_ids, month_ids, day_ids

    def _tokenize_first_n_items(self, item_seq: list, time_seq: list) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        time_ids = [int(ts) for ts in time_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        time_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)
        return input_ids, time_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, time_seq: list, pad_labels: bool = True) -> tuple:
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        time_ids = [int(ts) for ts in time_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        time_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, time_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str, rote_mode: str = "unix") -> dict:
        max_item_seq_len = self.config["max_item_seq_len"]
        item_seq = example["item_seq"]
        user = example["user"]
        time_seq = self._get_aligned_time_seq(user, item_seq)

        if split == "train":
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
            first_item_seq = item_seq[: min(len(item_seq), max_item_seq_len + 1)]
            first_time_seq = time_seq[: len(first_item_seq)]
            input_ids, time_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=first_item_seq,
                time_seq=first_time_seq,
            )
            all_input_ids = [input_ids]
            all_time_ids = [time_ids]
            all_attention_mask = [attention_mask]
            all_labels = [labels]
            all_seq_lens = [seq_lens]

            for index in range(1, n_return_examples):
                cur_item_seq = item_seq[index : index + max_item_seq_len + 1]
                cur_time_seq = time_seq[index : index + max_item_seq_len + 1]
                input_ids, time_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                    cur_item_seq,
                    cur_time_seq,
                )
                all_input_ids.append(input_ids)
                all_time_ids.append(time_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            output = {
                "input_ids": all_input_ids,
                "time_ids": all_time_ids,
                "attention_mask": all_attention_mask,
                "labels": all_labels,
                "seq_lens": all_seq_lens,
            }
        else:
            eval_item_seq = item_seq[-(max_item_seq_len + 1) :]
            eval_time_seq = time_seq[-len(eval_item_seq) :]
            input_ids, time_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=eval_item_seq,
                time_seq=eval_time_seq,
                pad_labels=False,
            )
            output = {
                "input_ids": [input_ids],
                "time_ids": [time_ids],
                "attention_mask": [attention_mask],
                "labels": [labels[-1:]],
                "seq_lens": [seq_lens],
            }

        if rote_mode == "ymd":
            all_year_ids, all_month_ids, all_day_ids = [], [], []
            for time_ids in output["time_ids"]:
                year_ids, month_ids, day_ids = self._time_ids_to_ymd_ids(time_ids)
                all_year_ids.append(year_ids)
                all_month_ids.append(month_ids)
                all_day_ids.append(day_ids)
            output["year_ids"] = all_year_ids
            output["month_ids"] = all_month_ids
            output["day_ids"] = all_day_ids

        return output

    def tokenize(self, datasets: dict) -> dict:
        tokenized_datasets = {}
        rote_mode = self.config.get("rote_mode", "unix")

        for split, dataset in datasets.items():
            rows = []
            for example in dataset:
                tokenized = self.tokenize_function(example, split=split, rote_mode=rote_mode)
                n_examples = len(tokenized["input_ids"])
                for index in range(n_examples):
                    row = {key: value[index] for key, value in tokenized.items()}
                    rows.append(row)
            tokenized_datasets[split] = ExampleDataset(rows)

        return tokenized_datasets
