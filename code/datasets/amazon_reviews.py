# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from logging import getLogger
from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from engine.common import clean_text, download_file


class ExampleDataset(Dataset):
    """A tiny map-style dataset used to keep the public codepath dependency-light."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config["accelerator"]
        self.logger = getLogger()

        self.all_item_seqs = {}
        self.id_mapping = {
            "user2id": {"[PAD]": 0},
            "item2id": {"[PAD]": 0},
            "id2user": ["[PAD]"],
            "id2item": ["[PAD]"],
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return (
            f"[Dataset] {self.__class__.__name__}\n"
            f"\tNumber of users: {self.n_users}\n"
            f"\tNumber of items: {self.n_items}\n"
            f"\tNumber of interactions: {self.n_interactions}\n"
            f"\tAverage item sequence length: {self.avg_item_seq_len}"
        )

    @property
    def n_users(self):
        return len(self.user2id)

    @property
    def n_items(self):
        return len(self.item2id)

    @property
    def n_interactions(self):
        return sum(len(seq) for seq in self.all_item_seqs.values())

    @property
    def avg_item_seq_len(self):
        return self.n_interactions / max(1, self.n_users)

    @property
    def user2id(self):
        return self.id_mapping["user2id"]

    @property
    def item2id(self):
        return self.id_mapping["item2id"]

    def _download_and_process_raw(self):
        raise NotImplementedError

    def _leave_one_out(self):
        datasets = {
            "train": [],
            "val": [],
            "test": [],
        }
        for user, item_seq in self.all_item_seqs.items():
            datasets["test"].append({"user": user, "item_seq": item_seq})
            if len(item_seq) > 1:
                datasets["val"].append({"user": user, "item_seq": item_seq[:-1]})
            if len(item_seq) > 2:
                datasets["train"].append({"user": user, "item_seq": item_seq[:-2]})
        return {split: ExampleDataset(rows) for split, rows in datasets.items()}

    def split(self):
        if self.split_data is not None:
            return self.split_data

        split_strategy = self.config["split"]
        if split_strategy in {"leave_one_out", "last_out"}:
            self.split_data = self._leave_one_out()
            return self.split_data
        raise NotImplementedError(f"Split strategy [{split_strategy}] not implemented.")

    def log(self, message, level="info"):
        if getattr(self.accelerator, "is_main_process", True):
            getattr(self.logger, level)(message)


class AmazonReviews2014Time(AbstractDataset):
    """
    A class representing the Amazon Reviews 2014 dataset.

    Args:
        config (dict): A dictionary containing the configuration parameters for the dataset.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.category = config["category"]
        self._check_available_category()
        self.log(f"[DATASET] Amazon Reviews 2014 for category: {self.category}")

        self.cache_dir = os.path.join(config["cache_dir"], "AmazonReviews2014Time", self.category)
        self._download_and_process_raw()

    def _check_available_category(self):
        available_categories = [
            "Books",
            "Electronics",
            "Movies_and_TV",
            "CDs_and_Vinyl",
            "Clothing_Shoes_and_Jewelry",
            "Home_and_Kitchen",
            "Kindle_Store",
            "Sports_and_Outdoors",
            "Cell_Phones_and_Accessories",
            "Health_and_Personal_Care",
            "Toys_and_Games",
            "Video_Games",
            "Tools_and_Home_Improvement",
            "Beauty",
            "Apps_for_Android",
            "Office_Products",
            "Pet_Supplies",
            "Automotive",
            "Grocery_and_Gourmet_Food",
            "Patio_Lawn_and_Garden",
            "Baby",
            "Digital_Music",
            "Musical_Instruments",
            "Amazon_Instant_Video",
        ]
        if self.category not in available_categories:
            raise AssertionError(
                f'Category "{self.category}" not available. Available categories: {available_categories}'
            )

    def _download_raw(self, path: str, type: str = "reviews") -> str:
        url = (
            "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
            f'{type}_{self.category}{"_5" if type == "reviews" else ""}.json.gz'
        )
        base_name = os.path.basename(url)
        local_filepath = os.path.join(path, base_name)
        if not os.path.exists(local_filepath):
            download_file(url, local_filepath)
        return local_filepath

    def _parse_gz(self, path: str):
        with gzip.open(path, "r") as handle:
            for line in handle:
                line = line.replace(b"true", b"True").replace(b"false", b"False")
                yield eval(line)

    def _load_reviews(self, path: str) -> list:
        self.log("[DATASET] Loading reviews...")
        reviews = []
        for inter in self._parse_gz(path):
            reviews.append((inter["reviewerID"], inter["asin"], int(inter["unixReviewTime"])))
        return reviews

    def _get_item_seqs(self, reviews: list[tuple]) -> dict:
        item_seqs = defaultdict(list)
        time_seqs = defaultdict(list)
        for user, item, timestamp in reviews:
            item_seqs[user].append((item, timestamp))

        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda pair: pair[1])
            item_seqs[user] = [pair[0] for pair in item_time]
            time_seqs[user] = [pair[1] for pair in item_time]
        return item_seqs, time_seqs

    def _remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:
        self.log("[DATASET] Remapping user and item IDs...")
        for user, items in item_seqs.items():
            if user not in self.id_mapping["user2id"]:
                self.id_mapping["user2id"][user] = len(self.id_mapping["id2user"])
                self.id_mapping["id2user"].append(user)
            remapped = []
            for item in items:
                if item not in self.id_mapping["item2id"]:
                    self.id_mapping["item2id"][item] = len(self.id_mapping["id2item"])
                    self.id_mapping["id2item"].append(item)
                remapped.append(item)
            self.all_item_seqs[user] = remapped
        return self.all_item_seqs, self.id_mapping

    def _process_reviews(self, input_path: str, output_path: str) -> tuple[dict, dict, dict]:
        seq_file = os.path.join(output_path, "all_item_seqs.json")
        id_mapping_file = os.path.join(output_path, "id_mapping.json")
        time_seqs_file = os.path.join(output_path, "time_seqs.json")
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file) and os.path.exists(time_seqs_file):
            self.log("[DATASET] Reviews have been processed...")
            with open(seq_file, "r") as handle:
                all_item_seqs = json.load(handle)
            with open(id_mapping_file, "r") as handle:
                id_mapping = json.load(handle)
            with open(time_seqs_file, "r") as handle:
                time_seqs = json.load(handle)
            return all_item_seqs, id_mapping, time_seqs

        self.log("[DATASET] Processing reviews...")
        reviews = self._load_reviews(input_path)
        item_seqs, time_seqs = self._get_item_seqs(reviews)
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)

        self.log("[DATASET] Saving mapping data...")
        with open(seq_file, "w") as handle:
            json.dump(all_item_seqs, handle)
        with open(id_mapping_file, "w") as handle:
            json.dump(id_mapping, handle)
        with open(time_seqs_file, "w") as handle:
            json.dump(time_seqs, handle)
        return all_item_seqs, id_mapping, time_seqs

    def _load_metadata(self, path: str, item2id: dict) -> dict:
        self.log("[DATASET] Loading metadata...")
        data = {}
        item_asins = set(item2id.keys())
        for info in tqdm(self._parse_gz(path)):
            if info["asin"] not in item_asins:
                continue
            data[info["asin"]] = info
        return data

    def _sent_process(self, raw) -> str:
        sentence = ""
        if isinstance(raw, float):
            sentence += f"{raw}."
        elif len(raw) > 0 and isinstance(raw[0], list):
            for values in raw:
                for value in values:
                    sentence += clean_text(value)[:-1]
                    sentence += ", "
            sentence = sentence[:-2] + "."
        elif isinstance(raw, list):
            for value in raw:
                sentence += clean_text(value)
        else:
            sentence = clean_text(raw)
        return sentence + " "

    def _extract_meta_sentences(self, metadata: dict) -> dict:
        self.log("[DATASET] Extracting meta sentences...")
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            meta_sentence = ""
            keys = set(meta.keys())
            for feature in ["title", "price", "brand", "feature", "categories", "description"]:
                if feature in keys:
                    meta_sentence += self._sent_process(meta[feature])
            item2meta[item] = meta_sentence
        return item2meta

    def _process_meta(self, input_path: str, output_path: str) -> Optional[dict]:
        process_mode = self.config["metadata"]
        meta_file = os.path.join(output_path, f"metadata.{process_mode}.json")
        if os.path.exists(meta_file):
            self.log("[DATASET] Metadata has been processed...")
            with open(meta_file, "r") as handle:
                return json.load(handle)

        self.log(f"[DATASET] Processing metadata, mode: {process_mode}")
        if process_mode == "none":
            return None

        item2meta = self._load_metadata(path=input_path, item2id=self.item2id)
        if process_mode == "raw":
            pass
        elif process_mode == "sentence":
            item2meta = self._extract_meta_sentences(metadata=item2meta)
        else:
            raise NotImplementedError("Metadata processing type not implemented.")

        with open(meta_file, "w") as handle:
            json.dump(item2meta, handle)
        return item2meta

    def _download_and_process_raw(self):
        processed_data_path = os.path.join(self.cache_dir, "processed")
        os.makedirs(processed_data_path, exist_ok=True)

        seq_file = os.path.join(processed_data_path, "all_item_seqs.json")
        id_mapping_file = os.path.join(processed_data_path, "id_mapping.json")
        time_seqs_file = os.path.join(processed_data_path, "time_seqs.json")
        meta_file = os.path.join(processed_data_path, f'metadata.{self.config["metadata"]}.json')
        if (
            os.path.exists(seq_file)
            and os.path.exists(id_mapping_file)
            and os.path.exists(time_seqs_file)
            and (self.config["metadata"] == "none" or os.path.exists(meta_file))
        ):
            self.log("[DATASET] Found processed cache, skipping raw download.")
            with open(seq_file, "r") as handle:
                self.all_item_seqs = json.load(handle)
            with open(id_mapping_file, "r") as handle:
                self.id_mapping = json.load(handle)
            with open(time_seqs_file, "r") as handle:
                self.time_seqs = json.load(handle)
            if self.config["metadata"] == "none":
                self.item2meta = None
            else:
                with open(meta_file, "r") as handle:
                    self.item2meta = json.load(handle)
            return

        raw_data_path = os.path.join(self.cache_dir, "raw")
        os.makedirs(raw_data_path, exist_ok=True)
        with self.accelerator.main_process_first():
            reviews_localpath = self._download_raw(path=raw_data_path, type="reviews")
            meta_localpath = self._download_raw(path=raw_data_path, type="meta")

        np.random.seed(12345)

        self.all_item_seqs, self.id_mapping, self.time_seqs = self._process_reviews(
            input_path=reviews_localpath,
            output_path=processed_data_path,
        )
        self.item2meta = self._process_meta(input_path=meta_localpath, output_path=processed_data_path)
