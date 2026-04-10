"""Sequential TXT dataset utilities extracted from the SASRec codebase."""

from __future__ import annotations

import copy
from collections import defaultdict
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np


def read_interactions(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sequential dataset not found: {path}")
    data = np.loadtxt(path, dtype=np.int64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def build_index(path: str | Path):
    ui_mat = read_interactions(path).astype(np.int64)
    n_users = int(ui_mat[:, 0].max())
    n_items = int(ui_mat[:, 1].max())
    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]
    if ui_mat.shape[1] == 2:
        for u, i in ui_mat[:, :2]:
            u2i_index[int(u)].append(int(i))
            i2u_index[int(i)].append(int(u))
        return u2i_index, i2u_index
    u2t_index = [[] for _ in range(n_users + 1)]
    for u, i, t in ui_mat[:, :3]:
        u = int(u)
        i = int(i)
        t = int(t)
        u2i_index[u].append(i)
        i2u_index[i].append(u)
        u2t_index[u].append(t)
    return u2i_index, i2u_index, u2t_index


def partition_by_leave_one_out(path: str | Path) -> dict:
    rows = read_interactions(path)
    has_time = rows.shape[1] >= 3

    usernum = 0
    itemnum = 0
    user_items = defaultdict(list)
    user_times = defaultdict(list)

    for row in rows:
        u = int(row[0])
        i = int(row[1])
        usernum = max(usernum, u)
        itemnum = max(itemnum, i)
        user_items[u].append(i)
        if has_time:
            user_times[u].append(int(row[2]))

    user_train = {}
    user_valid = {}
    user_test = {}
    user_train_time = {}
    user_valid_time = {}
    user_test_time = {}

    for user, items in user_items.items():
        nfeedback = len(items)
        times = user_times.get(user, [])
        if nfeedback < 4:
            user_train[user] = items
            user_valid[user] = []
            user_test[user] = []
            user_train_time[user] = times
            user_valid_time[user] = []
            user_test_time[user] = []
        else:
            user_train[user] = items[:-2]
            user_valid[user] = [items[-2]]
            user_test[user] = [items[-1]]
            if has_time:
                user_train_time[user] = times[:-2]
                user_valid_time[user] = [times[-2]]
                user_test_time[user] = [times[-1]]
            else:
                user_train_time[user] = []
                user_valid_time[user] = []
                user_test_time[user] = []

    return {
        "has_time": has_time,
        "train": user_train,
        "valid": user_valid,
        "test": user_test,
        "usernum": usernum,
        "itemnum": itemnum,
        "train_time": user_train_time,
        "valid_time": user_valid_time,
        "test_time": user_test_time,
    }


def random_neq(left: int, right: int, seen: set[int]) -> int:
    value = np.random.randint(left, right)
    while value in seen:
        value = np.random.randint(left, right)
    return int(value)


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, seed):
    def sample(uid: int):
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for item in reversed(user_train[uid][:-1]):
            seq[idx] = item
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = item
            idx -= 1
            if idx == -1:
                break
        return uid, seq, pos, neg

    np.random.seed(seed)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample(int(uids[counter % usernum])))
            counter += 1
        result_queue.put(zip(*one_batch))


def sample_function_with_time(user_train, user_time, usernum, itemnum, batch_size, maxlen, result_queue, seed):
    def sample(uid: int):
        while len(user_train[uid]) <= 1:
            uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int64)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        seen = set(user_train[uid])
        items = user_train[uid][:-1]
        times = user_time[uid][:-1]
        for item, timestamp in zip(reversed(items), reversed(times)):
            seq[idx] = item
            time_seq[idx] = timestamp
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, seen)
            nxt = item
            idx -= 1
            if idx == -1:
                break
        return uid, seq, time_seq, pos, neg

    np.random.seed(seed)
    uids = np.arange(1, usernum + 1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample(int(uids[counter % usernum])))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler:
    def __init__(self, user_train, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            proc = Process(
                target=sample_function,
                args=(
                    user_train,
                    usernum,
                    itemnum,
                    batch_size,
                    maxlen,
                    self.result_queue,
                    np.random.randint(2e9),
                ),
            )
            proc.daemon = True
            proc.start()
            self.processors.append(proc)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for proc in self.processors:
            proc.terminate()
            proc.join()


class WarpSamplerWithTime:
    def __init__(self, user_train, user_time, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            proc = Process(
                target=sample_function_with_time,
                args=(
                    user_train,
                    user_time,
                    usernum,
                    itemnum,
                    batch_size,
                    maxlen,
                    self.result_queue,
                    np.random.randint(2e9),
                ),
            )
            proc.daemon = True
            proc.start()
            self.processors.append(proc)

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for proc in self.processors:
            proc.terminate()
            proc.join()


def clone_dataset_bundle(bundle: dict) -> dict:
    return copy.deepcopy(bundle)
