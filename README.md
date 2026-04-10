# RoTE

Official SIGIR 2026 implementation of **RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation**.

RoTE is a shared temporal modeling module for Transformer-based sequential recommendation. This repository exposes four public variants:

- `SASRec`
- `RoTE-SASRec`
- `RPG`
- `RoTE-RPG`

## Method Overview

RoTE decomposes each interaction timestamp into coarse-to-fine calendar levels and injects those signals into self-attention through rotary transformations. The unified codebase keeps one shared implementation and plugs it into both the SASRec and RPG backbones.

## Repository Layout

This repository keeps only the public code path:

```text
RoTE/
  code/
  README.md
  requirements.txt
```

Runtime directories such as `data/`, `cache/`, and `runs/` are created locally when needed and are not tracked in the repository.

## Installation

Install the public dependencies from the project root:

```bash
pip install -r requirements.txt
```

If you use GPU FAISS locally, replace `faiss-cpu` with the FAISS build that matches your environment.

## Data Preparation

Place processed experiment data under `data/` at runtime. For Amazon 5-core reviews, you can generate sequential text files with:

```bash
python code/scripts/preprocess_amazon_5core.py --dataset Toys_and_Games_5
```

This repository does not ship the datasets. Training, evaluation, and benchmarking all assume that you have already prepared local data under `data/`.

## Training

Use the unified training entrypoint with one of the four public configs:

```bash
python code/train.py --config code/configs/sasrec_baseline.yaml
python code/train.py --config code/configs/rote_sasrec.yaml
python code/train.py --config code/configs/rpg_baseline.yaml
python code/train.py --config code/configs/rote_rpg.yaml
```

## Evaluation

Evaluate with the unified entrypoint:

```bash
python code/evaluate.py --config code/configs/rote_sasrec.yaml
python code/evaluate.py --config code/configs/rote_rpg.yaml
```

By default, `evaluate.py` resolves the latest checkpoint under the matching `runs/<backbone>/<variant>/<dataset>/` directory. You can also target a specific run or checkpoint:

```bash
python code/evaluate.py --config code/configs/rote_sasrec.yaml --run-name <run_name>
python code/evaluate.py --config code/configs/rote_rpg.yaml --state-dict-path runs/rpg/rote/<dataset>/<run>/ckpt/<file>.pth
```

Evaluation requires an existing trained checkpoint under `runs/` or an explicit `--state-dict-path`.

## Benchmark

Benchmark a representative forward pass with:

```bash
python code/scripts/benchmark_efficiency.py --config code/configs/rote_sasrec.yaml --metric all --batch-size 8 --steps 10
python code/scripts/benchmark_efficiency.py --config code/configs/rote_rpg.yaml --metric latency --batch-size 4 --steps 10
```

The benchmark reports parameter count, mean latency, and profiler-derived FLOPs when available. It requires a local PyTorch environment and locally available data.

## Notes

`data/`, `cache/`, `runs/`, real datasets, and checkpoints are runtime assets and should not be committed.
