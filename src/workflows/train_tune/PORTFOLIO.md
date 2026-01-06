# Train Tune Portfolio – Temporal + BERT Experiments

This folder contains a **series of related Temporal demos** that together form a coherent portfolio for durable ML training, evaluation, and experimentation.

Each subpackage focuses on a specific capability, but they are designed to be **composable** and **incremental**.

---

## Overview of Packages

- `bert_finetune` – Baseline BERT fine‑tuning + inference on Temporal.
  - Who it’s for: teams new to Temporal + Transformers who want a minimal, end‑to‑end training + inference example.
  - What it demonstrates:
    - Simple BERT fine‑tuning driven by a Temporal workflow.
    - Inference as a follow‑up activity / workflow.
  - Start here if you want the smallest possible Temporal + BERT demo.

- `bert_checkpointing` – Durable, checkpoint‑aware training with dataset snapshots.
  - Builds directly on `bert_finetune`.
  - What it adds:
    - Dataset snapshotting and content‑addressed storage.
    - Checkpoint‑aware training (`CheckpointedBertTrainingWorkflow`) with signals + queries.
    - Resumable runs across worker crashes and retries.
  - Use this when you care about **reproducibility** and **resume from checkpoint** behavior.

- `bert_eval` – Coordinated training + evaluation harness.
  - Builds on `bert_checkpointing`.
  - What it adds:
    - `CoordinatorWorkflow` that wires checkpointed training and evaluation together.
    - Canonical `run_id` propagation and consistent `model_path` handling.
    - A single API that turns configs into `BertEvalResult` objects.
  - Use this when you want an **experiment runner** that returns structured metrics.

- `bert_parallel` – Parallel checkpointed experiments.
  - Builds on `bert_eval`.
  - What it adds:
    - Parallel orchestration of multiple training + eval configs.
    - Clear separation of training vs eval/coordinator workers and task queues.
  - Use this when you need to run **many experiments in parallel** across models, datasets, or hyperparameters.

- `bert_sweeps` – Hyperparameter sweeps (random + ladder/TPE).
  - Builds on `bert_eval` / `bert_parallel` patterns.
  - What it adds:
    - `SweepWorkflow` for simple random sweeps.
    - `LadderSweepWorkflow` for staged, TPE‑style search over `SweepSpace`.
    - Durable, replayable sweeps where each trial is a full training + eval pipeline.
  - Use this when you want **adaptive, long‑running hyperparameter search** backed by Temporal.

- `cifar10_scaleup` – Non‑BERT example focusing on scaling patterns.
  - Uses the same Temporal patterns (workflows, activities, workers) applied to a different vision workload.
  - Good reference if you want to port the BERT patterns to other domains.

---

## Suggested Learning / Demo Path

1. **Start with `bert_finetune`**
   - Learn the basic Temporal + Transformers pattern (workflows vs activities, workers, starter script).
2. **Move to `bert_checkpointing`**
   - Add dataset snapshots, checkpoint signals/queries, and resume‑from‑checkpoint semantics.
3. **Add evaluation with `bert_eval`**
   - Use `CoordinatorWorkflow` to tie training and evaluation together.
4. **Scale out with `bert_parallel`**
   - Run multiple checkpointed experiments in parallel across queues and workers.
5. **Explore sweeps with `bert_sweeps`**
   - Implement durable, reproducible hyperparameter sweeps using Temporal workflows.
6. **Generalize to other domains with `cifar10_scaleup`**
   - Apply the same orchestration patterns to non‑NLP workloads.

At each step, you can reuse:

- Pydantic models in `custom_types.py`.
- Activity patterns around dataset loading, tokenization, training, and evaluation.
- Worker topology and task queue conventions.

---

## Cross‑Package Links

- `bert_finetune`
  - README: `src/workflows/train_tune/bert_finetune/docs/README.md`
  - Architecture: `src/workflows/train_tune/bert_finetune/docs/architecture.md`
  - Competitive comparison: `src/workflows/train_tune/bert_finetune/docs/competitive-comparison.md`

- `bert_checkpointing`
  - README: `src/workflows/train_tune/bert_checkpointing/docs/README.md`
  - Architecture: `src/workflows/train_tune/bert_checkpointing/docs/architecture.md`
  - Competitive comparison: `src/workflows/train_tune/bert_checkpointing/docs/competitive-comparison.md`
  - Build guide: `src/workflows/train_tune/bert_checkpointing/docs/CREATE_BERT_CHECKPOINTING.md`

- `bert_eval`
  - README: `src/workflows/train_tune/bert_eval/docs/README.md`
  - Architecture: `src/workflows/train_tune/bert_eval/docs/architecture.md`
  - Competitive comparison: `src/workflows/train_tune/bert_eval/docs/competitive-comparison.md`
  - Build guide from checkpointing: `src/workflows/train_tune/bert_eval/docs/CREATE_BERT_EVAL_FROM_CHECKPOINTING.md`

- `bert_parallel`
  - README: `src/workflows/train_tune/bert_parallel/docs/README.md`
  - Architecture: `src/workflows/train_tune/bert_parallel/docs/architecture.md`
  - Competitive comparison: `src/workflows/train_tune/bert_parallel/docs/competitive-comparison.md`
  - Build guide from eval: `src/workflows/train_tune/bert_parallel/docs/CREATE_BERT_PARALLEL_FROM_EVAL.md`

- `bert_sweeps`
  - README: `src/workflows/train_tune/bert_sweeps/docs/README.md`
  - Architecture: `src/workflows/train_tune/bert_sweeps/docs/architecture.md`
  - Competitive comparison: `src/workflows/train_tune/bert_sweeps/docs/competitive-comparison.md`

- `cifar10_scaleup`
  - See the docs under `src/workflows/train_tune/cifar10_scaleup/docs/` for the vision‑focused variant of these patterns.

---

## Why Temporal for this portfolio

Across all of these packages, Temporal provides:

- **Durable long‑running workflows** for training, evaluation, and sweeps.
- **Deterministic orchestration**: workflows are pure Python; all ML I/O lives in activities.
- **Replayable histories**: you can debug experiments and sweeps by replaying workflows.
- **Scalable architecture**: separate task queues and workers for training vs eval vs orchestration.
- **Portable deployment story**: the same code runs on a laptop, a small cluster, or Temporal Cloud.

Taken together, `train_tune` is a **portfolio‑grade showcase** of how to run serious ML experiments with Temporal while keeping code, reproducibility, and reliability front and center.
