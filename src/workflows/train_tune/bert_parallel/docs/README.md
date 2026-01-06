# Parallel BERT Training + Evaluation (Temporal)

This folder contains a demo that uses Temporal to **coordinate checkpointed BERT training and evaluation across multiple configurations in parallel**.

It builds on the patterns from:

- `bert_finetune` – baseline BERT fine‑tuning + inference.
- `bert_checkpointing` – dataset snapshots + checkpoint‑aware training.

and adds:

- A coordinator workflow that fans out multiple checkpointed training runs.
- A matching evaluation workflow per trained run.

## What this package demonstrates

- **Parallel orchestration** of multiple BERT fine‑tune + eval runs using Temporal child workflows.
- **Checkpoint‑aware training** with dataset snapshots and mid‑run checkpoint signals.
- **Flexible schema inference** for text/label fields and task type.
- Separation of:
  - Training/snapshot activities on `bert-training-task-queue`.
  - Evaluation/coordinator workflows on `bert-eval-task-queue`.

## Quickstart

From the project root (`temporal_training/`):

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start the training worker** (GPU recommended):

   ```bash
   uv run -m src.workflows.train_tune.bert_parallel.training_worker
   ```

3. **Start the evaluation / coordinator worker** (CPU is fine):

   ```bash
   uv run -m src.workflows.train_tune.bert_parallel.worker
   ```

4. **Run the parallel experiment starter** in another terminal:

   ```bash
   uv run -m src.workflows.train_tune.bert_parallel.starter
   ```

   This:

   - Builds a `CoordinatorWorkflowInput` with one or more `CoordinatorWorkflowConfig` entries.
   - Each config defines:
     - A `BertFineTuneConfig` (model, dataset, hyperparameters, sampling, seed).
     - A `BertEvalRequest` (dataset split + eval settings).
   - Starts `CoordinatorWorkflow` on `bert-eval-task-queue`.
   - The coordinator:
     - Spins up a `CheckpointedBertTrainingWorkflow` child per config on `bert-training-task-queue`.
     - After training, starts a `BertEvalWorkflow` child per config.
   - Returns a list of `BertEvalResult` objects—one per configuration.

## Durability demo

To see Temporal’s durability with parallel runs:

1. Start both workers and run the `starter` script.
2. Once several training runs are in progress, **kill the training worker** (Ctrl‑C).
3. Restart the training worker:

   ```bash
   uv run -m src.workflows.train_tune.bert_parallel.training_worker
   ```

4. Observe in Temporal Web and logs that:
   - Each `CheckpointedBertTrainingWorkflow` resumes from its last checkpointed state (using `CheckpointInfo` from signals).
   - Once all training runs complete, the coordinator still executes evaluation for each configuration and returns a coherent set of `BertEvalResult` values.

Because all dataset/model I/O is in activities, workflows remain deterministic and can be replayed safely even after failures.

## Why Temporal (for this example)

- **Parallel, checkpoint‑aware experiments** – Multiple BERT runs (different models/datasets/hyperparams) can execute safely in parallel, each with its own snapshots and checkpoints.
- **Crash‑safe orchestration** – Child workflows and activity retries ensure long‑running training and evaluation survive restarts.
- **Deterministic replay** – Orchestration is pure Python; all nondeterminism (I/O, randomness, ML libraries) is in activities.
- **Flexible scaling** – Training and eval workers run on separate queues so you can scale GPU‑heavy work independently from orchestration and evaluation.

## Repo map (local to this folder)

- `custom_types.py` – Pydantic models for:
  - Dataset snapshots and checkpoints (`DatasetSnapshot*`, `CheckpointInfo`).
  - Training/eval configs and results (`BertFineTuneConfig`, `BertEvalRequest`, `BertEvalResult`).
  - Inference types (`BertInferenceRequest`, `BertInferenceResult`).
  - Coordinator config and inputs (`CoordinatorWorkflowConfig`, `CoordinatorWorkflowInput`).
- `bert_activities.py` – Activities for:
  - Snapshotting datasets into content‑addressed directories.
  - Schema‑aware, checkpoint‑aware fine‑tuning (`BertFineTuneActivities`).
  - Batch inference (`BertInferenceActivities`).
  - Dataset evaluation (`BertEvalActivities`).
- `workflows.py` – Temporal workflows:
  - `CheckpointedBertTrainingWorkflow` – checkpoint‑aware training with snapshot reuse.
  - `BertInferenceWorkflow` – inference wrapper.
  - `BertEvalWorkflow` – dataset evaluation of a fine‑tuned run.
  - `CoordinatorWorkflow` – orchestrates multiple training + eval runs.
- `worker.py` – Worker hosting `BertEvalWorkflow`, `CoordinatorWorkflow`, and `CheckpointedBertTrainingWorkflow` plus evaluation activities on `bert-eval-task-queue`.
- `training_worker.py` – Worker hosting `CheckpointedBertTrainingWorkflow` plus training/snapshot activities on `bert-training-task-queue`.
- `starter.py` – CLI entrypoint that kicks off `CoordinatorWorkflow` with sample configurations.
- `tests/` – Tests for activities and workflows (to be aligned with `bert_parallel` types and behaviors).

## Architecture

For a deeper architectural breakdown (workflows, activities, state model, signals/queries, timeouts, scaling), see:

- `src/workflows/train_tune/bert_parallel/docs/architecture.md`

## Competitive comparison

To understand how this parallel checkpointed pattern compares to AWS Step Functions, Azure Durable Functions, Airflow, Dagster/Prefect, and others, see:

- `src/workflows/train_tune/bert_parallel/docs/competitive-comparison.md`

## Build guide from `bert_eval`

For a step‑by‑step walkthrough of how to construct the `bert_parallel` package starting from `bert_eval` (types → activities → workflows → workers → tests), see:

- `src/workflows/train_tune/bert_parallel/docs/CREATE_BERT_PARALLEL_FROM_EVAL.md`

