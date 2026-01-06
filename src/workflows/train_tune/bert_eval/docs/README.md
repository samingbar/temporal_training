# BERT Evaluation Coordinator (on top of Checkpointed Training)

This folder builds on the `bert_checkpointing` example and adds a **coordinated training + evaluation pipeline** for BERT. It reuses the same checkpoint‑aware training pattern, then layers on a coordinator workflow that:

- Normalizes and propagates a canonical `run_id` across training and evaluation.
- Starts checkpointed training as a child workflow per config.
- Evaluates each trained run on a public dataset and returns structured metrics.

Where `bert_checkpointing` focuses on *one* durable training + inference run, `bert_eval` turns that primitive into a reusable **end‑to‑end experiment runner**.

## What this repo demonstrates

- Coordinating **checkpointed training** and **deterministic evaluation** via Temporal child workflows.
- Sharing Pydantic types across activities, workflows, and CLI clients for end‑to‑end type safety.
- Separating training and evaluation workers so you can scale GPU‑heavy fine‑tuning independently.

## Quickstart

From the project root (`temporal_training/`):

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start the training worker** (GPU recommended):

   ```bash
   uv run -m src.workflows.train_tune.bert_eval.training_worker
   ```

3. **Start the evaluation / coordinator worker** (CPU is fine):

   ```bash
   uv run -m src.workflows.train_tune.bert_eval.worker
   ```

4. **Run the end‑to‑end evaluation starter** in another terminal:

   ```bash
   uv run -m src.workflows.train_tune.bert_eval.starter
   ```

   This:

   - Builds a `CoordinatorWorkflowInput` with one `CoordinatorWorkflowConfig` that includes:
     - A `BertFineTuneConfig` (model + dataset + hyperparameters).
     - A `BertEvalRequest` (dataset split + eval settings).
   - Starts `CoordinatorWorkflow` on `bert-eval-task-queue`.
   - The coordinator:
     - Spins up a `CheckpointedBertTrainingWorkflow` child per config on `bert-training-task-queue`.
     - After training completes, starts a `BertEvalWorkflow` child per config.
   - Returns a list of `BertEvalResult` objects and prints a concise summary (run ID, dataset, split, num examples, accuracy).

## How this enhances `bert_checkpointing`

Relative to `bert_checkpointing`, this package adds:

- **CoordinatorWorkflow**: Orchestrates multiple checkpointed training runs and their corresponding evaluations, ensuring all artifacts live under `./bert_runs/{run_id}` and that `model_path` is set consistently.
- **Flexible schema inference** for training and evaluation:
  - `BertFineTuneActivities` and `BertEvalActivities` infer text/label columns and task type from dataset features when not explicitly configured.
  - The trainer returns not only loss and accuracy but also inferred schema metadata in `BertFineTuneResult`.
- **Clear separation of concerns**:
  - Training and checkpointing activities (`BertFineTuneActivities`, `BertCheckpointingActivities`) run on a dedicated training queue.
  - Evaluation activities (`BertEvalActivities`) are hosted on a separate worker and queue.

In practice, you can point both `bert_checkpointing` and `bert_eval` at the same model family and dataset; `bert_eval` simply adds a reusable “experiment + metrics” layer on top.

## Durability demo

To see Temporal’s durability across coordinated training and evaluation:

1. Start both workers and run the starter as shown above.
2. Once training has begun (you see logs from `BertFineTuneActivities`), **kill the training worker** (Ctrl‑C).
3. Restart the training worker:

   ```bash
   uv run -m src.workflows.train_tune.bert_eval.training_worker
   ```

4. Observe in Temporal Web and logs that:
   - The child `CheckpointedBertTrainingWorkflow` resumes from the last checkpointed state.
   - After training completes, the coordinator still triggers evaluation and returns a coherent `BertEvalResult`.

Because all model and dataset I/O is in activities, workflows remain deterministic and can be replayed safely even after failures.

## Why Temporal (for this example)

- **Training + eval as a single durable unit**: Coordinator workflows track the full lifecycle of each experiment, not just individual tasks.
- **Crash‑safe orchestration**: Child workflows and activity retries ensure that long‑running fine‑tuning and evaluation survive restarts.
- **Deterministic replay**: Workflows express orchestration in pure Python while delegating non‑determinism to activities, making it easy to replay, debug, and evolve the system.
- **Scalable architecture**: Separate task queues and workers for training vs. eval let you run GPU‑heavy workloads and lightweight orchestration on appropriately sized machines.

## Repo map (local to this folder)

- `custom_types.py` – Pydantic models for training configs, evaluation configs, checkpointing, and results (`BertFineTuneConfig`, `BertEvalRequest`, `CoordinatorWorkflowInput`, etc.).
- `bert_activities.py` – Activities for:
  - Checkpointed fine‑tuning with schema inference and mid‑run checkpoint signaling.
  - Dataset snapshotting and content‑addressed storage.
  - Evaluation of fine‑tuned checkpoints on public datasets.
- `workflows.py` – Temporal workflows:
  - `CheckpointedBertTrainingWorkflow` – checkpoint‑aware training.
  - `BertEvalWorkflow` – evaluation of a fine‑tuned run on a dataset split.
  - `CoordinatorWorkflow` – ties training and eval together across one or more configs.
- `worker.py` – Worker hosting `BertEvalWorkflow` and `CoordinatorWorkflow` plus `BertEvalActivities` on `bert-eval-task-queue`.
- `training_worker.py` – Worker hosting `CheckpointedBertTrainingWorkflow` plus training/snapshot activities on `bert-training-task-queue`.
- `starter.py` – CLI entrypoint that kicks off the coordinator workflow with a sample GLUE SST-2 configuration and prints metrics.
- `tests/` – Tests for activities and workflows (to be aligned fully with the `bert_eval` types as the package evolves).

For broader project documentation, Temporal patterns, and troubleshooting, see the root `README.md` and `docs/` directory.  
For details on how the underlying checkpointed training pattern is constructed, see `src/workflows/train_tune/bert_checkpointing/README.md` and `CREATE_BERT_CHECKPOINTING.md`.

## Architecture

For a deeper look at how the workflows, activities, state model, signals/queries, and task queues fit together, see:

- `src/workflows/train_tune/bert_eval/docs/architecture.md`

That document walks through:

- Which workflows exist (`CheckpointedBertTrainingWorkflow`, `BertEvalWorkflow`, `CoordinatorWorkflow`) and what each is responsible for.
- How `run_id`, dataset snapshots, and model paths flow through the system.
- Determinism rules, timeouts, retries, and scaling/backpressure considerations.

## Competitive comparison

To understand why this Temporal pattern is preferable to common orchestration alternatives for checkpointed BERT experiments, see:

- `src/workflows/train_tune/bert_eval/docs/competitive-comparison.md`

It compares this repo’s approach against AWS Step Functions, Azure Durable Functions, Airflow, Dagster/Prefect, and others along key dimensions such as durability, long‑running support, code‑first expressiveness, deterministic replay, portability, operational ergonomics, and scaling model.

## Build guide from `bert_checkpointing`

For a step‑by‑step walkthrough of how to construct the `bert_eval` package starting from `bert_checkpointing` (types → activities → workflows → workers → tests), see:

- `src/workflows/train_tune/bert_eval/docs/CREATE_BERT_EVAL_FROM_CHECKPOINTING.md`

This guide is the evaluation‑layer counterpart to the `CREATE_BERT_CHECKPOINTING` tutorial and is useful if you want to recreate or adapt the pattern in another project.

