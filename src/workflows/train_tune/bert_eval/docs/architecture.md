# BERT Eval Coordinator – Architecture

This document explains how the `bert_eval` package composes checkpoint‑aware training and dataset evaluation on top of Temporal.

The core idea: **treat “train + evaluate” as a single, durable experiment** that can be restarted, inspected, and evolved without losing lineage.

---

## Components

### Workflows

- `CheckpointedBertTrainingWorkflow`
  - Runs a single fine‑tuning job with:
    - Dataset snapshotting via `create_dataset_snapshot`.
    - Checkpoint‑aware training via `fine_tune_bert`.
  - Tracks the latest checkpoint in workflow state via the `update_checkpoint` signal.
  - Exposes the latest checkpoint via the `get_latest_checkpoint` query.

- `BertEvalWorkflow`
  - Evaluates a fine‑tuned checkpoint on a dataset split (`train` / `validation` / `test`).
  - Delegates all model + dataset work to the `evaluate_bert_model` activity.

- `CoordinatorWorkflow`
  - Accepts a `CoordinatorWorkflowInput` containing one or more `CoordinatorWorkflowConfig` entries.
  - For each config:
    - Normalizes a canonical `run_id`.
    - Starts a child `CheckpointedBertTrainingWorkflow` on the training queue.
    - After training, starts a child `BertEvalWorkflow` on the eval queue.
  - Returns a list of `BertEvalResult` objects, one per configuration.

### Activities

- `BertCheckpointingActivities`
  - `_create_dataset_snapshot_sync`:
    - Loads a Hugging Face dataset and optionally subsamples it.
    - Writes a content‑addressed snapshot to `./data_snapshots/{snapshot_id}`.
    - Persists metadata (schema, counts, hash) in `metadata.json`.
  - `create_dataset_snapshot`:
    - Async wrapper that offloads to a worker thread and returns `DatasetSnapshotResult`.

- `BertFineTuneActivities`
  - Infers text/label columns and task type when not explicitly configured.
  - `_fine_tune_bert_sync`:
    - Loads dataset or snapshot.
    - Builds tokenizer + model.
    - Trains with Transformers’ `Trainer`, saving mid‑run checkpoints under `./bert_runs/{run_id}`.
    - Returns a rich `BertFineTuneResult` including metrics, inferred schema, and checkpoint counts.
  - `fine_tune_bert`:
    - Offloads sync training to a thread.
    - Sends heartbeats to Temporal.
    - Streams `CheckpointInfo` updates to the parent workflow via signals.

- `BertEvalActivities`
  - `_evaluate_bert_model_sync`:
    - Loads a fine‑tuned model from `model_path` or `./bert_runs/{run_id}`.
    - Runs batched evaluation on the requested dataset split.
    - Returns `BertEvalResult` with accuracy and example counts.
  - `evaluate_bert_model`:
    - Async wrapper that logs start/end and offloads evaluation to a thread.

---

## State model

### Workflow state

- `CheckpointedBertTrainingWorkflow`
  - `latest_checkpoint: CheckpointInfo | None` – updated by signals during training.
  - `run_id: str | None` – the canonical identifier for this run, used to name checkpoint directories.

- `BertEvalWorkflow`
  - Purely functional: constructs no long‑lived state, returns `BertEvalResult`.

- `CoordinatorWorkflow`
  - `run_ids: list[str]` – the list of canonical run IDs across all configs.
  - Input is a `CoordinatorWorkflowInput` with `configs: list[CoordinatorWorkflowConfig]`.
  - `set_run_id` mutates a config in place so that:
    - `run_id` is written into:
      - The coordinator config.
      - The nested `BertFineTuneConfig`.
      - The nested `BertEvalRequest`.
    - `model_path` defaults to `./bert_runs/{run_id}`.

### Signals and queries

- Signal:
  - `CheckpointedBertTrainingWorkflow.update_checkpoint(info: CheckpointInfo)`:
    - Called from `BertFineTuneActivities.fine_tune_bert` whenever a new checkpoint is produced.
    - Carries epoch, step, loss, path, and timestamp.

- Query:
  - `CheckpointedBertTrainingWorkflow.get_latest_checkpoint() -> CheckpointInfo | None`:
    - Allows dashboards or external tools to inspect progress without touching the filesystem.

---

## Determinism

Temporal workflows in this package follow strict determinism rules:

- **No direct I/O inside workflows**:
  - All dataset and model operations happen in activities (`bert_activities.py`).
  - Workflows pass **paths and configuration**, not open file handles or datasets.

- **Imports behind `workflow.unsafe.imports_passed_through()`**:
  - `workflows.py` wraps imports from `custom_types` to avoid replay issues while still using rich Pydantic models.

- **Non‑deterministic behavior delegated to Temporal APIs**:
  - Child workflows are started via `workflow.execute_child_workflow`.
  - Timeouts are expressed through `start_to_close_timeout` on activity invocations.
  - Random run IDs are generated via `workflow.uuid4()` when no ID is provided.

Because of this, you can safely:

- Replay workflows from history for debugging.
- Upgrade code (with care) and re‑run historical workflows.

---

## Timeouts, retries, and idempotency

### Activity options

- `create_dataset_snapshot`
  - `start_to_close_timeout=timedelta(minutes=10)` (in the training workflow that calls its sibling in `bert_checkpointing`; in this package it is used via the coordinator’s child training and evaluation pattern).
  - Idempotent by design:
    - If the same dataset + config + `max_samples` combination is requested again, the snapshot is reused by hash.

- `fine_tune_bert`
  - `start_to_close_timeout=timedelta(hours=2)` in the training workflows.
  - Periodic `activity.heartbeat` calls ensure:
    - Liveness tracking.
    - Heartbeat timeouts can be used to detect stuck training.
  - Retries:
    - The example relies on Temporal’s default retry policy.
    - Checkpoints let retried activities resume work without starting from scratch.

- `evaluate_bert_model`
  - `start_to_close_timeout=timedelta(minutes=10)` in `BertEvalWorkflow`.
  - Idempotent as long as the underlying dataset and model path are stable.

### Idempotency strategy

- **Business key = `run_id`**
  - All training artifacts are written under `./bert_runs/{run_id}`.
  - Training repeats with the same logical run ID will overwrite or reuse that directory.

- **Dataset snapshots**
  - `DatasetSnapshotRequest` + snapshot hashing ensure that identical dataset slices map to the same `snapshot_id`.
  - Snapshots are reused when present, avoiding duplicate work.

- **Evaluation**
  - `BertEvalRequest` references a `run_id` and model path.
  - Re‑running evaluation for the same `(run_id, dataset_name, split, config)` is safe and repeatable.

---

## Backpressure and scaling

- **Separate task queues**:
  - `bert-training-task-queue`:
    - Hosts `CheckpointedBertTrainingWorkflow` and the training/checkpointing activities.
    - Intended for GPU / accelerator machines.
  - `bert-eval-task-queue`:
    - Hosts `BertEvalWorkflow` and `CoordinatorWorkflow` plus evaluation activities.
    - Can run on CPU‑only machines.

- **Worker concurrency**:
  - `training_worker.py` sets:
    - `max_concurrent_activities=1` to avoid local OOM when running large models.
    - `max_cached_workflows` tuned conservatively.
  - `worker.py` (eval) uses a small `ThreadPoolExecutor` for light evaluation workloads.

- **Scaling out**:
  - Add more training workers on `bert-training-task-queue` to increase throughput.
  - Add more eval/coordinator workers on `bert-eval-task-queue` for heavier experiment orchestration.

---

## Failure modes and behavior

### Training worker crash / restart

- Impact:
  - The in‑flight `fine_tune_bert` activity is retried according to Temporal’s retry policy.
  - The workflow history is preserved; checkpoints already written to disk remain intact.
- Behavior:
  - The next training attempt uses:
    - The dataset snapshot from `DatasetSnapshotResult`.
    - The latest known checkpoint path from `CheckpointedBertTrainingWorkflow.latest_checkpoint`, if available.
  - The workflow resumes training instead of starting from epoch 0.

### Evaluation worker crash / restart

- Impact:
  - In‑flight `evaluate_bert_model` activities are retried.
  - Coordinator and eval workflows remain in history and can be replayed.
- Behavior:
  - Since evaluation is pure read‑only over a fixed model and dataset split, retries are safe and produce the same `BertEvalResult`.

### Upstream dataset changes

- If the source dataset (e.g., GLUE SST‑2) changes, prior runs remain reproducible as long as:
  - You use dataset snapshots for training.
  - You capture dataset metadata alongside your results.

### Non‑deterministic bugs

- If non‑deterministic logic leaks into workflows (e.g., raw `random.random()`, I/O), Temporal will detect a mismatch during replay.
- The design here intentionally centralizes such behavior inside activities, minimizing that risk.

---

## Production path (high level)

While this repository is demo‑grade, the architecture scales into production:

- Deploy Temporal Server in your chosen environment (Temporal Cloud or self‑hosted).
- Package training and eval workers into separate Docker images:
  - Training image with GPU drivers and heavy ML dependencies.
  - Eval/orchestration image with lighter runtime and fewer ML dependencies.
- Use:
  - `bert-training-task-queue` for GPU worker deployments.
  - `bert-eval-task-queue` for CPU orchestration and eval workers.
- Feed real experiment configurations into `CoordinatorWorkflowInput` from:
  - A config repo.
  - An internal experiment UI.
  - An external orchestration layer (e.g., an MLOps platform) that treats Temporal as the durable control‑plane.
