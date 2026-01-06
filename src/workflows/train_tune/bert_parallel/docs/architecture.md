# BERT Parallel Training + Evaluation – Architecture

This document explains how the `bert_parallel` package coordinates multiple checkpointed BERT training and evaluation runs using Temporal.

---

## Components

### Workflows

- `CheckpointedBertTrainingWorkflow`
  - Input: `BertFineTuneConfig`.
  - Responsibilities:
    - Normalize a canonical `run_id`:
      - Use `config.run_id` if provided, otherwise derive one from `workflow.info().run_id`.
    - Build a `DatasetSnapshotRequest` for that config.
    - Call `create_dataset_snapshot` to materialize/reuse a dataset snapshot.
    - Build a `BertFineTuneRequest` with:
      - The `BertFineTuneConfig`.
      - The `DatasetSnapshotResult`.
      - Optional `resume_from_checkpoint` from the latest recorded `CheckpointInfo`.
    - Execute the `fine_tune_bert` activity.
    - Normalize dict vs model results and return `BertFineTuneResult`.
  - State:
    - `latest_checkpoint: CheckpointInfo | None`.
    - `run_id: str | None`.
  - Signals:
    - `update_checkpoint(info: CheckpointInfo)` updates `latest_checkpoint`.
  - Queries:
    - `get_latest_checkpoint() -> CheckpointInfo | None`.

- `BertInferenceWorkflow`
  - Input: `BertInferenceRequest | dict`.
  - Thin wrapper around `run_bert_inference` activity.
  - Returns `BertInferenceResult`, normalizing dict vs model inputs/outputs.

- `BertEvalWorkflow`
  - Input: `BertEvalRequest | dict`.
  - Calls `evaluate_bert_model` activity.
  - Returns `BertEvalResult`, logging dataset/split/accuracy.

- `CoordinatorWorkflow`
  - Input: `CoordinatorWorkflowInput` with `configs: list[CoordinatorWorkflowConfig]`.
  - Responsibilities:
    - Normalize `run_id` across each config:
      - Writes canonical `run_id` into:
        - The coordinator config.
        - Nested `BertFineTuneConfig`.
        - Nested `BertEvalRequest`.
      - Defaults `model_path` to `./bert_runs/{run_id}` when unset.
    - Start a child `CheckpointedBertTrainingWorkflow` per config on `bert-training-task-queue`.
    - Wait for all training children to complete.
    - Start a child `BertEvalWorkflow` per config on the eval queue.
    - Return a list of `BertEvalResult` objects.

---

## Activities

### Snapshotting – `BertCheckpointingActivities`

- `_create_dataset_snapshot_sync(request: DatasetSnapshotRequest) -> DatasetSnapshotResult`
  - Loads dataset via `load_dataset(request.dataset_name, request.dataset_config)`.
  - Optionally subsamples using `max_samples`.
  - Computes a content hash from examples.
  - Writes a JSONL snapshot + metadata into `./data_snapshots/{snapshot_id}`.
  - Reuses snapshot if directory + metadata exist.
- `create_dataset_snapshot` (async)
  - Offloads sync helper via `asyncio.to_thread`.
  - Logs start/end and returns `DatasetSnapshotResult`.

### Training – `BertFineTuneActivities`

- Responsibilities:
  - Infers text/label fields and task type for arbitrary datasets.
  - Configures and runs a checkpoint‑aware Transformers `Trainer`.
  - Emits `CheckpointInfo` via a queue to be signalled back to the workflow.
  - Returns a rich `BertFineTuneResult` including eval metrics and inferred schema.

Key methods:

- `_infer_text_fields(sample: dict)`
  - Uses config overrides when present.
  - Falls back to:
    - Common text columns (e.g., `"sentence"`, `"text"`).
    - Common pair schemas (e.g., `("premise", "hypothesis")`).
    - First string column otherwise.

- `_infer_label_field_and_task(train_features, sample: dict)`
  - Picks label column from common names or any numeric scalar column.
  - Infers `task_type` (`classification` vs `regression`).
  - Infers or approximates `num_labels`.

- `tokenize_function(batch: dict)`
  - Uses inferred text fields to tokenize each batch with `max_seq_length`.

- `_fine_tune_bert_sync(request: BertFineTuneRequest, checkpoint_queue: queue.Queue | None)`
  - Loads data from snapshot or directly from dataset.
  - Performs schema inference, tokenization, label normalization.
  - Configures `TrainingArguments` with:
    - `output_dir="./bert_runs/{run_id}"`
    - Checkpointing (e.g., `save_strategy="steps"`, `save_steps` based on dataset size).
  - Chooses `resume_from_checkpoint`:
    - Takes `request.resume_from_checkpoint` if present.
    - Else picks latest `checkpoint-*` under `output_dir`.
  - Runs `Trainer.train(resume_from_checkpoint=resume_path)` and optional `evaluate()`.
  - Saves model + tokenizer, counts checkpoints, and returns `BertFineTuneResult`.

- `fine_tune_bert` (async)
  - Offloads `_fine_tune_bert_sync` to a thread.
  - Sends heartbeats.
  - Emits `CheckpointInfo` via `update_checkpoint` signals to the training workflow.

### Inference – `BertInferenceActivities`

- `_run_bert_inference_sync(request: BertInferenceRequest) -> BertInferenceResult`
  - Loads tokenizer + model from `./bert_runs/{run_id}`.
  - Runs batched inference and returns labels + confidences.
- `run_bert_inference` (async)
  - Thread‑offloaded wrapper that logs start/end.

### Evaluation – `BertEvalActivities`

- `_evaluate_bert_model_sync(request: BertEvalRequest) -> BertEvalResult`
  - Loads a fine‑tuned checkpoint from `request.model_path` or `./bert_runs/{run_id}`.
  - Loads the requested dataset split and optionally subsamples.
  - Infers text fields and tokenizes the split.
  - Runs batched evaluation and computes accuracy (and potentially other metrics).
  - Returns `BertEvalResult`.
- `evaluate_bert_model` (async)
  - Thread‑offloaded wrapper with logging.

---

## State model

### Workflow state

- `CheckpointedBertTrainingWorkflow`
  - `latest_checkpoint: CheckpointInfo | None`
  - `run_id: str | None`

- `BertEvalWorkflow`
  - Stateless aside from parameters and return type.

- `CoordinatorWorkflow`
  - `run_ids: list[str]`
  - `run_pointers: list[ChildWorkflowHandle]` (training children).
  - `eval_pointers: list[ChildWorkflowHandle]` (eval children).

### Signals and queries

- Signals:
  - `CheckpointedBertTrainingWorkflow.update_checkpoint(info: CheckpointInfo)` records checkpoints in workflow state.
- Queries:
  - `CheckpointedBertTrainingWorkflow.get_latest_checkpoint() -> CheckpointInfo | None` exposes latest checkpoint to external tools/clients.

---

## Determinism

- All dataset/model I/O and random behavior live in activities.
- Workflows:
  - Only manipulate Pydantic models and primitive data.
  - Use Temporal APIs (`execute_activity`, `execute_child_workflow`, signals, queries).
- `workflow.unsafe.imports_passed_through()` protects imports of `custom_types` while preserving replay safety.

This means you can:

- Replay workflows for debugging.
- Evolve activity implementations without invalidating workflow history (when done carefully).

---

## Timeouts, retries, and idempotency

- Activities:
  - `create_dataset_snapshot`: `start_to_close_timeout` on the order of minutes; idempotent due to hashing.
  - `fine_tune_bert`: `start_to_close_timeout` on the order of hours; heartbeats for liveness.
  - `evaluate_bert_model` and `run_bert_inference`: `start_to_close_timeout` on the order of minutes.
- Idempotency:
  - Snapshots are keyed by content hash and dataset config.
  - Training outputs live under `./bert_runs/{run_id}`.
  - Evaluations read from a fixed model path and dataset split, making retries safe.

---

## Backpressure and scaling

- Two queues:
  - `bert-training-task-queue` for heavy training + snapshot activities.
  - `bert-eval-task-queue` for coordinator + eval workflows/activities.
- Scaling:
  - Add more training workers on GPU nodes for the training queue.
  - Add more eval/coordinator workers on CPU nodes for the eval queue.
- Concurrency:
  - Use `ThreadPoolExecutor` in workers to offload ML work and tune concurrency.

---

## Failure modes and behavior

- **Training worker crash/restart**
  - In‑flight `fine_tune_bert` activities are retried per policy.
  - Training resumes from latest checkpoint path if available, avoiding full restarts.

- **Eval worker crash/restart**
  - In‑flight `evaluate_bert_model` or `run_bert_inference` activities are retried.
  - Because these are read‑only, retries are deterministic.

- **Snapshot failures**
  - `create_dataset_snapshot` can be retried; if identical data is requested, the same snapshot is reused.

---

## Production path (high level)

The `bert_parallel` pattern scales into production as:

- A reusable coordinator for multi‑config BERT experiments.
- A foundation for:
  - Hyperparameter sweeps.
  - Multi‑dataset evaluations.
  - Multi‑model comparisons.

In a real deployment:

- Temporal Server runs in Temporal Cloud or self‑hosted.
- Training workers run on GPU nodes, eval/coordinator workers on CPU nodes.
- Experiment metadata (`BertFineTuneResult`, `BertEvalResult`, snapshot info) can be stored in a DB or experiment tracking system.

