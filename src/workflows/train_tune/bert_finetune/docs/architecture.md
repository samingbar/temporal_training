# BERT Fine-Tuning – Architecture

This document explains how the `bert_finetune` package uses Temporal to orchestrate baseline BERT training and inference.

It is intentionally simpler than `bert_checkpointing` and `bert_eval` and serves as a “starter pattern” for Temporal + ML workflows.

---

## Components

### Workflows

- `BertFineTuningWorkflow`
  - Input: `BertExperimentInput` (experiment name + list of `BertFineTuneConfig` runs).
  - For each config:
    - Derives a stable `run_id` of the form
      `"{experiment_name}-run-{index}-{model_name_sanitized}"`.
    - Builds a `BertFineTuneRequest(run_id, config)`.
    - Executes the `fine_tune_bert` activity with a generous timeout.
  - Output: `BertExperimentOutput` with `runs: list[BertFineTuneResult]`.

- `BertInferenceWorkflow`
  - Thin wrapper around the `run_bert_inference` activity.
  - Input: `BertInferenceRequest` (run ID + texts + tokenization settings).
  - Output: `BertInferenceResult` (texts, labels, confidences).

### Activities

- `fine_tune_bert`
  - Async Temporal activity that:
    - Offloads `_fine_tune_bert_sync` to a background thread.
    - Sends periodic heartbeats (`activity.heartbeat`) with the `run_id`.
    - Returns `BertFineTuneResult`.

- `_fine_tune_bert_sync`
  - Pure ML helper that:
    - Loads a Hugging Face dataset and tokenizer (`load_dataset`, `AutoTokenizer`).
    - Tokenizes the `"sentence"` field into `input_ids`, `attention_mask`, and `labels`.
    - Optionally subsamples train/eval sets using `max_train_samples` and `max_eval_samples`.
    - Creates `AutoModelForSequenceClassification` and `TrainingArguments`.
    - Trains using `Trainer.train()` and optionally `Trainer.evaluate()`.
    - Saves the fine-tuned model and tokenizer to `./bert_runs/{run_id}`.
    - Computes summary metrics and returns a `BertFineTuneResult`.

- `run_bert_inference`
  - Async activity that:
    - Offloads `_run_bert_inference_sync` to a background thread.
    - Logs start and end.

- `_run_bert_inference_sync`
  - Loads a fine‑tuned model + tokenizer from `./bert_runs/{run_id}`.
  - Runs batched inference on the supplied texts and returns `BertInferenceResult`.

---

## State model

### Workflow state

- `BertFineTuningWorkflow`
  - Does not maintain explicit mutable state; it:
    - Iterates over input configs.
    - Derives per‑run `run_id`s.
    - Aggregates `BertFineTuneResult` objects into a list.

- `BertInferenceWorkflow`
  - Stateless: accepts a request and returns a single `BertInferenceResult`.

All long‑lived state (models, checkpoints, logs) lives:

- In the Temporal workflow history (inputs, activity results).
- On disk under `./bert_runs/{run_id}` (owned by activities).

### Signals and queries

- `bert_finetune` deliberately does **not** use signals or queries.
- These concepts are introduced in the next stage (`bert_checkpointing`), where they are needed to track and expose mid‑run checkpoints.

---

## Determinism

Temporal workflows in this package obey the standard determinism rules:

- No direct I/O inside workflows:
  - All dataset and model operations (`load_dataset`, model loading, checkpoint I/O) happen in activities.
  - Workflows pass config objects and simple data structures only.

- `workflow.unsafe.imports_passed_through()`:
  - `bert_workflow.py` wraps imports from `custom_types` using this API to keep workflows replay‑safe while still using rich Pydantic models.

- Non‑deterministic behavior is encapsulated in activities:
  - Device selection, random initialization, and filesystem layout are all inside `_fine_tune_bert_sync` / `_run_bert_inference_sync`.

This makes workflows:

- Safe to replay for debugging.
- Straightforward to evolve into checkpoint‑aware or multi‑stage coordinators.

---

## Timeouts, retries, and idempotency

### Activity options

- `fine_tune_bert`
  - Called from workflows with:
    - `start_to_close_timeout=timedelta(hours=2)` (sized for long‑running training).
  - Heartbeats every `HEARTBEAT_INTERVAL_SECONDS`:
    - Enable Temporal to detect stuck or cancelled work.

- `run_bert_inference`
  - Called with:
    - `start_to_close_timeout=timedelta(minutes=10)` (suitable for batch inference).

The code relies on Temporal’s default retry policies. For the baseline demo, this is sufficient; the next stage (`bert_checkpointing`) adds smarter resume behavior.

### Idempotency strategy

- **Business key = `run_id`**
  - Activities write artifacts under `./bert_runs/{run_id}`.
  - Re‑running the same `run_id` overwrites that directory, but from the workflow perspective, it is a single logical experiment.

- **Evaluation**
  - `BertInferenceRequest` references an existing `run_id`.
  - Repeating the same request is read‑only with respect to the model and dataset.

For production, you would layer on:

- Versioned model directories.
- Immutable run IDs (no overwrites).
- External metadata storage (DB / registry).

---

## Backpressure and scaling

- **Task queue**
  - All work runs on `bert-finetune-task-queue`.
  - Multiple workers can be attached to this queue to increase throughput.

- **Worker concurrency**
  - The worker uses a `ThreadPoolExecutor` to offload heavy ML work from the event loop.
  - For local demos, a small pool is sufficient; production setups can tune:
    - Number of workers.
    - Per‑worker concurrency limits.

- **Scaling out**
  - Horizontal scaling is as simple as starting more workers with the same task queue and environment.
  - Heavier setups can dedicate some workers to training and others to inference by splitting queues later on (as `bert_eval` does).

---

## Failure modes and behavior

### Training worker crash / restart

- Temporal retains workflow history and `fine_tune_bert` state.
- With default retry policies:
  - The activity is retried; training restarts from scratch for that `run_id`.
  - No checkpoints are reused (that comes with `bert_checkpointing`).

### Inference worker crash / restart

- `run_bert_inference` is retried, reloading the same saved model from disk.
- Because inference is read‑only, retries are safe and deterministic.

### Non‑deterministic bugs in workflows

- If non‑deterministic behavior is accidentally introduced into workflows (e.g., direct I/O, raw randomness), Temporal will detect mismatches between replayed and recorded histories.
- Keeping orchestration logic small and delegating all side effects to activities minimizes this risk.

---

## Production path (high level)

`bert_finetune` is demo‑grade, but the pattern scales into production:

- Keep `bert_finetune` as a “thin” orchestrator around training and inference.
- Introduce:
  - Dataset snapshots + checkpoint resumption (`bert_checkpointing`).
  - Coordinated multi‑config experiments with evaluation (`bert_eval`).
- Package workers into containers:
  - Training workers with GPU access.
  - Inference workers potentially on cheaper CPU instances.

This folder is the first step in that progression.
