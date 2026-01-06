# BERT Fine-Tuning Workflow (Baseline Temporal Example)

This folder contains the **baseline BERT fine-tuning demo**. It is intentionally simpler than the `bert_checkpointing` and `bert_eval` packages and serves as a precursor to them:

- `bert_finetune` shows how to orchestrate BERT training and inference with Temporal.
- `bert_checkpointing` builds on this to add dataset snapshots and checkpoint-aware resumption.
- `bert_eval` builds on both to coordinate training + evaluation across multiple configs.

At a high level this package provides:

- `bert_workflow.py` – `BertFineTuningWorkflow` and `BertInferenceWorkflow` orchestration.
- `bert_activities.py` – Side-effecting activities for fine-tuning and inference.
- `custom_types.py` – Pydantic models shared across workflows, activities, and clients.
- `worker.py` – Worker hosting the workflows and activities on `bert-finetune-task-queue`.
- `train.py` / `inference.py` – CLI-style entrypoints for running experiments from a terminal.

## Quickstart

From the project root (`temporal_training/`):

1. **Start Temporal Server**:

   ```bash
   temporal server start-dev
   ```

2. **Start the BERT fine-tuning worker**:

   ```bash
   uv run -m src.workflows.train_tune.bert_finetune.worker
   ```

3. **Run the training demo**:

   ```bash
   uv run -m src.workflows.train_tune.bert_finetune.train
   ```

   This:

   - Connects to Temporal using the Pydantic data converter.
   - Builds a `BertExperimentInput` with two `BertFineTuneConfig` entries.
   - Starts `BertFineTuningWorkflow` on `bert-finetune-task-queue`.
   - Prints a compact summary (run ID, epochs, batch size, train loss, eval accuracy, wall time) for each run.

4. **Run the inference demo**:

   ```bash
   uv run -m src.workflows.train_tune.bert_finetune.inference
   ```

   This:

   - Sends a `BertInferenceRequest` referencing one of the fine-tuned runs.
   - Executes `BertInferenceWorkflow` on the same task queue.
   - Prints predictions and confidences for a small batch of texts.

## Durability and how this differs from `bert_checkpointing`

Even without explicit checkpoint management, this example already benefits from Temporal’s durability:

- Each fine-tuning run is a long-lived `fine_tune_bert` activity.
- If the worker crashes mid-training, Temporal can retry the activity.
- The workflow history (inputs, decisions, results) is durably stored; you never “lose” the experiment orchestration.

Compared to `bert_checkpointing`:

- `bert_finetune` does **not**:
  - Snapshot datasets.
  - Track mid-run checkpoints via signals.
  - Resume from checkpoint-aware paths.
- It **does**:
  - Demonstrate the basic pattern of:
    - Pydantic config → workflow → long-running activity → result aggregation.
  - Provide a simpler code path for learning Temporal’s workflow/activity split.

Use this folder as your “hello world” for BERT + Temporal. Once you’re comfortable here, move on to `bert_checkpointing` and `bert_eval` for more advanced patterns.

## Why Temporal (for this example)

- **Durable experiments** – Fine-tuning runs live in Temporal history even if workers or clients fail.
- **Code-first orchestration** – Workflows are expressed in Python (loops, conditionals) instead of YAML DAGs.
- **Separation of concerns** – Heavy ML logic stays in `bert_activities.py`; workflows remain deterministic and replay-safe.
- **Easy evolution path** – The same pattern scales into checkpoint-aware training (`bert_checkpointing`) and full eval pipelines (`bert_eval`).

## Repo map (local to this folder)

- `custom_types.py` – Pydantic models:
  - `BertFineTuneConfig`, `BertFineTuneRequest`, `BertFineTuneResult`
  - `BertInferenceRequest`, `BertInferenceResult`
  - `BertExperimentInput`, `BertExperimentOutput`
- `bert_activities.py` – Fine-tuning + inference activities using Hugging Face Transformers and Datasets.
- `bert_workflow.py` – Deterministic workflows:
  - `BertFineTuningWorkflow` – sequentially runs multiple `fine_tune_bert` activities.
  - `BertInferenceWorkflow` – wraps `run_bert_inference` for batch scoring.
- `worker.py` – Temporal worker on `bert-finetune-task-queue`.
- `train.py` – CLI entrypoint that runs `BertFineTuningWorkflow` and prints summary metrics.
- `inference.py` – CLI entrypoint that runs `BertInferenceWorkflow` for a given `run_id`.
- `tests/` – Unit and workflow tests covering the workflows and activities.

## Architecture

For a deeper look at how workflows and activities are structured in this baseline example, see:

- `src/workflows/train_tune/bert_finetune/docs/architecture.md`

It explains:

- How `BertFineTuningWorkflow` and `BertInferenceWorkflow` are wired.
- How `fine_tune_bert` and `run_bert_inference` behave and where state lives.
- Determinism rules, timeouts, and idempotency for this precursor pattern.

## Competitive comparison

For a brief comparison of this baseline Temporal pattern against other orchestration options (Step Functions, Airflow, Dagster/Prefect, etc.), see:

- `src/workflows/train_tune/bert_finetune/docs/competitive-comparison.md`

## Build guide

For a step‑by‑step walkthrough of how to construct the `bert_finetune` package from scratch (types → activities → workflows → worker → CLIs → tests), see:

- `src/workflows/train_tune/bert_finetune/docs/CREATE_BERT_FINETUNE_FROM_SCRATCH.md`
