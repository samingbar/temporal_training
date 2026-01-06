# Creating the BERT Parallel Demo from `bert_eval`

This document is a **step‑by‑step build log** for evolving the `bert_eval` package into the more advanced `bert_parallel` package.

> Starting point: you already have a working `bert_eval` package with:
> - `custom_types.py` (checkpointed training + eval config/result types)
> - `bert_activities.py` (snapshotting + checkpoint‑aware training + eval)
> - `workflows.py` (checkpointed training + eval + coordinator)
> - `worker.py`, `training_worker.py`, `starter.py`, and tests.
>
> Goal: create `bert_parallel`, which focuses on **parallel, multi‑config BERT training + evaluation**, preserving the same checkpointing mechanics but tuned for orchestrating *many* runs at once.

---

## 0. Prerequisites

You can reuse the same prerequisites as `bert_eval`:

- Python 3.12+
- Temporal dev server:

  ```bash
  temporal server start-dev
  ```

- `uv` for env + deps:

  ```bash
  pip install uv
  uv sync --dev
  ```

**Why this step**  
You already have the dependencies and Temporal setup from `bert_eval`. `bert_parallel` adds orchestration logic on top; no new libraries are required.

---

## 1. Create the `bert_parallel` package skeleton

From your repo root:

```bash
mkdir -p src/workflows/train_tune/bert_parallel/tests
mkdir -p src/workflows/train_tune/bert_parallel/docs
touch src/workflows/train_tune/bert_parallel/{__init__.py,custom_types.py,bert_activities.py,workflows.py,worker.py,training_worker.py,starter.py}
```

**Why this step**  
You mirror the structure of `bert_eval`, but keep `bert_parallel` as its own, focused package so that parallel experiment patterns can evolve independently.

---

## 2. Seed `custom_types.py` from `bert_eval`

Use `bert_eval.custom_types` as your starting template.

1. Copy the following types into `bert_parallel/custom_types.py`:
   - Snapshot + checkpoint:
     - `DatasetSnapshotRequest`, `DatasetSnapshotResult`, `CheckpointInfo`.
   - Training:
     - `BertFineTuneConfig`, `BertFineTuneRequest`, `BertFineTuneResult`.
   - Inference:
     - `BertInferenceRequest`, `BertInferenceResult`.
   - Evaluation:
     - `BertEvalRequest`, `BertEvalResult`.
   - Coordinator:
     - `CoordinatorWorkflowConfig`, `CoordinatorWorkflowInput`.

2. If you plan to run more diverse datasets/models in parallel, ensure `BertFineTuneConfig` includes:
   - Sampling controls: `max_train_samples`, `max_eval_samples`.
   - Schema overrides: `text_field`, `text_pair_field`, `label_field`, `task_type`.
   - Repro knobs: `seed`, `shuffle_before_select`.

**Why this step**  
`bert_parallel` reuses the same “contract” as `bert_eval`, but it is explicitly designed to manage *lists* of configurations. Keeping types aligned ensures:

- Activities and workflows can move between `bert_eval` and `bert_parallel` easily.
- Higher‑level orchestration (e.g. sweeps/ladder patterns) can reuse config/result objects across packages.

---

## 3. Extend `bert_activities.py` for parallel scenarios

Start from `bert_eval.bert_activities` and copy its classes into `bert_parallel/bert_activities.py`:

- `BertCheckpointingActivities`
- `BertFineTuneActivities`
- `BertEvalActivities`
- `BertInferenceActivities`

Then refine where necessary.

### 3.1 Keep snapshot + checkpoint‑aware training unchanged

1. **Snapshotting** – `BertCheckpointingActivities`:
   - Reuse the same `_create_dataset_snapshot_sync` and `create_dataset_snapshot` logic as `bert_eval`.
   - This gives you content‑addressed dataset snapshots under `./data_snapshots/{snapshot_id}`.

2. **Training** – `BertFineTuneActivities`:
   - Keep:
     - Schema inference (`_infer_text_fields`, `_infer_label_field_and_task`).
     - Tokenization (`tokenize_function`).
     - Label casting (`_cast_labels`).
     - Trainer configuration and checkpoint‑aware resume logic.
   - Ensure `BertFineTuneResult` in `custom_types` matches what `_fine_tune_bert_sync` returns (including metrics and inferred schema fields if you use them).

3. **Inference** – `BertInferenceActivities`:
   - Keep `_run_bert_inference_sync` and `run_bert_inference` behavior from `bert_eval`, including:
     - Loading from `./bert_runs/{run_id}`.
     - Device selection.
     - Batched inference.

### 3.2 Evaluation tailored for parallel runs

4. **Evaluation** – `BertEvalActivities`:
   - Reuse `_evaluate_bert_model_sync` from `bert_eval`:
     - Load model from `request.model_path` or `./bert_runs/{run_id}`.
     - Load requested dataset split.
     - Apply optional subsampling (`max_eval_samples`).
     - Infer text fields, tokenize, and compute accuracy.
   - Keep `evaluate_bert_model` as the async wrapper.

**Why this step**  
The core of `bert_parallel` is not new activity behavior but **how those activities are orchestrated in parallel**. Reusing `bert_eval`’s activities keeps logic consistent and lets you focus on multi‑config workflow design.

---

## 4. Build `workflows.py` on top of `bert_eval` workflows

Use `bert_eval.workflows` as the base and adapt it to reflect the parallel experiment focus.

1. **Imports with determinism**:
   - Wrap imports from `bert_parallel.custom_types` in `workflow.unsafe.imports_passed_through()`.

2. **CheckpointedBertTrainingWorkflow**:
   - Copy the implementation from `bert_eval` (or `bert_checkpointing`) but import from `bert_parallel.custom_types`.
   - Responsibilities remain:
     - Normalize `run_id`.
     - Create snapshot via `create_dataset_snapshot`.
     - Run `fine_tune_bert` with `BertFineTuneRequest`.
     - Log and return `BertFineTuneResult`.

3. **BertInferenceWorkflow**:
   - identical to `bert_eval`’s inference workflow; reuse it.

4. **BertEvalWorkflow**:
   - identical to `bert_eval`’s eval workflow; reuse it.

5. **CoordinatorWorkflow** (the key difference):
   - Start from `bert_eval.CoordinatorWorkflow` and ensure:
     - Input type is `CoordinatorWorkflowInput` with `configs: list[CoordinatorWorkflowConfig]`.
     - `set_run_id`:
       - Normalizes `run_id` across:
         - Top‑level config.
         - Nested `fine_tune_config`.
         - Nested `evaluation_config`.
       - Sets default `model_path` to `./bert_runs/{run_id}`.
     - `run` method:
       - Loops over `input.configs`:
         - Calls `set_run_id(cfg)`.
         - Starts a child `CheckpointedBertTrainingWorkflow.run` per config on `bert-training-task-queue`.
       - Waits for all training children (via `asyncio.gather` or sequential awaits).
       - Then loops again, starting a child `BertEvalWorkflow.run` per config on `bert-eval-task-queue`.
       - Collects and returns a `list[BertEvalResult]`.

**Why this step**  
The main enhancement from `bert_eval` to `bert_parallel` is *volume and coordination*:

- `bert_eval` makes it easy to run a single coordinated training + eval pair.
- `bert_parallel` generalizes that idea to many configs at once using the same primitives.

---

## 5. Wire workers (`worker.py`, `training_worker.py`)

### 5.1 Training worker

1. In `training_worker.py`:
   - Connect a Temporal client with Pydantic data converter.
   - Use `task_queue = "bert-training-task-queue"`.
   - Register:
     - Workflow: `CheckpointedBertTrainingWorkflow`.
     - Activities:
       - `BertFineTuneActivities.fine_tune_bert`.
       - `BertCheckpointingActivities.create_dataset_snapshot`.
   - Configure a `ThreadPoolExecutor` for activity execution and tune concurrency based on hardware.

### 5.2 Eval / coordinator worker

2. In `worker.py`:
   - Connect a Temporal client with Pydantic data converter.
   - Use `task_queue = "bert-eval-task-queue"`.
   - Register:
     - Workflows:
       - `BertEvalWorkflow`.
       - `CoordinatorWorkflow`.
       - Optionally `CheckpointedBertTrainingWorkflow` if you want flexibility in routing.
     - Activities:
       - `BertEvalActivities.evaluate_bert_model`.

**Why this step**  
Splitting workers by queue lets you:

- Run training on GPU‑backed workers.
- Run eval/coordinator workflows on cheaper CPU nodes.
- Scale each side independently.

---

## 6. Starter script (`starter.py`)

Use `bert_eval/starter.py` as the template and extend it for multiple configs:

1. Connect a Temporal client with Pydantic data converter.
2. Construct several `CoordinatorWorkflowConfig` entries with different:
   - Models (e.g., `"bert-base-uncased"`, `"distilbert-base-uncased"`, `"microsoft/deberta-v3-small"`).
   - Datasets (e.g., GLUE SST‑2, IMDB, SciCite).
   - Hyperparameters (epochs, batch size, learning rate, sequence length).
3. Build a `CoordinatorWorkflowInput(configs=[config_1, config_2, ...])`.
4. Invoke:

   ```python
   results = await client.execute_workflow(
       CoordinatorWorkflow.run,
       request,
       id="bert-parallel-demo",
       task_queue="bert-eval-task-queue",
   )
   ```

5. Print a concise summary per `BertEvalResult` (run ID, dataset, split, num examples, accuracy).

**Why this step**  
The starter script becomes your “parallel experiment harness”—a single CLI command that demonstrates how multiple experiment configs are driven through the same checkpointed training + eval stack.

---

## 7. Tests

Build on the test patterns from `bert_eval`:

1. **Activity tests**
   - Use `ActivityEnvironment` and patch:
     - `BertFineTuneActivities._fine_tune_bert_sync` to return dummy `BertFineTuneResult` with metrics.
     - `BertEvalActivities._evaluate_bert_model_sync` to return dummy `BertEvalResult`.
   - Assert:
     - Async activities return the expected results.
     - The sync helpers are called exactly once.

2. **Workflow tests**
   - Spin up a `Worker` with:
     - `BertEvalWorkflow` (and optionally `CheckpointedBertTrainingWorkflow`, `CoordinatorWorkflow`).
     - Mocked activities.
   - For `BertEvalWorkflow`:
     - Validate it delegates to `evaluate_bert_model` and returns a `BertEvalResult`.
   - Optionally, for `CoordinatorWorkflow` (heavier integration test):
     - Mock snapshot, training, and eval activities.
     - Verify:
       - It returns a list of results matching the number of configs.
       - Run IDs and model paths are normalized correctly.

**Why this step**  
The tests ensure your parallel orchestration logic remains correct as you tweak activity behavior or extend configuration space. They also catch accidental regressions if you refactor workflows or types.

---

## 8. Summary

To create `bert_parallel` from `bert_eval`, you:

1. Copy and adapt shared types for parallel experiments.
2. Reuse snapshotting, checkpoint‑aware training, inference, and evaluation activities.
3. Build workflows that:
   - Reuse `CheckpointedBertTrainingWorkflow` and `BertEvalWorkflow`.
   - Extend `CoordinatorWorkflow` to orchestrate **many** training + eval runs at once.
4. Wire training and eval workers to separate queues.
5. Add a starter script and tests that showcase and validate parallel experiment behavior.

This pattern generalizes to other model families and datasets whenever you need **parallel, checkpoint‑aware ML experimentation** with Temporal.

