# Creating the BERT Fine-Tuning Demo from Scratch

This document is a **step‑by‑step build log** for recreating the `bert_finetune` example from an empty folder. It is the baseline pattern that `bert_checkpointing` and `bert_eval` build on.

> Goal: a small, reproducible package that runs **BERT fine‑tuning + inference** on Temporal, without dataset snapshots or checkpoint‑aware resume.

---

## 0. Prerequisites

- Python 3.12+
- Temporal dev server:

  ```bash
  temporal server start-dev
  ```

- `uv` for dependency + environment management:

  ```bash
  pip install uv
  ```

**Why this step**  
You need a modern Python runtime, a running Temporal server, and a reproducible environment tool so the rest of the tutorial can focus on workflows and activities, not setup issues.

---

## 1. Directory and file scaffold

From your repo root (or a new one):

```bash
mkdir -p src/workflows/train_tune/bert_finetune/tests
touch src/workflows/train_tune/bert_finetune/{__init__.py,custom_types.py,bert_activities.py,bert_workflow.py,worker.py,train.py,inference.py}
```

If you’re creating a standalone repo, ensure `pyproject.toml` includes at least:

- `temporalio`
- `pydantic`
- `transformers[torch]`
- `datasets`
- `torch`

Then install:

```bash
uv sync --dev
```

**Why this step**  
You carve out a self‑contained `bert_finetune` package with clear slots for types, activities, workflows, worker, CLIs, and tests—mirroring how production Temporal projects are organized.

---

## 2. Shared data models (`custom_types.py`)

Create Pydantic models that will be shared between:

- Workflows (inputs/outputs).
- Activities (inputs/outputs).
- External clients (`train.py`, `inference.py`).

Key models:

1. **Training**
   - `BertFineTuneConfig`:
     - Base model + dataset fields:
       - `model_name: str` (e.g., `"bert-base-uncased"`).
       - `dataset_name: str` (e.g., `"glue"`).
       - `dataset_config_name: str` (e.g., `"sst2"`).
     - Hyperparameters:
       - `num_epochs: int`
       - `batch_size: int`
       - `learning_rate: float`
       - `max_seq_length: int`
       - `use_gpu: bool`
     - Sampling for fast demos:
       - `max_train_samples: int | None`
       - `max_eval_samples: int | None`
   - `BertFineTuneRequest`:
     - `run_id: str`
     - `config: BertFineTuneConfig`
   - `BertFineTuneResult`:
     - `run_id: str`
     - `config: BertFineTuneConfig`
     - `train_loss: float`
     - `eval_accuracy: float | None`
     - `training_time_seconds: float`
     - `num_parameters: int`

2. **Inference**
   - `BertInferenceRequest`:
     - `run_id: str`
     - `texts: list[str]`
     - `max_seq_length: int`
     - `use_gpu: bool`
   - `BertInferenceResult`:
     - `run_id: str`
     - `texts: list[str]`
     - `predicted_labels: list[int]`
     - `confidences: list[float]`

3. **Workflow‑level models**
   - `BertExperimentInput`:
     - `experiment_name: str`
     - `runs: list[BertFineTuneConfig]`
   - `BertExperimentOutput`:
     - `experiment_name: str`
     - `runs: list[BertFineTuneResult]`

**Why this step**  
Centralizing all types in `custom_types.py` gives you a single source of truth for experiment configuration and results, and lets the Pydantic data converter handle serialization automatically across workflows, activities, and CLIs.

---

## 3. ML activities (`bert_activities.py`)

Activities own all non‑deterministic ML work: dataset access, tokenization, training, and inference. Workflows must stay deterministic and only orchestrate these activities.

### 3.1 Fine-tuning activity

1. Define sync helper `_fine_tune_bert_sync(request: BertFineTuneRequest) -> BertFineTuneResult`:

   - Import heavy ML deps inside the function to keep the module importable even if deps are missing:

     ```python
     try:
         import torch
         from datasets import load_dataset
         from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
     except ImportError as exc:
         raise RuntimeError(TRANSFORMERS_IMPORT_MESSAGE) from exc
     ```

   - Pick device:
     - If `config.use_gpu` and CUDA available → `cuda`.
     - Else if MPS available → `mps`.
     - Else → `cpu`.

   - Load dataset and tokenizer:
     - `raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name)`
     - `tokenizer = AutoTokenizer.from_pretrained(config.model_name)`

   - Define `tokenize_function` that tokenizes the `"sentence"` field with `max_length=config.max_seq_length`.
   - Map tokenizer over the dataset, rename `"label"` → `"labels"`, and set PyTorch format for `["input_ids", "attention_mask", "labels"]`.

   - Subsample if requested:
     - Use `max_train_samples` and `max_eval_samples` to `select(range(...))` from train/eval splits.

   - Build model and Trainer:
     - `AutoModelForSequenceClassification.from_pretrained(config.model_name, num_labels=2)`.
     - `TrainingArguments` with:
       - `output_dir=f"./bert_runs/{request.run_id}"`
       - `num_train_epochs=float(config.num_epochs)`
       - Batch sizes, learning rate.
       - `eval_strategy="epoch"` if eval dataset exists, otherwise `"no"`.
       - `save_strategy="no"` (no mid‑run checkpoints in this baseline).

   - Define `compute_metrics` that returns `{"accuracy": ...}` and pass it to `Trainer` if an eval set exists.
   - Run `trainer.train()`, then `trainer.evaluate()` if eval is available.
   - Save model + tokenizer to `./bert_runs/{run_id}`.
   - Compute:
     - `num_parameters` from `model.parameters()`.
     - `training_time_seconds` from a monotonic timer.
   - Return a `BertFineTuneResult`.

2. Wrap in async activity:

   ```python
   @activity.defn
   async def fine_tune_bert(request: BertFineTuneRequest) -> BertFineTuneResult:
       training_task = asyncio.create_task(asyncio.to_thread(_fine_tune_bert_sync, request))
       try:
           while not training_task.done():
               activity.heartbeat({"run_id": request.run_id})
               await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
           return await training_task
       except asyncio.CancelledError:
           training_task.cancel()
           with contextlib.suppress(Exception):
               await training_task
           raise
   ```

**Why this step**  
The sync helper encapsulates all ML logic and is easy to unit‑test. The async wrapper integrates with Temporal’s heartbeat and cancellation semantics while keeping the event loop responsive.

### 3.2 Inference activity

3. Define sync helper `_run_bert_inference_sync(request: BertInferenceRequest) -> BertInferenceResult`:

   - Import `torch`, `AutoTokenizer`, `AutoModelForSequenceClassification`.
   - Choose device using the same logic as training.
   - Load model + tokenizer from `./bert_runs/{request.run_id}`.
   - Tokenize `request.texts` with the configured `max_seq_length`.
   - Run model in `torch.no_grad()`, apply softmax, and take argmax and max probability.
   - Return `BertInferenceResult` with predicted labels and confidences.

4. Wrap in async activity:

   ```python
   @activity.defn
   async def run_bert_inference(request: BertInferenceRequest) -> BertInferenceResult:
       result = await asyncio.to_thread(_run_bert_inference_sync, request)
       return result
   ```

**Why this step**  
Inference is read‑only but still performs I/O and heavy compute; it belongs in an activity, not in a workflow. Reusing the same pattern as training keeps things symmetric and testable.

---

## 4. Workflows (`bert_workflow.py`)

Workflows orchestrate training and inference while remaining deterministic and free of direct I/O.

1. Use `workflow.unsafe.imports_passed_through()` around imports from `custom_types.py`:

   ```python
   from temporalio import workflow

   with workflow.unsafe.imports_passed_through():
       from .custom_types import (
           BertExperimentInput,
           BertExperimentOutput,
           BertFineTuneRequest,
           BertFineTuneResult,
           BertInferenceRequest,
           BertInferenceResult,
       )
   ```

2. Implement `BertFineTuningWorkflow`:

   - `@workflow.defn` with a `@workflow.run` method:
     - Logs experiment name and run count.
     - Iterates `enumerate(input.runs)`.
     - For each config, derives a `run_id` such as:

       ```python
       run_id = f"{input.experiment_name}-run-{idx}-{cfg.model_name.replace('/', '_')}"
       ```

     - Builds a `BertFineTuneRequest(run_id=run_id, config=cfg)`.
     - Calls `workflow.execute_activity("fine_tune_bert", request, start_to_close_timeout=timedelta(hours=2))`.
     - Collects results into a list and returns a `BertExperimentOutput`.

3. Implement `BertInferenceWorkflow`:

   - `@workflow.defn` with a `@workflow.run` method:
     - Logs requested run ID and number of texts.
     - Calls `workflow.execute_activity("run_bert_inference", input, start_to_close_timeout=timedelta(minutes=10))`.
     - Returns `BertInferenceResult`.

**Why this step**  
This is the “pure orchestration” layer: it sequences fine‑tune and inference activities but never touches the filesystem or ML libraries directly, preserving determinism.

---

## 5. Worker (`worker.py`)

Wire workflows and activities together in a worker process.

1. Connect a Temporal client:

   ```python
   client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
   ```

2. Create a `Worker` on `bert-finetune-task-queue`:

   ```python
   worker = Worker(
       client,
       task_queue="bert-finetune-task-queue",
       workflows=[BertFineTuningWorkflow, BertInferenceWorkflow],
       activities=[fine_tune_bert, run_bert_inference],
       activity_executor=ThreadPoolExecutor(5),
   )
   await worker.run()
   ```

**Why this step**  
The worker is the runtime engine for your workflows and activities. Keeping all registration in one module makes it easy to scale and to see what’s running where.

---

## 6. CLI entrypoints (`train.py`, `inference.py`)

### 6.1 Training CLI (`train.py`)

1. Connect a Temporal client with the Pydantic data converter.
2. Build a `BertExperimentInput` with a human‑readable `experiment_name` and a list of `BertFineTuneConfig` objects.
3. Call:

   ```python
   result = await client.execute_workflow(
       BertFineTuningWorkflow.run,
       input_data,
       id="bert-finetune-demo-id",
       task_queue="bert-finetune-task-queue",
   )
   ```

4. Pretty‑print each `BertFineTuneResult` (run ID, epochs, batch size, loss, accuracy, time).

### 6.2 Inference CLI (`inference.py`)

1. Connect a Temporal client.
2. Build a `BertInferenceRequest`:
   - Set `run_id` to one of the training run IDs used earlier.
   - Provide a small list of texts.
3. Execute the workflow:

   ```python
   result = await client.execute_workflow(
       BertInferenceWorkflow.run,
       request,
       id="bert-inference-demo-id",
       task_queue="bert-finetune-task-queue",
   )
   ```

4. Print predictions and confidences per text.

**Why this step**  
These scripts are your “happy path” demos—copy‑pasteable commands that exercise the full stack (client → workflow → activities → results) and serve as live documentation.

---

## 7. Tests

1. **Activity tests** (`tests/bert_activities_tests.py`):
   - Use `ActivityEnvironment` and patch the sync helpers:
     - Patch `_fine_tune_bert_sync` to return a dummy `BertFineTuneResult`.
     - Patch `_run_bert_inference_sync` to return a dummy `BertInferenceResult`.
   - Assert that the async activities return the expected models and that the sync helpers are called once.

2. **Workflow tests** (`tests/bert_workflow_tests.py`):
   - Spin up a `Worker` with:
     - `BertFineTuningWorkflow`, `BertInferenceWorkflow`.
     - Mocked `fine_tune_bert` and `run_bert_inference` activities.
   - For training:
     - Verify that providing multiple configs yields a `BertExperimentOutput` with the expected number of runs and that metrics behave as mocked.
   - For inference:
     - Verify that the workflow delegates to the activity and returns a `BertInferenceResult`.

**Why this step**  
Tests ensure you can evolve the internals (e.g., change metric structure, modify sampling behavior) while preserving the external API that other code and demos rely on.

---

## 8. Summary

To build `bert_finetune` from scratch, you:

1. Scaffold a dedicated package with types, activities, workflows, worker, and CLIs.
2. Define shared Pydantic models for training and inference.
3. Implement ML activities as pure sync helpers wrapped in async Temporal activities.
4. Write deterministic workflows that orchestrate those activities.
5. Wire everything together in a worker and expose simple CLI entrypoints.
6. Add tests to validate behavior and guard against regressions.

Once this baseline is solid, you can extend it to:

- `bert_checkpointing` for dataset snapshots and checkpoint‑aware resume.
- `bert_eval` for coordinated training + evaluation experiments.

