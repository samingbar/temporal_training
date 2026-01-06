# Creating the BERT Eval Package from `bert_checkpointing`

This document is a **step‑by‑step build log** for turning the `bert_checkpointing` package into the more featureful `bert_eval` package.

> Starting point: you already have a working `bert_checkpointing` package with:
> - `custom_types.py` (training + inference + checkpoint types)
> - `bert_activities.py` (snapshotting + checkpoint‑aware training + inference)
> - `workflow.py` (checkpointed training + inference workflows)
> - `worker.py` and a simple CLI starter.

The goal of `bert_eval` is to **add a coordinated training + evaluation layer** on top of that:

- Introduce evaluation types and activities.
- Add a `BertEvalWorkflow` for dataset evaluation.
- Add a `CoordinatorWorkflow` that runs checkpointed training then evaluation per config.
- Provide a CLI starter that exercises the whole experiment.

---

## 0. Create the new package skeleton

1. Create the folder and basic files:

   ```bash
   mkdir -p src/workflows/train_tune/bert_eval/tests
   touch src/workflows/train_tune/bert_eval/{__init__.py,custom_types.py,bert_activities.py,workflows.py,worker.py,training_worker.py,starter.py}
   ```

2. Optionally create a `docs` subfolder for `README.md`, `architecture.md`, and `competitive-comparison.md` (this repository already has them; here we focus on the code).

**Why this step**  
You want `bert_eval` to be its own self‑contained package, parallel to `bert_checkpointing`, but reusing design patterns and, where appropriate, logic.

---

## 1. Seed `custom_types.py` from `bert_checkpointing`

Start by copying the checkpointing types, then adding evaluation‑specific models.

1. Copy the training/inference/checkpoint types:
   - `DatasetSnapshotRequest`, `DatasetSnapshotResult`
   - `CheckpointInfo`
   - `BertFineTuneConfig`, `BertFineTuneRequest`, `BertFineTuneResult`
   - `BertInferenceRequest`, `BertInferenceResult`

2. Extend `BertFineTuneResult` for richer metadata:
   - Replace `eval_accuracy: float | None` with:
     - `eval_metrics: dict[str, float] | None` (e.g., accuracy, f1, mse).
   - Add optional schema hints inferred at runtime:
     - `inferred_text_field: str | None`
     - `inferred_text_pair_field: str | None`
     - `inferred_label_field: str | None`
     - `inferred_task_type: str | None`
     - `inferred_num_labels: int | None`

3. Add evaluation‑specific models:
   - `BertEvalRequest`:
     - Optional `run_id` (filled in later by the coordinator).
     - Dataset name, config, split, `max_eval_samples`, `max_seq_length`, `batch_size`, `use_gpu`.
     - `model_path: str | None` (where to load the fine‑tuned checkpoint from).
   - `BertEvalResult`:
     - `run_id`, `dataset_name`, `dataset_config_name`, `split`.
     - `num_examples`, `accuracy`.

4. Add coordinator types:
   - `CoordinatorWorkflowConfig`:
     - Optional top‑level `run_id`.
     - `fine_tune_config: BertFineTuneConfig`.
     - `dataset_snapshot: DatasetSnapshotResult | None`.
     - `evaluation_config: BertEvalRequest`.
   - `CoordinatorWorkflowInput`:
     - `configs: list[CoordinatorWorkflowConfig]`.

**Why this step**  
Keeping all types (training, inference, evaluation, coordination) in a single `custom_types.py` mirrors the `bert_checkpointing` pattern and simplifies Pydantic data‑converter usage.

---

## 2. Build `bert_activities.py` on top of checkpointing activities

The `bert_eval` activities file extends the checkpointing activities with:

- Schema‑aware training (inferred text/label fields, task type, num labels).
- Evaluation over a dataset split.

### 2.1 Reuse and extend checkpointing activities

1. Start from `BertCheckpointingActivities` in `bert_checkpointing.bert_activities`:
   - Copy the snapshotting code unchanged into `bert_eval.bert_activities`.

2. Introduce a richer `BertFineTuneActivities`:
   - Start from the training logic in `bert_checkpointing.bert_activities`.
   - Add helper methods for schema inference:
     - `_infer_text_fields(sample: dict)` to pick `text_field`/`text_pair_field`.
     - `_infer_label_field_and_task(train_features, sample: dict)` to pick label column, task type, and num labels.
   - Update `_fine_tune_bert_sync` to:
     - Call the inference helpers and store:
       - `self.text_field`, `self.text_pair_field`.
       - `self.label_field`, `self.task_type`, `self.num_labels`.
     - Tokenize using inferred fields instead of a hard‑coded `"sentence"`.
     - Populate the richer `BertFineTuneResult` (`eval_metrics`, inferred schema fields).

3. Preserve checkpoint signaling:
   - Keep the `queue.Queue[CheckpointInfo]` and callback used to send checkpoints back to `CheckpointedBertTrainingWorkflow` via signals.
   - This gives `bert_eval` the same mid‑run checkpoint visibility as `bert_checkpointing`.

### 2.2 Add evaluation activities

4. Create `BertEvalActivities`:
   - `_evaluate_bert_model_sync(request: BertEvalRequest) -> BertEvalResult`:
     - Load tokenizer + model from `request.model_path` or `./bert_runs/{run_id}`.
     - Load the dataset split via `load_dataset(request.dataset_name, request.dataset_config_name)[request.split]`.
     - Optionally subsample using `max_eval_samples`.
     - Infer `text_field`/`text_pair_field` similarly to training.
     - Tokenize the dataset and run batched evaluation to compute accuracy.
   - `@activity.defn async def evaluate_bert_model(request: BertEvalRequest) -> BertEvalResult`:
     - Log start/end.
     - Offload sync helper via `asyncio.to_thread`.

**Why this step**  
This is where `bert_eval` diverges most from `bert_checkpointing`: you keep the snapshot + checkpoint‑aware training backbone, but you add schema inference and evaluation logic so the coordinator can return usable metrics.

---

## 3. Define `workflows.py` – training, eval, coordinator

`bert_eval.workflows` combines three workflow classes.

1. Import types inside `workflow.unsafe.imports_passed_through()`:

   - From `bert_eval.custom_types` import:
     - Training: `BertFineTuneConfig`, `BertFineTuneRequest`, `BertFineTuneResult`.
     - Inference: `BertInferenceRequest`, `BertInferenceResult`.
     - Snapshot/checkpoint: `DatasetSnapshotRequest`, `DatasetSnapshotResult`, `CheckpointInfo`.
     - Evaluation: `BertEvalRequest`, `BertEvalResult`.
     - Coordination: `CoordinatorWorkflowConfig`, `CoordinatorWorkflowInput`.

2. Reuse `CheckpointedBertTrainingWorkflow`:
   - Copy the implementation from `bert_checkpointing.workflow`, but adjust imports to use `bert_eval.custom_types`.
   - Keep:
     - `latest_checkpoint` and `run_id` fields.
     - `update_checkpoint` signal and `get_latest_checkpoint` query.
     - Snapshot + training activity invocations.

3. Add `BertEvalWorkflow`:
   - A small workflow that:
     - Logs evaluation start.
     - Calls the `evaluate_bert_model` activity with `BertEvalRequest`.
     - Normalizes dict vs model results.
     - Logs and returns `BertEvalResult`.

4. Implement `CoordinatorWorkflow`:
   - Internal state:
     - `run_ids: list[str]`.
   - Helper `set_run_id(cfg: CoordinatorWorkflowConfig)`:
     - Choose a canonical `run_id`:
       - Prefer `cfg.run_id`, then `cfg.fine_tune_config.run_id`, then `cfg.evaluation_config.run_id`.
       - Else generate `workflow.uuid4()`.
     - Write `canonical_run_id` into:
       - `cfg.run_id`
       - `cfg.fine_tune_config.run_id`
       - `cfg.evaluation_config.run_id`
     - Default `cfg.evaluation_config.model_path` to `./bert_runs/{run_id}` if not set.
   - `@workflow.run async def run(self, input: CoordinatorWorkflowInput) -> list[BertEvalResult]`:
     - Log how many configs were provided.
     - For each config:
       - Call `set_run_id(cfg)`.
       - Start a child `CheckpointedBertTrainingWorkflow.run` on `bert-training-task-queue`, passing a `BertFineTuneConfig` built from `cfg.fine_tune_config`.
     - After all training has finished, loop again and:
       - Start a child `BertEvalWorkflow.run` per config on `bert-eval-task-queue`, passing a `BertEvalRequest` built from `cfg.evaluation_config`.
       - Collect and return the list of `BertEvalResult` objects.

**Why this step**  
This is where you lift `bert_checkpointing` from “one training run per workflow” to “experiment coordinator” that glues together training and evaluation for multiple configs.

---

## 4. Wire workers and starter

### 4.1 Training worker

1. In `training_worker.py`:
   - Connect a Temporal `Client` with the Pydantic data converter.
   - Set `task_queue = "bert-training-task-queue"`.
   - Register:
     - Workflow: `CheckpointedBertTrainingWorkflow`.
     - Activities: `BertFineTuneActivities.fine_tune_bert`, `BertCheckpointingActivities.create_dataset_snapshot`.
   - Use a `ThreadPoolExecutor` for activity execution and tune concurrency as needed.

### 4.2 Eval / coordinator worker

2. In `worker.py`:
   - Connect a Temporal `Client`.
   - Set `task_queue = "bert-eval-task-queue"`.
   - Register:
     - Workflows: `BertEvalWorkflow`, `CoordinatorWorkflow`, (optionally `CheckpointedBertTrainingWorkflow` for flexibility).
     - Activities: `BertEvalActivities.evaluate_bert_model`.

### 4.3 CLI starter

3. In `starter.py`:
   - Connect a client with Pydantic converter.
   - Build a `CoordinatorWorkflowInput` with one or more `CoordinatorWorkflowConfig` entries:

     ```python
     request = CoordinatorWorkflowInput(
         configs=[
             CoordinatorWorkflowConfig(
                 fine_tune_config=BertFineTuneConfig(...),
                 evaluation_config=BertEvalRequest(...),
             ),
         ],
     )
     ```

   - Call:

     ```python
     results = await client.execute_workflow(
         CoordinatorWorkflow.run,
         request,
         id="bert-end2end-demo",
         task_queue="bert-eval-task-queue",
     )
     ```

   - Print a summary from `results[0]` (run ID, dataset, split, num examples, accuracy).

**Why this step**  
Workers and the starter are what transform your reusable orchestration code into a runnable “demo”: one command to start workers, one to run an experiment, easy to share with other engineers.

---

## 5. Tests

1. Activity tests:
   - Use `ActivityEnvironment` and patch:
     - `BertFineTuneActivities._fine_tune_bert_sync` to return a synthetic `BertFineTuneResult`.
     - `BertEvalActivities._evaluate_bert_model_sync` to return a synthetic `BertEvalResult`.
   - Assert that the async activities return the expected models and that the sync helpers are called once.

2. Workflow tests:
   - Spin up a `Worker` with:
     - `BertEvalWorkflow` (and optionally `CheckpointedBertTrainingWorkflow`, `CoordinatorWorkflow`).
     - Mocked activities as needed.
   - For `BertEvalWorkflow`:
     - Verify it delegates to `evaluate_bert_model` and returns a `BertEvalResult`.
   - For `CoordinatorWorkflow` (optional, more advanced):
     - Mock snapshot, training, and eval activities.
     - Assert it returns a list of `BertEvalResult` and that `run_id` normalization behaves as expected.

**Why this step**  
Tests ensure you can refactor the internals of `bert_eval` (e.g., change how metrics are stored) without breaking the high‑level behavior that other modules and demos depend on.

---

## 6. Summary

To create `bert_eval` from `bert_checkpointing`, you:

1. Copy and extend shared types to include evaluation and coordination.
2. Reuse snapshot + checkpointed training activities, then add schema inference and evaluation activities.
3. Build `BertEvalWorkflow` and `CoordinatorWorkflow` on top of `CheckpointedBertTrainingWorkflow`.
4. Wire training and eval workers to separate task queues.
5. Add a CLI starter and tests to validate the end‑to‑end experiment flow.

This pattern generalizes beyond BERT: the same approach can coordinate any checkpoint‑capable training loop and evaluation pipeline over Temporal.

