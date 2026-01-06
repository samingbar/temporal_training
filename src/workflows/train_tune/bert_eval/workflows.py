"""Checkpoint-aware Temporal workflows for BERT training and inference.

This module demonstrates how to:

- Create a reproducible dataset snapshot once and reuse it across runs.
- Run checkpoint-aware fine-tuning activities that can resume from a prior
  checkpoint path.
- Track the latest checkpoint in workflow state via signals so that external
  clients can orchestrate resumptions.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from src.workflows.train_tune.bert_eval.custom_types import (
        BertEvalRequest,
        BertEvalResult,
        BertFineTuneConfig,
        BertFineTuneRequest,
        BertFineTuneResult,
        BertInferenceRequest,
        BertInferenceResult,
        CheckpointInfo,
        CoordinatorWorkflowConfig,
        CoordinatorWorkflowInput,
        DatasetSnapshotRequest,
        DatasetSnapshotResult,
    )


@workflow.defn
class CheckpointedBertTrainingWorkflow:
    """Workflow that runs checkpoint-aware fine-tuning with a shared snapshot."""

    def __init__(self) -> None:
        self.latest_checkpoint: CheckpointInfo | None = None
        self.run_id = None

    @workflow.signal
    def update_checkpoint(self, info: CheckpointInfo) -> None:
        """Record the most recent checkpoint information in workflow state."""
        self.latest_checkpoint = info

    @workflow.query
    def get_latest_checkpoint(self) -> CheckpointInfo | None:
        """Expose the most recently recorded checkpoint (if any)."""
        return self.latest_checkpoint

    @workflow.run
    async def run(self, config: BertFineTuneConfig) -> BertFineTuneResult:
        """Run a single checkpoint-aware fine-tuning job."""
        # Derive a human-friendly, unique run identifier for this workflow run.
        # We base this on Temporal's run_id so each execution gets a fresh name.

        run_id = f"bert-checkpointed-{workflow.info().run_id}"
        self.run_id = run_id

        handle = workflow.get_external_workflow_handle("bert-end2end-demo")
        await handle.signal("set_run_id", {"run_id": run_id})

        workflow.logger.info(
            "Starting checkpointed BERT run for model %s on %s/%s",
            config.model_name,
            config.dataset_name,
            config.dataset_config_name,
        )

        # Step 1: Materialize (or reuse) a dataset snapshot for this configuration.
        snapshot_request = DatasetSnapshotRequest(
            run_id=run_id,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config_name,
            max_samples=config.max_train_samples,
        )
        snapshot: DatasetSnapshotResult = await workflow.execute_activity(
            "create_dataset_snapshot",
            snapshot_request,
            start_to_close_timeout=timedelta(minutes=10),
        )

        # Step 2: Run checkpoint-aware fine-tuning, optionally resuming from the
        # latest known checkpoint path (if one has already been recorded).
        resume_from = self.latest_checkpoint.path if self.latest_checkpoint else None

        request = BertFineTuneRequest(
            run_id=run_id,
            config=config,
            dataset_snapshot=snapshot,
            resume_from_checkpoint=resume_from,
        )

        result: BertFineTuneResult | dict = await workflow.execute_activity(
            "fine_tune_bert",
            request,
            start_to_close_timeout=timedelta(hours=2),
        )

        # Guard against cases where the Pydantic data converter returns a plain
        # dict instead of a model instance (e.g., if imports are misaligned).
        if isinstance(result, dict):
            run_id = result.get("run_id")
            checkpoints_saved = result.get("total_checkpoints_saved")
        else:
            run_id = result.run_id
            checkpoints_saved = result.total_checkpoints_saved

        workflow.logger.info(
            "Completed checkpointed BERT run %s (checkpoints_saved=%s)",
            run_id,
            checkpoints_saved,
        )
        # If we got a dict back, re-wrap it as a BertFineTuneResult so callers
        # see a consistent type.
        if isinstance(result, dict):
            return BertFineTuneResult(**result)
        return result


@workflow.defn
class BertInferenceWorkflow:
    """Workflow that runs inference using a fine-tuned BERT checkpoint."""

    @workflow.run
    async def run(self, input: BertInferenceRequest) -> BertInferenceResult:
        """Execute BERT inference for a batch of texts."""
        # Handle both model instances and plain dicts defensively.
        if isinstance(input, dict):
            run_id = input.get("run_id")
            texts = input.get("texts", [])
        else:
            run_id = input.run_id
            texts = input.texts

        workflow.logger.info(
            "Starting BERT inference workflow for run %s on %s text(s)",
            run_id,
            len(texts),
        )
        result: BertInferenceResult | dict = await workflow.execute_activity(
            "run_bert_inference",
            input,
            start_to_close_timeout=timedelta(minutes=10),
        )
        if isinstance(result, dict):
            out = BertInferenceResult(**result)
        else:
            out = result
        workflow.logger.info("Completed BERT inference workflow for run %s", run_id)
        return out


@workflow.defn
class BertEvalWorkflow:
    """Workflow that evaluates a fine-tuned BERT model on a public dataset."""

    @workflow.run
    async def run(self, input: BertEvalRequest) -> BertEvalResult:
        """Execute evaluation for a fine-tuned BERT run."""
        if isinstance(input, dict):
            run_id = input.get("run_id")
            dataset_name = input.get("dataset_name")
            dataset_config_name = input.get("dataset_config_name")
            split = input.get("split")
        else:
            run_id = input.run_id
            dataset_name = input.dataset_name
            dataset_config_name = input.dataset_config_name
            split = input.split

        workflow.logger.info(
            "Starting BERT evaluation workflow for run %s on %s/%s[%s]",
            run_id,
            dataset_name,
            dataset_config_name,
            split,
        )

        result: BertEvalResult | dict = await workflow.execute_activity(
            "evaluate_bert_model",
            input,
            start_to_close_timeout=timedelta(minutes=10),
        )

        if isinstance(result, dict):
            out = BertEvalResult(**result)
        else:
            out = result

        workflow.logger.info(
            "Completed BERT evaluation workflow for run %s: accuracy=%.3f over %s examples",
            run_id,
            out.accuracy,
            out.num_examples,
        )
        return out


@workflow.defn
class CoordinatorWorkflow:
    """Coordinate checkpointed training and evaluation for one or more configs.

    From a caller's perspective this workflow turns a list of
    :class:`CoordinatorWorkflowConfig` objects into a list of
    :class:`BertEvalResult` objects. Internally it:

    - Normalizes and propagates a single ``run_id`` across training/eval
      configs so all artifacts live under ``./bert_runs/{run_id}``.
    - Starts a child :class:`CheckpointedBertTrainingWorkflow` per config.
    - Once all training runs are finished, starts a matching
      :class:`BertEvalWorkflow` per config and returns the results.
    """

    def __init__(self):
        self.run_ids = []

    def set_run_id(self, cfg: CoordinatorWorkflowConfig) -> None:
        """Choose and propagate a canonical ``run_id`` for a config.

        The same logical ``run_id`` is written into the high-level config, the
        nested fine-tuning config, and the evaluation config so that downstream
        activities can locate checkpoints and logs by directory alone.
        """
        # Normalize run_id across the coordinator-level config, training config,
        # and evaluation config so that a single logical identifier flows
        # through training and evaluation.
        if cfg.run_id:
            canonical_run_id = cfg.run_id
        elif cfg.fine_tune_config.run_id:
            canonical_run_id = cfg.fine_tune_config.run_id
        elif cfg.evaluation_config.run_id:
            canonical_run_id = cfg.evaluation_config.run_id
        else:
            workflow.logger.info("No run id provided, generating a new one.")
            canonical_run_id = str(workflow.uuid4())

        cfg.run_id = canonical_run_id
        cfg.fine_tune_config.run_id = canonical_run_id
        cfg.evaluation_config.run_id = canonical_run_id
        self.run_ids.append(canonical_run_id)

        # If the caller did not explicitly choose a model_path for evaluation,
        # default it to the run-scoped directory that training writes to. This
        # keeps all path decisions centralized in the coordinator.
        if cfg.evaluation_config.model_path is None:
            cfg.evaluation_config.model_path = f"./bert_runs/{canonical_run_id}"

    @workflow.run
    async def run(self, input: CoordinatorWorkflowInput) -> list[BertEvalResult]:
        """Execute the coordinator workflow and return per-config evaluation results."""
        workflow.logger.info("Coordinator workflow started with %s config(s)", len(input.configs))

        eval_results: list[BertEvalResult] = []

        # Step 1: normalize run IDs and launch checkpointed training children.
        for config in input.configs:
            self.set_run_id(cfg=config)

            await workflow.execute_child_workflow(
                CheckpointedBertTrainingWorkflow.run,
                BertFineTuneConfig(
                    run_id=config.run_id,
                    model_name=config.fine_tune_config.model_name,
                    dataset_name=config.fine_tune_config.dataset_name,
                    dataset_config_name=config.fine_tune_config.dataset_config_name,
                    num_epochs=config.fine_tune_config.num_epochs,
                    batch_size=config.fine_tune_config.batch_size,
                    learning_rate=config.fine_tune_config.learning_rate,
                    max_seq_length=config.fine_tune_config.max_seq_length,
                    use_gpu=bool(config.fine_tune_config.use_gpu),
                    max_train_samples=config.fine_tune_config.max_train_samples,
                    max_eval_samples=config.fine_tune_config.max_eval_samples,
                    seed=config.fine_tune_config.seed,
                ),
                id=f"checkpointed-bert-training-workflow-{config.run_id}",
                task_queue="bert-training-task-queue",
            )

        # Step 2: fan out evaluation workflows, one per training run.
        for config in input.configs:
            eval_result: BertEvalResult = await workflow.execute_child_workflow(
                BertEvalWorkflow.run,
                BertEvalRequest(
                    run_id=config.run_id,
                    dataset_name=config.evaluation_config.dataset_name,
                    dataset_config_name=config.evaluation_config.dataset_config_name,
                    split=config.evaluation_config.split,
                    max_eval_samples=config.evaluation_config.max_eval_samples,
                    max_seq_length=config.evaluation_config.max_seq_length,
                    batch_size=config.evaluation_config.batch_size,
                    use_gpu=bool(config.evaluation_config.use_gpu),
                    model_path=config.evaluation_config.model_path,
                    seed=config.evaluation_config.seed,
                ),
                id=f"bert-eval-workflow-{config.run_id}",
                task_queue="bert-eval-task-queue",
            )
            eval_results.append(eval_result)

        workflow.logger.info(
            "Coordinator workflow completed with %s evaluation run(s)", len(eval_results)
        )
        return eval_results
