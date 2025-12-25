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
    from src.workflows.train_tune.bert_checkpointing.custom_types import (
        BertFineTuneConfig,
        BertFineTuneRequest,
        BertFineTuneResult,
        BertInferenceRequest,
        BertInferenceResult,
        CheckpointInfo,
        DatasetSnapshotRequest,
        DatasetSnapshotResult,
    )
    from src.workflows.train_tune.bert_eval.custom_types import (
        BertEvalRequest,
        BertEvalResult,
        CoordinatorWorkflowInput,
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
    """Workflow that coordinates checkpointed training, inference, and evaluation."""

    def __init__(self):
        self.run_id = None

    @workflow.run
    async def run(self, input: CoordinatorWorkflowInput) -> BertEvalResult:
        """Execute the coordinator workflow."""
        workflow.logger.info("Coordinator workflow started")

        await workflow.execute_child_workflow(
            CheckpointedBertTrainingWorkflow.run,
            BertFineTuneConfig(
                model_name=input.fine_tune_config.model_name,
                dataset_name=input.fine_tune_config.dataset_name,
                dataset_config_name=input.fine_tune_config.dataset_config_name,
                num_epochs=input.fine_tune_config.num_epochs,
                batch_size=input.fine_tune_config.batch_size,
                learning_rate=input.fine_tune_config.learning_rate,
                max_seq_length=input.fine_tune_config.max_seq_length,
                use_gpu=bool(input.fine_tune_config.use_gpu),
                max_train_samples=input.fine_tune_config.max_train_samples,
                max_eval_samples=input.fine_tune_config.max_eval_samples,
            ),
            id="checkpointed-bert-training-workflow",
            task_queue="bert-training-task-queue",
        )

        result = await workflow.execute_child_workflow(
            BertEvalWorkflow.run,
            BertEvalRequest(
                run_id=self.run_id,
                dataset_name=input.evaluation_config.dataset_name,
                dataset_config_name=input.evaluation_config.dataset_config_name,
                split=input.evaluation_config.split,
                max_eval_samples=input.fine_tune_config.max_eval_samples,
                max_seq_length=input.evaluation_config.max_seq_length,
                batch_size=input.evaluation_config.batch_size,
                use_gpu=bool(input.evaluation_config.use_gpu),
            ),
            id=f"bert-eval-workflow-{self.run_id}",
        )
        workflow.logger.info("Coordinator workflow completed")
        return result

    @workflow.signal
    def set_run_id(self, request: dict) -> None:
        """Receive the training run_id from the training workflow."""
        self.run_id = request.get("run_id")
