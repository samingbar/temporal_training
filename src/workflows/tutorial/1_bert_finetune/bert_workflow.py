"""Temporal workflows for orchestrating BERT fine-tuning and inference.

This module contains *deterministic* orchestration logic only:

- :class:`BertFineTuningWorkflow` sequences one or more fine-tuning runs by
  delegating to the long-running ``fine_tune_bert`` activity.
- :class:`BertInferenceWorkflow` performs batch inference using a previously
  fine-tuned checkpoint by delegating to ``run_bert_inference``.

All heavy-weight ML work (loading datasets, training, and forward passes)
lives in ``bert_activities.py`` so that these workflows can be safely replayed
by Temporal without talking to external systems.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    # Data models are defined in a separate module so they can be shared across
    # activities, workflows, and external clients (e.g., CLI scripts).
    from src.workflows.train_tune.bert_finetune.custom_types import (
        BertExperimentInput,
        BertExperimentOutput,
        BertFineTuneConfig,
        BertFineTuneRequest,
        BertFineTuneResult,
        BertInferenceRequest,
        BertInferenceResult,
    )


@workflow.defn
class BertFineTuningWorkflow:
    """Workflow that sequences multiple BERT fine-tuning runs.

    Each fine-tuning run is a long-running activity, making this workflow a
    good example of how Temporal orchestrates ML experiments while leaving
    all heavy lifting to external libraries and infrastructure.
    """

    @workflow.run
    async def run(self, input: BertExperimentInput) -> BertExperimentOutput:
        """Execute the configured fine-tuning runs sequentially."""
        workflow.logger.info(
            "Starting BERT experiment '%s' with %s runs",
            input.experiment_name,
            len(input.runs),
        )

        # Collect the per-run training results so the caller can compare
        # hyperparameter settings (e.g., epochs, batch size) after the fact.
        results: list[BertFineTuneResult] = []
        for idx, cfg in enumerate(input.runs):
            # Derive a human-readable, stable run ID that ties together:
            # - The experiment name
            # - The index within the experiment
            # - The model name (with `/` made filesystem-safe)
            run_id = f"{input.experiment_name}-run-{idx}-{cfg.model_name.replace('/', '_')}"
            workflow.logger.info(
                "Triggering BERT fine-tuning run %s (epochs=%s, batch_size=%s)",
                run_id,
                cfg.num_epochs,
                cfg.batch_size,
            )
            # Package the configuration into the activity input type so that the
            # activity can remain decoupled from workflow-specific concerns.
            request = BertFineTuneRequest(run_id=run_id, config=cfg)
            result: BertFineTuneResult = await workflow.execute_activity(
                # Use the registered activity name so that the worker can route
                # this call even if the implementation moves modules.
                "fine_tune_bert",
                request,
                start_to_close_timeout=timedelta(hours=2),
            )
            results.append(result)

        workflow.logger.info(
            "Completed BERT experiment '%s' with %s runs",
            input.experiment_name,
            len(results),
        )
        return BertExperimentOutput(experiment_name=input.experiment_name, runs=results)


@workflow.defn
class BertInferenceWorkflow:
    """Workflow that runs inference using a fine-tuned BERT checkpoint.

    The workflow itself is deliberately thin: it forwards the user-provided
    :class:`BertInferenceRequest` to the ``run_bert_inference`` activity and
    returns the structured :class:`BertInferenceResult`.
    """

    @workflow.run
    async def run(self, input: BertInferenceRequest) -> BertInferenceResult:  # noqa: A002
        """Execute BERT inference for a batch of texts."""
        workflow.logger.info(
            "Starting BERT inference workflow for run %s on %s text(s)",
            input.run_id,
            len(input.texts),
        )
        result: BertInferenceResult = await workflow.execute_activity(
            "run_bert_inference",
            input,
            start_to_close_timeout=timedelta(minutes=10),
        )
        workflow.logger.info("Completed BERT inference workflow for run %s", input.run_id)
        return result
