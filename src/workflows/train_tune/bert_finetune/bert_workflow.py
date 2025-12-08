"""Workflow for orchestrating BERT fine-tuning experiments."""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel

    from src.workflows.train_tune.bert_finetune.bert_activities import (
        BertFineTuneConfig,
        BertFineTuneRequest,
        BertFineTuneResult,
        fine_tune_bert,
    )

class BertExperimentInput(BaseModel):
    """Input to the BERT fine-tuning workflow."""

    experiment_name: str
    """Human-readable label for this experiment."""

    runs: list[BertFineTuneConfig]
    """List of fine-tuning configurations to execute."""

class BertExperimentOutput(BaseModel):
    """Summary of the BERT fine-tuning experiment."""

    experiment_name: str
    """Echoed experiment name."""

    runs: list[BertFineTuneResult]
    """Per-configuration fine-tuning results."""

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

        results: list[BertFineTuneResult] = []
        for idx, cfg in enumerate(input.runs):
            run_id = f"{input.experiment_name}-run-{idx}-{cfg.model_name.replace('/', '_')}"
            workflow.logger.info(
                "Triggering BERT fine-tuning run %s (epochs=%s, batch_size=%s)",
                run_id,
                cfg.num_epochs,
                cfg.batch_size,
            )
            request = BertFineTuneRequest(run_id=run_id, config=cfg)
            result: BertFineTuneResult = await workflow.execute_activity(
                fine_tune_bert,
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

async def main() -> None:  
    """Execute a sample BERT fine-tuning workflow against a local Temporal server."""
    from temporalio.client import Client  
    from temporalio.contrib.pydantic import pydantic_data_converter  

    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    input_data = BertExperimentInput(
        experiment_name="bert-finetune-demo",
        runs=[
            BertFineTuneConfig(
                model_name="bert-base-uncased",
                dataset_name="glue",
                dataset_config_name="sst2",
                num_epochs=2,
                batch_size=16,
                learning_rate=5e-5,
                max_seq_length=128,
                use_gpu=True,
            ),
            BertFineTuneConfig(
                model_name="bert-base-uncased",
                dataset_name="glue",
                dataset_config_name="sst2",
                num_epochs=4,
                batch_size=32,
                learning_rate=3e-5,
                max_seq_length=128,
                use_gpu=True,
            ),
        ],
    )

    result = await client.execute_workflow(
        BertFineTuningWorkflow.run,
        input_data,
        id="bert-finetune-demo-id",
        task_queue="bert-finetune-task-queue",
    )

    print("\nBERT fine-tuning experiment results:") 
    for run in result.runs:
        print(
            f"- {run.run_id}: "
            f"epochs={run.config.num_epochs}, "
            f"batch_size={run.config.batch_size}, "
            f"train_loss={run.train_loss:.4f}, "
            f"eval_acc={run.eval_accuracy if run.eval_accuracy is not None else 'N/A'}, "
            f"time_s={run.training_time_seconds:.1f}",
        )

if __name__ == "__main__":  
    asyncio.run(main())

