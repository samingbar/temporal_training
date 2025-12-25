"""CLI-style entrypoint for kicking off the BERT fine-tuning workflow.

This script is intentionally small and tutorial-friendly. It shows the steps
you would follow from *outside* Temporal to run an experiment:

1. Connect a Temporal client to the local test server.
2. Construct a ``BertExperimentInput`` describing the experiment.
3. Start ``BertFineTuningWorkflow`` with that input.
4. Print the aggregated metrics returned by the workflow.
"""

import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_finetune.bert_workflow import BertFineTuningWorkflow
from src.workflows.train_tune.bert_finetune.custom_types import (
    BertExperimentInput,
    BertFineTuneConfig,
)


async def main() -> None:
    """Execute a sample BERT fine-tuning workflow against a local Temporal server."""
    # 1. Connect to the Temporal server using the Pydantic data converter so
    #    that our Pydantic models can be sent over the wire transparently.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Describe the experiment: a human-readable name and a list of
    #    configurations to sweep over. Each configuration becomes one activity
    #    execution inside the workflow.
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

    # 3. Start the workflow and wait for the result. The workflow itself
    #    orchestrates the individual fine-tuning activities.
    result = await client.execute_workflow(
        BertFineTuningWorkflow.run,
        input_data,
        id="bert-finetune-demo-id",
        task_queue="bert-finetune-task-queue",
    )

    # 4. Pretty-print a compact summary of each run so it is easy to see how
    #    different hyperparameters affect loss, accuracy, and runtime.
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
