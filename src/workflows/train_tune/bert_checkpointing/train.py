"""CLI entrypoint for checkpoint-aware BERT training.

This script drives the ``CheckpointedBertTrainingWorkflow`` in the
``bert_checkpointing`` package, which:

- Creates (or reuses) a dataset snapshot for reproducible training.
- Runs checkpoint-aware fine-tuning that can resume from a prior checkpoint.
"""

import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_checkpointing.checkpointed_training import (
    CheckpointedBertTrainingWorkflow,
)
from src.workflows.train_tune.bert_checkpointing.custom_types import BertFineTuneConfig


async def main() -> None:
    """Execute a sample checkpoint-aware BERT training workflow."""
    # 1. Connect to the Temporal server using the Pydantic data converter so
    #    that our Pydantic models can be sent over the wire transparently.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Describe a single training configuration. The workflow will:
    #    - Snapshot the dataset for this configuration.
    #    - Run checkpoint-aware fine-tuning over that snapshot.
    config = BertFineTuneConfig(
        model_name="bert-base-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=20,
        batch_size=128,
        learning_rate=5e-5,
        max_seq_length=128,
        use_gpu=True,
    )

    # 3. Start the checkpointed training workflow and wait for the result.
    result = await client.execute_workflow(
        CheckpointedBertTrainingWorkflow.run,
        config,
        id="bert-checkpointed-training-demo-id",
        task_queue="bert-checkpointing-task-queue",
    )

    # 4. Print a concise summary of the run, including how many checkpoints
    #    were saved along the way.
    print("\nCheckpointed BERT training result:")
    print(
        f"- run_id={result.run_id}, "
        f"epochs={result.config.num_epochs}, "
        f"batch_size={result.config.batch_size}, "
        f"train_loss={result.train_loss:.4f}, "
        f"eval_acc={result.eval_accuracy if result.eval_accuracy is not None else 'N/A'}, "
        f"time_s={result.training_time_seconds:.1f}, "
        f"checkpoints_saved={result.total_checkpoints_saved}",
    )


if __name__ == "__main__":
    asyncio.run(main())
