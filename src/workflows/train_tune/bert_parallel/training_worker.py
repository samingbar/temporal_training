"""Temporal worker for the BERT evaluation example."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.train_tune.bert_parallel.bert_activities import (
    BertCheckpointingActivities,
    BertFineTuneActivities,
)
from src.workflows.train_tune.bert_parallel.workflows import (
    CheckpointedBertTrainingWorkflow,
)


async def main() -> None:
    """Start a worker for BERT evaluation workflows."""
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    task_queue = "bert-training-task-queue"

    fine_tune_activities = BertFineTuneActivities()
    checkpointing_activities = BertCheckpointingActivities()

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[CheckpointedBertTrainingWorkflow],
        activities=[
            fine_tune_activities.fine_tune_bert,
            checkpointing_activities.create_dataset_snapshot,
        ],
        activity_executor=ThreadPoolExecutor(5),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
