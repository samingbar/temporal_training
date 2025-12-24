"""Temporal worker for the BERT evaluation example."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.train_tune.bert_eval.bert_activities import (
    BertEvalActivities,
)
from src.workflows.train_tune.bert_eval.checkpointed_training import (
    BertEvalWorkflow,
)


async def main() -> None:
    """Start a worker for BERT evaluation workflows."""
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    task_queue = "bert-eval-task-queue"

    eval_activities = BertEvalActivities()
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BertEvalWorkflow],
        activities=[eval_activities.evaluate_bert_model],
        activity_executor=ThreadPoolExecutor(5),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
