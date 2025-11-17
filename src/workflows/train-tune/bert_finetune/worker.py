"""Worker for the BERT fine-tuning workflow."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.bert.bert_activities import fine_tune_bert
from src.workflows.bert.bert_workflow import BertFineTuningWorkflow


async def main() -> None:
    """Start a worker for BERT fine-tuning experiments."""
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    task_queue = "bert-finetune-task-queue"
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BertFineTuningWorkflow],
        activities=[fine_tune_bert],
        activity_executor=ThreadPoolExecutor(5),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

