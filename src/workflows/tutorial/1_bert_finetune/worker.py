"""Temporal worker for the BERT fine-tuning and inference example.

This module wires together:

- The Temporal client and Pydantic data converter.
- The task queue used by the BERT workflows and activities.
- Registration of the BERT workflows and activities with the worker.

Run this module in one terminal, and use ``train.py`` / ``inference.py`` or the
``BertFineTuningWorkflow.main``/``BertInferenceWorkflow`` entrypoints to drive
end-to-end experiments from another terminal.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.train_tune.bert_finetune.bert_activities import (
    fine_tune_bert,
    run_bert_inference,
)
from src.workflows.train_tune.bert_finetune.bert_workflow import (
    BertFineTuningWorkflow,
    BertInferenceWorkflow,
)


async def main() -> None:
    """Start a worker for BERT fine-tuning and inference workflows."""
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    task_queue = "bert-finetune-task-queue"
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BertFineTuningWorkflow, BertInferenceWorkflow],
        activities=[fine_tune_bert, run_bert_inference],
        activity_executor=ThreadPoolExecutor(5),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
