"""Temporal worker for the BERT evaluation example."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.train_tune.bert_eval.bert_activities import (
    BertEvalActivities,
)
from src.workflows.train_tune.bert_eval.workflows import (
    BertEvalWorkflow,
    CheckpointedBertTrainingWorkflow,
    CoordinatorWorkflow,
)


async def main() -> None:
    # 1. Connect to Temporal Server using the same Pydantic data converter
    # used by the starter script so typed models round-trip cleanly.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Set the task queue that this worker will poll. This must match the
    # ``task_queue`` used when starting workflows from ``starter.py``.
    task_queue = "bert-eval-task-queue"

    # 3. Instantiate activity collections. For this worker we only need the
    # evaluation activities; training and checkpointing are handled elsewhere.
    eval_activities = BertEvalActivities()

    # 4. Build worker
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[BertEvalWorkflow, CoordinatorWorkflow, CheckpointedBertTrainingWorkflow],
        activities=[eval_activities.evaluate_bert_model],
        activity_executor=ThreadPoolExecutor(5),
    )

    # 5. Run worker
    await worker.run()

#CLI Hook
if __name__ == "__main__":
    asyncio.run(main())
