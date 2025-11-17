"""Worker for the Ray Serve batch inference workflow.

This worker registers the workflow and activity with the Temporal Server and
executes tasks from the ``serve-inference-task-queue``. Run it alongside a
Temporal dev server and a Ray Serve deployment.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.serve_inference.serve_inference_activities import call_serve_inference
from src.workflows.serve_inference.serve_inference_workflow import ServeBatchInferenceWorkflow


async def main() -> None:
    """Start a worker that runs the Serve inference workflow and activities.

    The worker uses Pydantic data conversion for seamless model (de-)serialization
    between clients, workflows, and activities.
    """
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    task_queue = "serve-inference-task-queue"
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[ServeBatchInferenceWorkflow],
        activities=[call_serve_inference],
        activity_executor=ThreadPoolExecutor(max_workers=16),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
