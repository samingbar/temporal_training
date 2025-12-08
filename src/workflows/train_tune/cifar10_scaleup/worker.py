"""Worker for the CIFAR-10 Ray scaling workflow."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.worker import Worker

from src.workflows.train_tune.cifar10_scaleup.cifar10_activities import train_cifar10_with_ray
from src.workflows.train_tune.cifar10_scaleup.cifar10_workflow import Cifar10ScalingWorkflow


async def main() -> None:
    """Start a worker for CIFAR-10 scaling experiments."""
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)
    task_queue = "cifar10-ray-task-queue"
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[Cifar10ScalingWorkflow],
        activities=[train_cifar10_with_ray],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())

