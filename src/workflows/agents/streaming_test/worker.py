import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter
from .activity import stream_llm_activity
from .workflow import TestWorkflow


interrupt_event = asyncio.Event()


async def main():
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    async with Worker(
        client,
        task_queue="testing_task_queue",
        workflows=[TestWorkflow],
        activities=[stream_llm_activity],
    ):
        # Keep the worker alive until interrupted (Ctrl+C during demos)
        await interrupt_event.wait()


if __name__ == "__main__":
    asyncio.run(main())
