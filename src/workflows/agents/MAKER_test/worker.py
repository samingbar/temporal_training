import asyncio

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter
from .workflow import MakerWorkflow,NormalAgentWorkflow
from .activities import llm_step_activity, tool_activity
from .config import TASK_QUEUE, ADDRESS

interrupt_event = asyncio.Event()

async def main():
    client = await Client.connect(
        ADDRESS,
        data_converter=pydantic_data_converter)
    
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MakerWorkflow,NormalAgentWorkflow],
        activities=[llm_step_activity, tool_activity],
        ):
            # Keep the worker alive until interrupted (Ctrl+C during demos)
            await interrupt_event.wait()

if __name__ == "__main__":
    asyncio.run(main())
