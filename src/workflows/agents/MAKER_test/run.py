"""Entry point script to execute the MakerWorkflow against a local Temporal server."""

from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from .config import ADDRESS, TASK_QUEUE
from .workflow import MakerWorkflow, NormalAgentWorkflow


async def main() -> None: 
    """Connect to Temporal and execute a single MakerWorkflow run."""
    client = await Client.connect(ADDRESS, data_converter=pydantic_data_converter)

    normal_result = client.execute_workflow(
        NormalAgentWorkflow.run,
        id="normal-agent-workflow",
        task_queue=TASK_QUEUE,
    )

    maker_result = client.execute_workflow(
        MakerWorkflow,
        id="maker-agent-workflow",
        task_queue = TASK_QUEUE
    )

    await asyncio.gather(normal_result,maker_result)

    print(f"Normal Agent proceeded {normal_result} steps. \nMakerWorkflow proceeded {maker_result} steps.")  # noqa: T201


if __name__ == "__main__": 
    asyncio.run(main())
