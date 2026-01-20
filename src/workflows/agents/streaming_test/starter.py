"""CLI starter for the streaming_test workflow.

This script connects to a local Temporal server, starts ``TestWorkflow``
with a simple text prompt, and prints the final aggregated LLM text.
"""

from __future__ import annotations

import argparse
import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from .workflow import TestWorkflow


ADDRESS = "localhost:7233"
TASK_QUEUE = "testing_task_queue"


async def main() -> None:
    """Connect to Temporal and execute a single TestWorkflow run."""
    parser = argparse.ArgumentParser(
        description="Start the streaming_test Temporal workflow.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello from streaming_test!",
        help="Prompt text to send to the TestWorkflow.",
    )
    args = parser.parse_args()

    client = await Client.connect(ADDRESS, data_converter=pydantic_data_converter)

    result = await client.execute_workflow(
        TestWorkflow.run,
        args.prompt,
        id="streaming-test-workflow",
        task_queue=TASK_QUEUE,
    )

    print("Final LLM text:", result)


if __name__ == "__main__":
    asyncio.run(main())
