"""Workflow that orchestrates batch inference via Ray Serve activities.

The workflow remains deterministic and delegates all non-deterministic HTTP
I/O to the activity ``call_serve_inference``. It demonstrates a classic
fan-out/fan-in pattern by executing one activity per input item and gathering
results in order.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from temporalio import workflow

# Guard imports that are not required for determinism so Temporal can safely replay
with workflow.unsafe.imports_passed_through():
    from src.workflows.serve_inference.serve_inference_activities import (
        call_serve_inference,
    )
    from src.workflows.serve_inference.types import (
        BatchInferenceInput,
        BatchInferenceItem,
        BatchInferenceOutput,
        InferenceRequest,
        InferenceResponse,
    )

@workflow.defn
class ServeBatchInferenceWorkflow:
    """Orchestrates parallel Ray Serve calls via activities and aggregates results."""

    @workflow.run
    async def run(self, input: BatchInferenceInput) -> BatchInferenceOutput:  # noqa: A002
        """Run the batch inference orchestration.

        The workflow triggers one activity per input item concurrently and
        aggregates responses. Per-request timeouts are enforced at the activity
        layer; consider adding a workflow-level retry policy depending on your
        error semantics (e.g., retry on 429/503 only).
        """
        workflow.logger.info(
            "Starting batch inference for %d item(s) via %s%s",
            len(input.items),
            input.endpoint_url,
            input.route,
        )

        async def one_call(item: BatchInferenceItem) -> InferenceResponse:
            req = InferenceRequest(
                endpoint_url=input.endpoint_url,
                route=input.route,
                payload=item.payload,
                timeout_seconds=input.per_request_timeout_seconds,
            )
            return await workflow.execute_activity(
                call_serve_inference,
                req,
                start_to_close_timeout=timedelta(seconds=max(1, int(input.per_request_timeout_seconds) + 2)),
            )

        # Fan-out and gather preserving order
        tasks = [one_call(item) for item in input.items]
        results: list[InferenceResponse] = await asyncio.gather(*tasks)
        return BatchInferenceOutput(results=results)


