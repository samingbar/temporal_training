"""Integration tests for ServeBatchInferenceWorkflow with mocked activity."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from src.workflows.serve_inference.serve_inference_activities import InferenceRequest
from src.workflows.serve_inference.serve_inference_workflow import (
    BatchInferenceInput,
    BatchInferenceItem,
    BatchInferenceOutput,
    ServeBatchInferenceWorkflow,
)


class TestServeBatchInferenceWorkflow:
    """Test suite for ServeBatchInferenceWorkflow."""

    @pytest.fixture
    def task_queue(self) -> str:
        return f"test-serve-inference-{uuid.uuid4()}"

    @pytest.mark.asyncio
    async def test_workflow_returns_ordered_results_with_mocked_activity(
        self, client: Client, task_queue: str
    ) -> None:
        """Should orchestrate two mocked inference calls and preserve order."""

        @activity.defn(name="call_serve_inference")
        async def mocked_inference(req: InferenceRequest):  # noqa: ANN201
            # Simulate a Ray Serve response that echoes the text
            return {"status_code": 200, "output": {"prediction": req.payload}}

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[ServeBatchInferenceWorkflow],
            activities=[mocked_inference],
            activity_executor=ThreadPoolExecutor(max_workers=8),
        ):
            input_data = BatchInferenceInput(
                endpoint_url="http://fake:8000",
                route="/inference",
                items=[
                    BatchInferenceItem(payload={"text": "a"}),
                    BatchInferenceItem(payload={"text": "b"}),
                ],
            )
            result = await client.execute_workflow(
                ServeBatchInferenceWorkflow.run,
                input_data,
                id=f"serve-inference-{uuid.uuid4()}",
                task_queue=task_queue,
            )

            assert isinstance(result, BatchInferenceOutput)
            assert len(result.results) == 2
            assert result.results[0].status_code == 200
            assert result.results[0].output == {"prediction": {"text": "a"}}
            assert result.results[1].output == {"prediction": {"text": "b"}}

