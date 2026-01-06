"""Tests for the BERT evaluation workflows (bert_eval package)."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from src.workflows.train_tune.bert_eval.custom_types import (
    BertEvalRequest,
    BertEvalResult,
    BertFineTuneConfig,
)
from src.workflows.train_tune.bert_eval.workflows import (
    BertEvalWorkflow,
    CheckpointedBertTrainingWorkflow,
    CoordinatorWorkflow,
)


class TestBertEvalWorkflows:
    """Test suite for BERT eval workflows using mocked activities."""

    @pytest.fixture
    def task_queue(self) -> str:
        """Generate a unique task queue name for each test."""
        return f"test-bert-eval-workflows-{uuid.uuid4()}"

    @pytest.mark.asyncio
    async def test_bert_eval_workflow_with_mocked_activity(
        self,
        client: Client,
        task_queue: str,
    ) -> None:
        """Test BERT eval workflow with a mocked activity."""

        @activity.defn(name="evaluate_bert_model")
        async def evaluate_bert_model_mocked(
            request: BertEvalRequest,
        ) -> BertEvalResult:
            activity.logger.info(
                "Mocked BERT eval for run %s on %s/%s[%s]",
                request.run_id,
                request.dataset_name,
                request.dataset_config_name,
                request.split,
            )
            return BertEvalResult(
                run_id=request.run_id or "test-bert-run",
                dataset_name=request.dataset_name,
                dataset_config_name=request.dataset_config_name,
                split=request.split,
                num_examples=100,
                accuracy=0.9,
            )

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[BertEvalWorkflow],
            activities=[evaluate_bert_model_mocked],
            activity_executor=ThreadPoolExecutor(5),
        ):
            request = BertEvalRequest(
                run_id="test-bert-run",
                dataset_name="glue",
                dataset_config_name="sst2",
                split="validation",
                max_eval_samples=100,
                max_seq_length=64,
                batch_size=16,
                use_gpu=False,
                model_path="./bert_runs/test-bert-run",
            )

            result = await client.execute_workflow(
                BertEvalWorkflow.run,
                request,
                id=f"test-bert-eval-workflow-{uuid.uuid4()}",
                task_queue=task_queue,
            )

            assert isinstance(result, BertEvalResult)
            assert result.run_id == "test-bert-run"
            assert result.dataset_name == "glue"
            assert result.dataset_config_name == "sst2"
            assert result.split == "validation"
            assert result.num_examples == 100
            assert pytest.approx(result.accuracy, rel=1e-6) == 0.9
