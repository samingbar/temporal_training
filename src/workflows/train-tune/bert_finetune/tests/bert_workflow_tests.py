"""Tests for the BERT fine-tuning workflow."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from src.workflows.bert.bert_activities import (
    BertFineTuneConfig,
    BertFineTuneRequest,
    BertFineTuneResult,
)
from src.workflows.bert.bert_workflow import (
    BertExperimentInput,
    BertExperimentOutput,
    BertFineTuningWorkflow,
)


class TestBertFineTuningWorkflow:
    """Test suite for BertFineTuningWorkflow.

    Uses a mocked fine-tuning activity to avoid depending on heavy ML libs.
    """

    @pytest.fixture
    def task_queue(self) -> str:
        """Generate a unique task queue name for each test."""
        return f"test-bert-finetune-workflow-{uuid.uuid4()}"

    @pytest.mark.asyncio
    async def test_bert_finetuning_workflow_with_mocked_activity(
        self,
        client: Client,
        task_queue: str,
    ) -> None:
        """Test BERT fine-tuning workflow with a mocked activity."""

        @activity.defn(name="fine_tune_bert")
        async def fine_tune_bert_mocked(
            request: BertFineTuneRequest,
        ) -> BertFineTuneResult:
            """Mocked fine-tuning activity for testing."""
            activity.logger.info(
                "Mocked BERT fine-tuning run %s with %s epochs",
                request.run_id,
                request.config.num_epochs,
            )
            base_acc = 0.70
            acc_gain = 0.02 * (request.config.num_epochs - 1)
            return BertFineTuneResult(
                run_id=request.run_id,
                config=request.config,
                train_loss=0.5 / request.config.num_epochs,
                eval_accuracy=base_acc + acc_gain,
                training_time_seconds=10.0 * request.config.num_epochs,
                num_parameters=110_000_000,
            )

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[BertFineTuningWorkflow],
            activities=[fine_tune_bert_mocked],
            activity_executor=ThreadPoolExecutor(5),
        ):
            input_data = BertExperimentInput(
                experiment_name="test-bert-finetune",
                runs=[
                    BertFineTuneConfig(
                        model_name="bert-base-uncased",
                        dataset_name="glue",
                        dataset_config_name="sst2",
                        num_epochs=1,
                        batch_size=8,
                        learning_rate=5e-5,
                        max_seq_length=64,
                        use_gpu=False,
                    ),
                    BertFineTuneConfig(
                        model_name="bert-base-uncased",
                        dataset_name="glue",
                        dataset_config_name="sst2",
                        num_epochs=3,
                        batch_size=8,
                        learning_rate=3e-5,
                        max_seq_length=64,
                        use_gpu=False,
                    ),
                ],
            )

            result = await client.execute_workflow(
                BertFineTuningWorkflow.run,
                input_data,
                id=f"test-bert-finetune-workflow-{uuid.uuid4()}",
                task_queue=task_queue,
            )

            assert isinstance(result, BertExperimentOutput)
            assert len(result.runs) == 2

            first_run = result.runs[0]
            second_run = result.runs[1]

            # More epochs should lead to lower loss and higher accuracy
            assert second_run.config.num_epochs > first_run.config.num_epochs
            assert second_run.train_loss < first_run.train_loss
            assert second_run.eval_accuracy > first_run.eval_accuracy

