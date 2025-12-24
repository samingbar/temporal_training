"""Tests for the CIFAR-10 scaling workflow."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from src.workflows.train_tune.cifar10_scaleup.cifar10_activities import (
    Cifar10TrainRequest,
    Cifar10TrainResult,
    RayScaleConfig,
)
from src.workflows.train_tune.cifar10_scaleup.cifar10_workflow import (
    Cifar10ScalingInput,
    Cifar10ScalingOutput,
    Cifar10ScalingWorkflow,
)


class TestCifar10ScalingWorkflow:
    """Test suite for Cifar10ScalingWorkflow.

    Tests cover end-to-end workflow execution with mocked training activities
    to avoid relying on Ray or deep learning libraries during tests.
    """

    @pytest.fixture
    def task_queue(self) -> str:
        """Generate a unique task queue name for each test."""
        return f"test-cifar10-scaling-workflow-{uuid.uuid4()}"

    @pytest.mark.asyncio
    async def test_cifar10_scaling_workflow_with_mocked_activity(
        self,
        client: Client,
        task_queue: str,
    ) -> None:
        """Test CIFAR-10 scaling workflow with a mocked training activity."""

        @activity.defn(name="train_cifar10_with_ray")
        async def train_cifar10_with_ray_mocked(
            request: Cifar10TrainRequest,
        ) -> Cifar10TrainResult:
            """Mocked CIFAR-10 training activity for testing."""
            activity.logger.info(
                "Mocked CIFAR-10 Ray training run %s with %s workers",
                request.run_id,
                request.scale.num_workers,
            )
            # Simulate better accuracy with more workers.
            base_accuracy = 0.60
            accuracy_gain = 0.05 * (request.scale.num_workers - 1)
            return Cifar10TrainResult(
                run_id=request.run_id,
                scale=request.scale,
                test_accuracy=base_accuracy + accuracy_gain,
                training_time_seconds=10.0 / request.scale.num_workers,
                num_parameters=100_000,
                effective_num_workers=request.scale.num_workers,
            )

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[Cifar10ScalingWorkflow],
            activities=[train_cifar10_with_ray_mocked],
            activity_executor=ThreadPoolExecutor(5),
        ):
            input_data = Cifar10ScalingInput(
                experiment_name="test-cifar10-scaling",
                scales=[
                    RayScaleConfig(
                        num_workers=1,
                        num_epochs=1,
                        batch_size=64,
                        num_gpus_per_worker=0.0,
                        description="baseline",
                    ),
                    RayScaleConfig(
                        num_workers=4,
                        num_epochs=1,
                        batch_size=64,
                        num_gpus_per_worker=0.0,
                        description="4x workers",
                    ),
                ],
                dataset_root=None,
                use_gpu=False,
                random_seed=42,
            )

            result = await client.execute_workflow(
                Cifar10ScalingWorkflow.run,
                input_data,
                id=f"test-cifar10-scaling-workflow-{uuid.uuid4()}",
                task_queue=task_queue,
            )

            assert isinstance(result, Cifar10ScalingOutput)
            assert len(result.runs) == 2

            # Verify that the higher-scale configuration achieves better accuracy
            # and that scaling affects the mocked wall-clock time.
            baseline_run = result.runs[0]
            scaled_run = result.runs[1]

            assert baseline_run.scale.num_workers == 1
            assert scaled_run.scale.num_workers == 4
            assert scaled_run.test_accuracy > baseline_run.test_accuracy
            assert scaled_run.training_time_seconds < baseline_run.training_time_seconds
