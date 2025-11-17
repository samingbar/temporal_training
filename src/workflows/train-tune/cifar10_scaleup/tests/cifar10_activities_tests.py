"""Tests for CIFAR-10 Ray training activities."""

from unittest.mock import patch
import pytest
from temporalio.testing import ActivityEnvironment

from src.workflows.cifar10.cifar10_activities import (
    Cifar10TrainRequest,
    Cifar10TrainResult,
    RayScaleConfig,
    train_cifar10_with_ray,
)
f
class TestCifar10Activities:
    """Test suite for CIFAR-10 training activities.

    These tests focus on integration with the Temporal activity environment and
    avoid importing heavyweight ML dependencies directly by mocking the
    synchronous training helper.
    """

    @pytest.mark.asyncio
    async def test_train_cifar10_with_ray_delegates_to_sync(self) -> None:
        """Verify that the async activity delegates to the sync helper."""
        activity_environment = ActivityEnvironment()
        request = Cifar10TrainRequest(
            run_id="test-run",
            scale=RayScaleConfig(
                num_workers=2,
                num_epochs=1,
                batch_size=64,
                num_gpus_per_worker=0.0,
                description="test-scale",
            ),
            dataset_root=None,
            use_gpu=False,
            random_seed=123,
        )

        expected_result = Cifar10TrainResult(
            run_id=request.run_id,
            scale=request.scale,
            test_accuracy=0.75,
            training_time_seconds=12.3,
            num_parameters=123_456,
            effective_num_workers=request.scale.num_workers,
        )

        with patch(
            "src.workflows.cifar10.cifar10_activities._train_cifar10_with_ray_sync",
            return_value=expected_result,
        ) as mock_train:
            result = await activity_environment.run(train_cifar10_with_ray, request)

        assert result == expected_result
        mock_train.assert_called_once_with(request)

