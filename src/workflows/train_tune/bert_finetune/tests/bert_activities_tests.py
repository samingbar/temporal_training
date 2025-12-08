"""Tests for BERT fine-tuning activities."""

from unittest.mock import patch

import pytest
from temporalio.testing import ActivityEnvironment

from src.workflows.bert.bert_activities import (
    BertFineTuneConfig,
    BertFineTuneRequest,
    BertFineTuneResult,
    fine_tune_bert,
)


class TestBertActivities:
    """Test suite for BERT fine-tuning activities.

    These tests verify Temporal integration and delegation to the synchronous
    helper without importing heavy ML dependencies.
    """

    @pytest.mark.asyncio
    async def test_fine_tune_bert_delegates_to_sync(self) -> None:
        """Verify that the async activity delegates to the sync helper."""
        activity_environment = ActivityEnvironment()
        request = BertFineTuneRequest(
            run_id="test-bert-run",
            config=BertFineTuneConfig(
                model_name="bert-base-uncased",
                dataset_name="glue",
                dataset_config_name="sst2",
                num_epochs=1,
                batch_size=8,
                learning_rate=5e-5,
                max_seq_length=64,
                use_gpu=False,
            ),
        )

        expected_result = BertFineTuneResult(
            run_id=request.run_id,
            config=request.config,
            train_loss=0.42,
            eval_accuracy=0.88,
            training_time_seconds=5.0,
            num_parameters=110_000_000,
        )

        with patch(
            "src.workflows.bert.bert_activities._fine_tune_bert_sync",
            return_value=expected_result,
        ) as mock_sync:
            result = await activity_environment.run(fine_tune_bert, request)

        assert result == expected_result
        mock_sync.assert_called_once_with(request)

