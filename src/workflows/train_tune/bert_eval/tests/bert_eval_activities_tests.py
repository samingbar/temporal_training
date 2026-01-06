"""Tests for BERT training and evaluation activities (bert_eval package)."""

from unittest.mock import patch

import pytest
from temporalio.testing import ActivityEnvironment

from src.workflows.train_tune.bert_eval.bert_activities import (
    BertEvalActivities,
    BertFineTuneActivities,
)
from src.workflows.train_tune.bert_eval.custom_types import (
    BertEvalRequest,
    BertEvalResult,
    BertFineTuneConfig,
    BertFineTuneRequest,
    BertFineTuneResult,
)


class TestBertEvalActivities:
    """Test suite for BERT eval/training activities with mocked ML deps."""

    @pytest.mark.asyncio
    async def test_fine_tune_bert_delegates_to_sync(self) -> None:
        """Verify that the async fine-tune activity delegates to the sync helper."""
        env = ActivityEnvironment()

        config = BertFineTuneConfig(
            model_name="bert-base-uncased",
            dataset_name="glue",
            dataset_config_name="sst2",
            num_epochs=1,
            batch_size=8,
            learning_rate=5e-5,
            max_seq_length=64,
            use_gpu=False,
        )
        request = BertFineTuneRequest(
            run_id="test-bert-run",
            config=config,
            dataset_snapshot=None,
            resume_from_checkpoint=None,
        )

        expected_result = BertFineTuneResult(
            run_id=request.run_id,
            config=request.config,
            train_loss=0.42,
            eval_metrics={"accuracy": 0.88},
            training_time_seconds=5.0,
            num_parameters=110_000_000,
            dataset_snapshot=None,
            total_checkpoints_saved=2,
        )

        with patch(
            "src.workflows.train_tune.bert_eval.bert_activities.BertFineTuneActivities._fine_tune_bert_sync",
            return_value=expected_result,
        ) as mock_sync:
            result = await env.run(BertFineTuneActivities().fine_tune_bert, request)

        assert result == expected_result
        mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluate_bert_model_delegates_to_sync(self) -> None:
        """Verify that the async eval activity delegates to the sync helper."""
        env = ActivityEnvironment()

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

        expected_result = BertEvalResult(
            run_id="test-bert-run",
            dataset_name="glue",
            dataset_config_name="sst2",
            split="validation",
            num_examples=100,
            accuracy=0.9,
        )

        with patch(
            "src.workflows.train_tune.bert_eval.bert_activities.BertEvalActivities._evaluate_bert_model_sync",
            return_value=expected_result,
        ) as mock_sync:
            result = await env.run(BertEvalActivities.evaluate_bert_model, request)

        assert result == expected_result
        mock_sync.assert_called_once_with(request)
