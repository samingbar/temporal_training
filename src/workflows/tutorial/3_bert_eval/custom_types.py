"""Shared data models for the BERT fine-tuning and inference example.

These Pydantic models are used in three places:

- As workflow inputs/outputs (e.g., ``BertExperimentInput``)
- As activity inputs/outputs (e.g., ``BertFineTuneRequest``)
- From external clients (see ``train.py`` and ``inference.py``) that submit
  workflows to Temporal.

Keeping the types in a dedicated module makes it easy to reuse them across
workflows, activities, and client code while preserving a single source of
truth for field names and validation rules.
"""

from typing import Final  # noqa: F401  - kept for parity with other modules.

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Checkpointing - related types
# ---------------------------------------------------------------------------
class DatasetSnapshotRequest(BaseModel):
    """Request to create a dataset snapshot."""

    run_id: str = Field(description="Identifier for the training run requesting this snapshot.")

    dataset_name: str = Field(description="HuggingFace dataset name (e.g., 'glue').")

    dataset_config: str = Field(description="Dataset configuration name (e.g., 'sst2').")

    max_samples: int | None = Field(
        default=None, description="Optional limit on number of samples to include."
    )

    snapshot_dir: str = Field(
        default="./data_snapshots", description="Directory to store snapshots."
    )


class DatasetSnapshotResult(BaseModel):
    """Result describing a materialized dataset snapshot.

    This model mirrors the information persisted alongside each snapshot on disk
    so that workflows and activities can reason about snapshot identity and size
    without re-reading the underlying data.
    """

    snapshot_id: str = Field(
        description=(
            "Stable identifier for the snapshot, typically derived from the "
            "dataset name, configuration, and a content hash."
        ),
    )
    dataset_name: str = Field(
        description="Hugging Face dataset name (e.g., 'glue').",
    )
    dataset_config: str = Field(
        description="Dataset configuration name (e.g., 'sst2').",
    )
    num_train_samples: int = Field(
        description="Number of training samples stored in the snapshot.",
    )
    num_eval_samples: int = Field(
        description="Number of evaluation samples stored in the snapshot.",
    )
    data_hash: str = Field(
        description="Short content hash used to detect identical snapshots.",
    )
    snapshot_timestamp: str = Field(
        description="ISO 8601 timestamp indicating when the snapshot was created.",
    )
    snapshot_path: str = Field(
        description="Filesystem path where the snapshot artifacts are stored.",
    )


class CheckpointInfo(BaseModel):
    """Checkpoint information sent via signals."""

    epoch: int = Field(description="Training epoch number")
    step: int = Field(description="Global training step")
    path: str = Field(description="Filesystem path to checkpoint")
    loss: float = Field(description="Training loss at this checkpoint")
    timestamp: str = Field(description="ISO timestamp when saved")


# ---------------------------------------------------------------------------
# Activity-level types (training and inference)
# ---------------------------------------------------------------------------


class BertFineTuneConfig(BaseModel):
    """Configuration for a single BERT fine-tuning run.

    This mirrors the configuration accepted by the fine-tuning activity and is
    deliberately lightweight so it can be serialized in workflow inputs and
    reused by external clients.
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="Hugging Face model identifier to fine-tune.",
    )
    dataset_name: str = Field(
        default="glue",
        description="Hugging Face dataset name (e.g. 'glue').",
    )
    dataset_config_name: str = Field(
        default="sst2",
        description="Dataset configuration name (e.g. 'sst2').",
    )
    num_epochs: int = Field(
        gt=0,
        le=20,
        default=3,
        description="Number of training epochs.",
    )
    batch_size: int = Field(
        gt=0,
        le=128,
        default=16,
        description="Per-device training batch size.",
    )
    learning_rate: float = Field(
        gt=0,
        le=1e-1,
        default=5e-5,
        description="Learning rate for the optimizer.",
    )
    max_seq_length: int = Field(
        gt=8,
        le=512,
        default=128,
        description="Maximum sequence length for tokenization.",
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU/MPS if available.",
    )

    max_train_samples: int | None = Field(
        default=2_000,
        description=(
            "Optional cap on the number of training examples to use. "
            "Set to None to train on the full dataset."
        ),
    )
    max_eval_samples: int | None = Field(
        default=1_000,
        description=(
            "Optional cap on the number of evaluation examples to use. "
            "Set to None to evaluate on the full validation set."
        ),
    )


class BertFineTuneRequest(BaseModel):
    """Input to a BERT fine-tuning activity run."""

    run_id: str
    """Logical identifier for this fine-tuning run."""

    config: BertFineTuneConfig
    """Configuration for the fine-tuning experiment."""

    # NEW: Pass the dataset snapshot to use for reproducible training
    dataset_snapshot: DatasetSnapshotResult | None = Field(
        default=None, description="Optional dataset snapshot to use for reproducible training."
    )

    # NEW: Allow resuming from checkpoint through act
    resume_from_checkpoint: str | None = Field(
        default=None, description="Path to checkpoint to resume from (if retrying)"
    )


class BertFineTuneResult(BaseModel):
    """Summary metrics from a BERT fine-tuning run."""

    run_id: str
    """Echoed identifier for the run."""

    config: BertFineTuneConfig
    """Configuration actually used for the run."""

    train_loss: float = Field(
        description="Final training loss reported by the Trainer.",
    )

    eval_accuracy: float | None = Field(
        default=None,
        description="Validation accuracy if available.",
    )

    training_time_seconds: float = Field(
        description="Wall-clock training time in seconds.",
    )

    num_parameters: int = Field(
        description="Total number of learnable parameters in the model.",
    )

    dataset_snapshot: DatasetSnapshotResult | None = Field(
        default=None, description="Dataset snapshot used for this training run (if any)."
    )

    total_checkpoints_saved: int = Field(
        description="Total number of checkpoints saved during training.",
    )


class BertInferenceRequest(BaseModel):
    """Input to a BERT inference activity run."""

    run_id: str = Field(
        description="Identifier of the fine-tuned run whose checkpoint to load.",
    )
    texts: list[str] = Field(
        min_length=1,
        description="List of input sentences to classify.",
    )
    max_seq_length: int = Field(
        gt=8,
        le=512,
        default=128,
        description="Maximum sequence length for tokenization during inference.",
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU/MPS for inference if available.",
    )


class BertInferenceResult(BaseModel):
    """Inference results for a batch of texts using a fine-tuned BERT model."""

    run_id: str
    """Identifier of the fine-tuned run used for inference."""

    texts: list[str]
    """Input texts that were classified."""

    predicted_labels: list[int] = Field(
        description="Predicted class indices for each input text.",
    )

    confidences: list[float] = Field(
        description="Confidence scores (max softmax probability) per prediction.",
    )


# ---------------------------------------------------------------------------
# Evaluation-level types
# ---------------------------------------------------------------------------


class BertEvalRequest(BaseModel):
    """Input configuration for evaluating a fine-tuned BERT model."""

    run_id: str = Field(
        description="Identifier of the fine-tuned run whose checkpoint directory to load.",
    )
    dataset_name: str = Field(
        default="glue",
        description="Hugging Face dataset name to evaluate on (e.g., 'glue').",
    )
    dataset_config_name: str = Field(
        default="sst2",
        description="Dataset configuration name (e.g., 'sst2').",
    )
    split: str = Field(
        default="validation",
        description="Dataset split to evaluate on (e.g., 'validation' or 'test').",
    )
    max_eval_samples: int | None = Field(
        default=1_000,
        description=(
            "Optional cap on the number of evaluation examples to use. "
            "Set to None to evaluate on the full split."
        ),
    )
    max_seq_length: int = Field(
        gt=8,
        le=512,
        default=128,
        description="Maximum sequence length for tokenization during evaluation.",
    )
    batch_size: int = Field(
        gt=0,
        le=256,
        default=32,
        description="Batch size to use during evaluation.",
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU/MPS for evaluation if available.",
    )


class BertEvalResult(BaseModel):
    """Aggregate metrics from evaluating a fine-tuned BERT model."""

    run_id: str = Field(
        description="Identifier of the fine-tuned run whose checkpoint was evaluated.",
    )
    dataset_name: str = Field(
        description="Hugging Face dataset name used for evaluation.",
    )
    dataset_config_name: str = Field(
        description="Dataset configuration name used for evaluation.",
    )
    split: str = Field(
        description="Dataset split that was evaluated (e.g., 'validation').",
    )
    num_examples: int = Field(
        ge=0,
        description="Number of examples included in the evaluation.",
    )
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Classification accuracy over the evaluated examples.",
    )


# ---------------------------------------------------------------------------
# Workflow-level types
# ---------------------------------------------------------------------------


class BertExperimentInput(BaseModel):
    """Input to the BERT fine-tuning workflow.

    The workflow fans out one fine-tuning activity per entry in ``runs`` and
    aggregates the resulting :class:`BertFineTuneResult` values.
    """

    experiment_name: str
    """Human-readable label for this experiment."""

    runs: list[BertFineTuneConfig]
    """List of fine-tuning configurations to execute."""


class BertExperimentOutput(BaseModel):
    """Summary of the BERT fine-tuning experiment."""

    experiment_name: str
    """Echoed experiment name."""

    runs: list[BertFineTuneResult]
    """Per-configuration fine-tuning results."""
