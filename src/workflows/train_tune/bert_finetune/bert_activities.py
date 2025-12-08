"""Activities for BERT fine-tuning using Hugging Face Transformers."""

import asyncio
import time
from typing import Final

from pydantic import BaseModel, Field
from temporalio import activity


class BertFineTuneConfig(BaseModel):
    """Configuration for a single BERT fine-tuning run."""

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


class BertFineTuneRequest(BaseModel):
    """Input to a BERT fine-tuning activity run."""

    run_id: str
    """Logical identifier for this fine-tuning run."""

    config: BertFineTuneConfig
    """Configuration for the fine-tuning experiment."""


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


TRANSFORMERS_IMPORT_MESSAGE: Final[str] = (
    "BERT fine-tuning dependencies are not installed. "
    "Install 'transformers', 'datasets', and 'torch' to execute this activity."
)


def _fine_tune_bert_sync(request: BertFineTuneRequest) -> BertFineTuneResult:
    """Run a BERT fine-tuning job with the requested configuration.

    This helper is synchronous and CPU/GPU bound; the async activity delegates
    to it via ``asyncio.to_thread`` so that Temporal's worker can continue
    polling for other tasks while the training loop runs.
    """
    try:
        import torch
        from datasets import load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - only hit when deps missing
        raise RuntimeError(TRANSFORMERS_IMPORT_MESSAGE) from exc

    start_time = time.perf_counter()

    config = request.config

    # Device selection with graceful CPU/MPS/CUDA fallback.
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load dataset and model/tokenizer from Hugging Face Hub.
    raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_function(batch: dict) -> dict:
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )
    model.to(device)

    # Standard Trainer setup. Transformers 4.57.1 uses `eval_strategy` instead
    # of the older `evaluation_strategy` argument.
    training_args = TrainingArguments(
        output_dir=f"./bert_runs/{request.run_id}",
        num_train_epochs=float(config.num_epochs),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        report_to=[],
        load_best_model_at_end=False,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation") or tokenized_datasets.get("validation_matched")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        accuracy = (predictions == labels).astype("float32").mean().item()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
    )

    train_result = trainer.train()
    metrics = {}
    if eval_dataset is not None:
        metrics = trainer.evaluate()

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_time_seconds = float(time.perf_counter() - start_time)

    return BertFineTuneResult(
        run_id=request.run_id,
        config=config,
        train_loss=float(train_result.training_loss),
        eval_accuracy=float(metrics.get("eval_accuracy")) if "eval_accuracy" in metrics else None,
        training_time_seconds=training_time_seconds,
        num_parameters=num_parameters,
    )


@activity.defn
async def fine_tune_bert(request: BertFineTuneRequest) -> BertFineTuneResult:
    """Temporal activity that runs a BERT fine-tuning job."""
    activity.logger.info(
        "Starting BERT fine-tuning run %s with model %s on %s/%s",
        request.run_id,
        request.config.model_name,
        request.config.dataset_name,
        request.config.dataset_config_name,
    )
    result = await asyncio.to_thread(_fine_tune_bert_sync, request) #Offload the training to a separate thread. In the future, move to a separate cluster?
    activity.logger.info(
        "Completed BERT fine-tuning run %s with loss %.4f and accuracy %s",
        request.run_id,
        result.train_loss,
        "N/A" if result.eval_accuracy is None else f"{result.eval_accuracy:.3f}",
    )
    return result
