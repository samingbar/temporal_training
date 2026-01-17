"""Temporal activities for BERT fine-tuning and inference.

This module contains the *non-deterministic* parts of the BERT example:

- Long-running, compute-heavy fine-tuning using Hugging Face Transformers
- Loading a saved checkpoint and running batch inference

The corresponding Temporal workflows orchestrate these activities, but all of the
actual ML logic (dataset loading, tokenization, model forward passes, etc.) lives
here so that workflow code can remain deterministic and replay-safe.
"""

import asyncio
import contextlib
import time
from typing import Final

from temporalio import activity

from src.workflows.train_tune.bert_finetune.custom_types import (
    BertFineTuneRequest,
    BertFineTuneResult,
    BertInferenceRequest,
    BertInferenceResult,
)

# Human-friendly error message surfaced when ML dependencies are missing. This keeps
# the Temporal worker process healthy even if the Python environment is not configured
# for running the BERT example.
TRANSFORMERS_IMPORT_MESSAGE: Final[str] = (
    "BERT fine-tuning dependencies are not installed. "
    "Install 'transformers', 'datasets', and 'torch' to execute this activity."
)

# How frequently the fine-tuning activity should send heartbeats while training is
# running in a background thread. A short interval keeps the example responsive in
# tests (each of which has a 5s timeout) while still being perfectly acceptable in
# development settings.
HEARTBEAT_INTERVAL_SECONDS: Final[float] = 1.0


def _fine_tune_bert_sync(request: BertFineTuneRequest) -> BertFineTuneResult:
    """Run a BERT fine-tuning job with the requested configuration.

    The goal of this function is to encapsulate *all* ML details in a pure
    Python helper that knows nothing about Temporal. The async activity wrapper
    offloads to this helper in a thread so that:

    - The code can be imported and unit-tested without a Temporal worker.
    - The Temporal worker can keep polling for new tasks while training runs.
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

    # -------------------------------------------------------------------------
    # 1. Choose an appropriate device (CUDA, Apple MPS, or CPU).
    # -------------------------------------------------------------------------
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # -------------------------------------------------------------------------
    # 2. Load the dataset and tokenizer from the Hugging Face Hub.
    #    For the demo we use GLUE/SST-2, which exposes a "sentence" field and
    #    a "label" that we rename to "labels" for the Trainer.
    # -------------------------------------------------------------------------
    raw_datasets = load_dataset(config.dataset_name, config.dataset_config_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Apply the tokenizer across the dataset; `batched=True` lets HF process
    # multiple rows at once for better throughput. We mirror the checkpointed
    # example's behavior and assume a ``sentence`` column by default.
    def tokenize_function(batch: dict) -> dict:
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
        )

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    # Tell Datasets to yield PyTorch tensors for the columns the Trainer needs.
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation") or tokenized_datasets.get(
        "validation_matched"
    )

    # -------------------------------------------------------------------------
    # 3. Optionally sub-sample train/eval for a fast demo run on laptops.
    #    This keeps the example runnable on a high-end MacBook within minutes
    #    while still exercising the full training pipeline.
    # -------------------------------------------------------------------------
    if config.max_train_samples is not None and config.max_train_samples < len(train_dataset):
        train_dataset = train_dataset.select(range(config.max_train_samples))
    if (
        eval_dataset is not None
        and config.max_eval_samples is not None
        and config.max_eval_samples < len(eval_dataset)
    ):
        eval_dataset = eval_dataset.select(range(config.max_eval_samples))

    # -------------------------------------------------------------------------
    # 4. Construct the classification head on top of the base encoder.
    #    AutoModelForSequenceClassification adds a small task-specific head
    #    while keeping the rest of the BERT architecture intact.
    # -------------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
    )
    model.to(device)

    # -------------------------------------------------------------------------
    # 5. Configure the Transformers Trainer.
    #
    # The Trainer owns the training loop, evaluation, and logging. The choice
    # of hyperparameters here is intentionally simple and tuned for readability
    # over state-of-the-art results.
    #
    # Note: Transformers 4.57.1 uses `eval_strategy` instead of the older
    # `evaluation_strategy` argument.
    # -------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=f"./bert_runs/{request.run_id}",
        num_train_epochs=float(config.num_epochs),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="no",
        logging_strategy="epoch",
        logging_dir=f"./bert_runs/{request.run_id}/tb",
        report_to=["tensorboard"],
        load_best_model_at_end=False,
    )

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

    # -------------------------------------------------------------------------
    # 6. Run training (and optional evaluation) and collect summary metrics.
    # -------------------------------------------------------------------------
    train_result = trainer.train()
    metrics = {}
    if eval_dataset is not None:
        metrics = trainer.evaluate()

    # Persist the fine-tuned model and tokenizer so that the inference activity
    # can load them later based solely on ``run_id``.
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

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


def _run_bert_inference_sync(request: BertInferenceRequest) -> BertInferenceResult:
    """Run batch inference using a fine-tuned BERT model checkpoint.

    As with ``_fine_tune_bert_sync``, this helper is intentionally free of any
    Temporal-specific APIs. It simply loads a saved model and tokenizer from
    disk and runs a forward pass over a batch of texts.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - only hit when deps missing
        raise RuntimeError(TRANSFORMERS_IMPORT_MESSAGE) from exc

    # Device selection mirroring the training configuration.
    if request.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif request.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Each training run writes its artifacts into a run-specific directory under
    # ``./bert_runs``. The ``run_id`` flowing through the workflow and activity
    # input acts as the glue between training and inference.
    model_dir = f"./bert_runs/{request.run_id}"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Tokenize the input texts into a padded batch of tensors that the model
    # can consume. We keep the interface high-level on purpose so that callers
    # do not have to manage token IDs directly.
    encoded = tokenizer(
        request.texts,
        padding=True,
        truncation=True,
        max_length=request.max_seq_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Standard PyTorch inference boilerplate: no gradients, softmax over logits,
    # and then take the argmax and associated probability for each example.
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        confidences, predicted = probs.max(dim=-1)

    predicted_labels = predicted.tolist()
    confidence_scores = confidences.tolist()

    return BertInferenceResult(
        run_id=request.run_id,
        texts=list(request.texts),
        predicted_labels=predicted_labels,
        confidences=confidence_scores,
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

    # Offload the training to a separate thread and send periodic heartbeats
    # so Temporal can detect liveness during long-running fine-tuning.
    #
    # This pattern lets us:
    # - Keep the heavy ML work off the event loop thread, and
    # - Give Temporal visibility into progress via heartbeats, which in turn
    #   enables heartbeat timeouts and cancellation handling.
    training_task = asyncio.create_task(asyncio.to_thread(_fine_tune_bert_sync, request))
    try:
        while not training_task.done():
            activity.heartbeat({"run_id": request.run_id})
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
        result = await training_task
    except asyncio.CancelledError:  # pragma: no cover - cancellation path
        training_task.cancel()
        with contextlib.suppress(Exception):
            await training_task
        raise

    activity.logger.info(
        "Completed BERT fine-tuning run %s with loss %.4f and accuracy %s",
        request.run_id,
        result.train_loss,
        "N/A" if result.eval_accuracy is None else f"{result.eval_accuracy:.3f}",
    )
    return result


@activity.defn
async def run_bert_inference(request: BertInferenceRequest) -> BertInferenceResult:
    """Temporal activity that runs BERT inference using a fine-tuned checkpoint."""
    activity.logger.info(
        "Starting BERT inference for run %s on %s text(s)",
        request.run_id,
        len(request.texts),
    )
    result = await asyncio.to_thread(_run_bert_inference_sync, request)
    activity.logger.info(
        "Completed BERT inference for run %s",
        request.run_id,
    )
    return result
