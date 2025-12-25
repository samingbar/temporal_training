"""CLI-style entrypoint for running BERT inference via Temporal.

This script mirrors ``train.py`` but exercises ``BertInferenceWorkflow``
instead. It demonstrates how to:

1. Connect a Temporal client to the local test server.
2. Build a ``BertInferenceRequest`` that references a fine-tuned run.
3. Invoke ``BertInferenceWorkflow`` to perform batch inference.
4. Print the predicted labels and confidences for each input text.
"""

import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_finetune.bert_activities import BertInferenceRequest
from src.workflows.train_tune.bert_finetune.bert_workflow import BertInferenceWorkflow


async def main() -> None:
    """Execute a sample BERT inference workflow against a local Temporal server."""
    # 1. Connect to Temporal using the same data converter as the training demo.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Point at a specific fine-tuned run and provide a batch of texts to score.
    request = BertInferenceRequest(
        run_id="bert-finetune-demo-run-0-bert-base-uncased",
        texts=[
            "This movie was great!",
            "I thought it was just okay.",
            "This was a terrible experience.",
            "I didn't see the movie!",
            "The movie is awfully badass.",
        ],
        max_seq_length=128,
        use_gpu=True,
    )

    # 3. Execute the inference workflow and wait for the aggregated result.
    result = await client.execute_workflow(
        BertInferenceWorkflow.run,
        request,
        id="bert-inference-demo-id",
        task_queue="bert-finetune-task-queue",
    )

    # 4. Print a compact report for each input text.
    for text, label, score in zip(
        result.texts,
        result.predicted_labels,
        result.confidences,
        strict=True,
    ):
        print(f"{text!r} -> label={label}, confidence={score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
