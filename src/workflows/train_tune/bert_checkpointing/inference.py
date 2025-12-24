"""CLI entrypoint for running inference against checkpointed BERT runs."""

import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_checkpointing.checkpointed_training import (
    BertInferenceWorkflow,
)
from src.workflows.train_tune.bert_checkpointing.custom_types import BertInferenceRequest


async def main() -> None:
    """Execute a sample BERT inference workflow against a local Temporal server."""

    # 1. Connect to Temporal using the same data converter as the training demo.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Point at a specific fine-tuned run and provide a batch of texts to score.
    #    Typically this run_id will match the workflow ID used for training.
    request = BertInferenceRequest(
        run_id="bert-checkpointed-training-demo-id",
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
        id="bert-checkpointed-inference-demo-id",
        task_queue="bert-checkpointing-task-queue",
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
