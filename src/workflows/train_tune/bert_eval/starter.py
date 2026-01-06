"""CLI entrypoint for evaluating a fine-tuned BERT model on a public dataset.

This script drives the ``BertEvalWorkflow`` in the ``bert_eval`` package, which:

- Loads a fine-tuned checkpoint from ``./bert_runs/{run_id}``.
- Evaluates it on a Hugging Face dataset split (GLUE SST-2 by default).
"""

import asyncio

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_eval.workflows import CoordinatorWorkflow
from src.workflows.train_tune.bert_eval.custom_types import (
    BertEvalRequest,
    BertFineTuneConfig,
    CoordinatorWorkflowConfig,
    CoordinatorWorkflowInput,
)


async def main() -> None:
    """Execute a sample BERT evaluation workflow."""
    # 1. Connect to the Temporal server using the Pydantic data converter so
    #    that our Pydantic models can be sent over the wire transparently.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Describe an evaluation configuration. By default this will:
    #    - Load GLUE SST-2.
    #    - Evaluate the model on the validation split with a capped number
    #      of examples for a fast, tutorial-friendly run.

    request = CoordinatorWorkflowInput(
        configs=[
            CoordinatorWorkflowConfig(
                fine_tune_config=BertFineTuneConfig(
                    model_name="bert-base-uncased",
                    dataset_name="glue",
                    dataset_config_name="sst2",
                    num_epochs=2,
                    batch_size=32,
                    learning_rate=2e-5,
                    max_seq_length=128,
                    use_gpu=True,
                    max_train_samples=3_000,
                    max_eval_samples=2_000,
                ),
                evaluation_config=BertEvalRequest(
                    dataset_name="glue",
                    dataset_config_name="sst2",
                    split="validation",
                    max_eval_samples=1_000,
                    max_seq_length=128,
                    batch_size=32,
                    use_gpu=True,
                ),
            ),
        ],
    )
    # 3. Start the evaluation workflow and wait for the result.
    results = await client.execute_workflow(
        CoordinatorWorkflow.run,
        request,
        id="bert-end2end-demo",
        task_queue="bert-eval-task-queue",
    )

    # 4. Print a concise summary of the first evaluation result.
    result = results[0]
    print("\nBERT evaluation result:")
    print(
        f"- run_id={result.run_id}, "
        f"dataset={result.dataset_name}/{result.dataset_config_name}, "
        f"split={result.split}, "
        f"num_examples={result.num_examples}, "
        f"accuracy={result.accuracy:.3f}",
    )


if __name__ == "__main__":
    asyncio.run(main())
