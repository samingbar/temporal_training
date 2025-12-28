"""CLI entrypoint for evaluating a fine-tuned BERT model on a public dataset.

This script drives the ``BertEvalWorkflow`` in the ``bert_eval`` package, which:

- Loads a fine-tuned checkpoint from ``./bert_runs/{run_id}``.
- Evaluates it on a Hugging Face dataset split (GLUE SST-2 by default).
"""

import asyncio
import random

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.train_tune.bert_parallel.custom_types import (
    BertEvalRequest,
    BertFineTuneConfig,
    CoordinatorWorkflowConfig,
    CoordinatorWorkflowInput,
)
from src.workflows.train_tune.bert_parallel.workflows import (
    CoordinatorWorkflow,
)


async def main() -> None:
    """Execute a sample BERT evaluation workflow."""
    # 1. Connect to the Temporal server using the Pydantic data converter so
    #    that our Pydantic models can be sent over the wire transparently.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Describe the overall configuration.
    config_1 = CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="bert-base-uncased",
            dataset_name="glue",
            dataset_config_name="sst2",
            num_epochs=2,
            batch_size=8,
            learning_rate=2e-5,
            max_seq_length=128,
            use_gpu=True,
            max_train_samples=3_000,
            max_eval_samples=2_000,
            seed=random.randint(0, 10000),
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
    )

    config_2 = CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="bert-base-cased",
            dataset_name="glue",
            dataset_config_name="sst2",
            num_epochs=10,
            batch_size=16,
            learning_rate=3e-5,
            max_seq_length=128,
            use_gpu=True,
            max_train_samples=3_000,
            max_eval_samples=2_000,
            seed=random.randint(0, 10000),
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
    )

    config_3 = CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="bert-base-uncased",
            dataset_name="imdb",
            dataset_config_name="plain_text",
            num_epochs=10,
            batch_size=32,
            learning_rate=2e-5,
            max_seq_length=256,
            use_gpu=True,
            max_train_samples=5_000,
            max_eval_samples=2_000,
            seed=random.randint(0, 10000),
        ),
        evaluation_config=BertEvalRequest(
            dataset_name="imdb",
            dataset_config_name="plain_text",
            split="test",
            max_eval_samples=1_000,
            max_seq_length=256,
            batch_size=32,
            use_gpu=True,
        ),
    )
    #### distilbert SLM configuration for testing ####
    config_4 = CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="distilbert-base-uncased",
            dataset_name="glue",
            dataset_config_name="sst2",
            num_epochs=2,
            batch_size=4,  # MPS-safe
            learning_rate=5e-5,
            max_seq_length=64,  # MPS-safe
            use_gpu=True,
            max_train_samples=2_000,
            max_eval_samples=1_000,
            shuffle_before_select=True,
            seed=42,
            # optional overrides if you added them:
            # text_field=None,
            # text_pair_field=None,
            # label_field=None,
            # task_type="auto",
        ),
        evaluation_config=BertEvalRequest(
            # run_id will be filled in by the coordinator after training
            run_id=None,
            dataset_name="glue",
            dataset_config_name="sst2",
            split="validation",
            max_eval_samples=1_000,
            max_seq_length=64,
            batch_size=32,
            use_gpu=True,
            # if you changed eval to require model_uri/model_path, leave it unset here
            # and let the coordinator populate it from the training result.
            # model_path=None,
            # model_uri=None,
        ),
        dataset_snapshot=None,  # or pass a DatasetSnapshotResult if you want reproducibility
    )

    #### MiniLM-L12-H384-uncased SLM configuration for testing ####
    config_5 = CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="microsoft/MiniLM-L12-H384-uncased",
            dataset_name="glue",
            dataset_config_name="sst2",
            num_epochs=2,
            batch_size=4,  # still MPS-safe
            learning_rate=3e-5,  # MiniLM often prefers slightly lower LR
            max_seq_length=64,  # safe starting point
            use_gpu=True,
            max_train_samples=2_000,
            max_eval_samples=1_000,
            shuffle_before_select=True,
            seed=42,
            # Optional schema overrides (usually not needed for GLUE)
            # text_field=None,
            # text_pair_field=None,
            # label_field=None,
            # task_type="auto",
        ),
        evaluation_config=BertEvalRequest(
            # run_id filled in by coordinator
            run_id=None,
            dataset_name="glue",
            dataset_config_name="sst2",
            split="validation",
            max_eval_samples=1_000,
            max_seq_length=64,
            batch_size=32,
            use_gpu=True,
            # If your eval requires an explicit model path/URI,
            # leave it unset here and let the coordinator fill it.
            # model_path=None,
            # model_uri=None,
        ),
        dataset_snapshot=None,  # pass a snapshot if you want strict reproducibility
    )
    #### DeBERTa-v3-small configuration for testing ####
    config_6 = CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="microsoft/deberta-v3-small",
            dataset_name="glue",
            dataset_config_name="sst2",
            num_epochs=2,
            batch_size=2,  # DeBERTa tends to be heavier on memory (safer on MPS)
            learning_rate=2e-5,  # common stable starting LR for DeBERTa fine-tuning
            max_seq_length=64,  # start safe; bump to 96/128 once stable
            use_gpu=True,
            max_train_samples=2_000,
            max_eval_samples=1_000,
            shuffle_before_select=True,
            seed=42,
            # Optional schema overrides (usually not needed for GLUE)
            # text_field=None,
            # text_pair_field=None,
            # label_field=None,
            # task_type="auto",
        ),
        evaluation_config=BertEvalRequest(
            run_id=None,  # coordinator fills this in
            dataset_name="glue",
            dataset_config_name="sst2",
            split="validation",
            max_eval_samples=1_000,
            max_seq_length=64,
            batch_size=32,
            use_gpu=True,
            # model_path/model_uri left for coordinator to populate from training result
        ),
        dataset_snapshot=None,
    )

    config_7 = CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="allenai/scibert_scivocab_uncased",
            dataset_name="scicite",  # start with SST-2 to validate pipeline
            dataset_config_name="default",
            num_epochs=2,  # SciBERT converges quickly
            batch_size=4,  # heavier than BERT-base on MPS
            learning_rate=2e-5,  # common SciBERT fine-tuning LR
            max_seq_length=128,  # safe baseline (raise later)
            use_gpu=True,
            max_train_samples=2_000,
            max_eval_samples=1_000,
            shuffle_before_select=True,
            seed=42,
            # Optional schema overrides (usually unnecessary)
            # text_field=None,
            # text_pair_field=None,
            # label_field=None,
            # task_type="auto",
        ),
        evaluation_config=BertEvalRequest(
            run_id=None,  # coordinator fills this in
            dataset_name="scicite",
            dataset_config_name="default",
            split="validation",
            max_eval_samples=1_000,
            max_seq_length=128,
            batch_size=32,
            use_gpu=True,
            # model_path / model_uri populated by coordinator after training
        ),
        dataset_snapshot=None,  # add snapshot later for reproducibility
    )

    request = CoordinatorWorkflowInput(configs=[config_7])
    # 3. Start the evaluation workflow and wait for the result.
    result = await client.execute_workflow(
        CoordinatorWorkflow.run,
        request,
        id="bert-end2end-demo",
        task_queue="bert-eval-task-queue",
    )

    # 4. Print a concise, tabular summary of the evaluations.
    results = result if isinstance(result, (list, tuple)) else [result]

    print("\n=== BERT evaluation summary ===")
    header = f"{'run_id':<36} {'dataset':<20} {'split':<10} {'examples':>10} {'accuracy':>9}"
    print(header)
    print("-" * len(header))

    for item in results:
        dataset = f"{item.dataset_name}/{item.dataset_config_name}"
        print(
            f"{item.run_id:<36} "
            f"{dataset:<20} "
            f"{item.split:<10} "
            f"{item.num_examples:>10} "
            f"{item.accuracy:>9.3f}",
        )


if __name__ == "__main__":
    asyncio.run(main())
