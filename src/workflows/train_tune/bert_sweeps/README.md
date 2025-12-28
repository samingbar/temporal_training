# BERT Hyperparameter Sweeps (Temporal + Transformers)

This folder contains a self-contained demo that uses Temporal to orchestrate
checkpointed BERT fine-tuning, evaluation, and hyperparameter sweeps on top of
Hugging Face Transformers and Datasets.

At a high level:

- **Activities** in `bert_activities.py` own all side effects: dataset loading,
  tokenization, model training, checkpointing, and evaluation.
- **Workflows** in `workflows.py` remain deterministic and focus purely on
  orchestration:
  - `CheckpointedBertTrainingWorkflow` runs a single fine-tuning job and emits
    checkpoints.
  - `BertEvalWorkflow` evaluates a fine-tuned checkpoint on a public dataset.
  - `CoordinatorWorkflow` wires training + evaluation together.
  - `SweepWorkflow` and `LadderSweepWorkflow` run hyperparameter sweeps.
- **Workers** in `worker.py` and `training_worker.py` split orchestration and
  heavy training onto separate task queues.
- **`starter.py`** is a tiny CLI entrypoint that kicks off a ladder-style sweep
  and prints a concise summary of results.

## Prerequisites

- A running Temporal server (for local development, `temporal server start-dev`)
- Project dependencies installed:

  ```bash
  uv sync --dev
  ```

  The root `pyproject.toml` includes the ML dependencies (`transformers`,
  `datasets`, `torch`) required for this example.

- Optional but recommended: a GPU or Apple Silicon/MPS device for faster
  training. The example is configured to run on CPU as well, just more slowly.

## Files at a Glance

- `custom_types.py` – Pydantic models shared between workflows, activities,
  and clients (training configs, eval configs, sweep requests/results, etc.).
- `bert_activities.py` – Activities for:
  - Creating reproducible dataset snapshots.
  - Running checkpoint-aware BERT fine-tuning.
  - Evaluating fine-tuned checkpoints on public datasets.
- `workflows.py` – Temporal workflows for:
  - Single-run training/eval orchestration.
  - Random hyperparameter sweeps (`SweepWorkflow`).
  - Ladder-style sweeps with a simple TPE-inspired sampler
    (`LadderSweepWorkflow`).
- `worker.py` – Worker hosting evaluation and sweep workflows on the
  `bert-eval-task-queue`.
- `training_worker.py` – Worker hosting training activities on the
  `bert-training-task-queue`, suitable for GPU machines.
- `starter.py` – CLI script that builds a `SweepRequest`, runs
  `LadderSweepWorkflow`, and prints a table of results.

## Running the Demo (Ladder Sweep)

These steps assume you are in the project root (`temporal_training/`).

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start the training worker** (ideally on a machine with a GPU):

   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.training_worker
   ```

3. **Start the evaluation/sweep worker** (CPU-only is fine):

   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.worker
   ```

4. **Run the ladder sweep starter** in a third terminal:

   ```bash
   uv run -m src.workflows.train_tune.bert_sweeps.starter
   ```

   This:

   - Connects to the Temporal server using the Pydantic data converter.
   - Uses the sample `ladder_config_1` defined in `starter.py`, which targets
     the SciBERT + SciCite combination by default.
   - Starts a `LadderSweepWorkflow` execution on the `bert-eval-task-queue`.
   - Prints a small table summarizing each evaluated run (dataset, split,
     number of examples, accuracy).

5. **Inspect checkpoints and snapshots**:

   - Fine-tuned models and tokenizers are written under `./bert_runs/{run_id}`.
   - Dataset snapshots (if enabled) are written under `./data_snapshots`.

## Customizing the Sweep

- Edit `ladder_config_1` in `starter.py` to:
  - Switch to a different base model or dataset.
  - Change the search space (`SweepSpace`) for learning rate, batch size,
    number of epochs, and sequence length.
  - Adjust `num_trials` and `max_concurrency` for your hardware.
- Tweak the `stages` list in `LadderSweepWorkflow.run` (see `workflows.py`) to
  change how many rungs the ladder uses and how much data/epochs each rung
  sees.
- For a simpler baseline, you can wire up `SweepWorkflow` instead of
  `LadderSweepWorkflow` to run a purely random sweep.

Because all randomness flows through Temporal's deterministic RNG, you can
re-run the same sweep with the same `SweepRequest.seed` and expect identical
trial configurations and ordering, which makes this a good template for
reproducible experimentation.

