# CIFAR-10 Ray Scale-Up (Temporal + Ray + PyTorch)

This package demonstrates how to use Temporal to orchestrate **scaling experiments** for CIFAR-10 training using **Ray** and **PyTorch**.

Where the BERT examples focus on NLP models, this folder shows how the same Temporal patterns apply to **vision workloads** and **cluster-scale training**.

---

## What this package demonstrates

- **Long-running, compute-heavy activities** (`train_cifar10_with_ray`) coordinated by lightweight workflows.
- **Scale sweeps** over different Ray configurations:
  - Number of workers.
  - Batch size.
  - Number of epochs.
  - GPUs per worker.
- **Separation of concerns**:
  - Workflow (`Cifar10ScalingWorkflow`) is deterministic and only orchestrates runs.
  - Activity (`train_cifar10_with_ray`) owns all Ray, dataset, and training logic.
- Integration with a **local or remote Ray cluster**, including a helper script to start a local head node.

---

## Quickstart

From the project root (`temporal_training/`):

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start (or connect to) a Ray cluster**:

   - For a local head node:

     ```bash
     uv run -m src.workflows.train_tune.cifar10_scaleup.local_ray_cluster
     ```

   - Export the Ray address (if not already set):

     ```bash
     export RAY_ADDRESS="127.0.0.1:6379"
     ```

3. **Start the CIFAR-10 scaling worker**:

   ```bash
   uv sync --dev
   uv run -m src.workflows.train_tune.cifar10_scaleup.worker
   ```

   This worker:

   - Connects to Temporal using the Pydantic data converter.
   - Polls the `cifar10-ray-task-queue`.
   - Registers:
     - `Cifar10ScalingWorkflow`.
     - `train_cifar10_with_ray` activity.

4. **Kick off a scaling experiment workflow** (optional helper):

   The `cifar10_workflow.py` module includes a `main()` helper you can run directly:

   ```bash
   uv run -m src.workflows.train_tune.cifar10_scaleup.cifar10_workflow
   ```

   This:

   - Builds a `Cifar10ScalingInput` with several `RayScaleConfig` entries (workers/epochs/batch size).
   - Executes `Cifar10ScalingWorkflow` on `cifar10-ray-task-queue`.
   - Prints a summary table of runs (`run_id`, number of workers, epochs, accuracy, wall-clock time).

   You can also start workflows from your own scripts using the same models.

---

## Durability demo

To exercise Temporal’s durability with a long-running CIFAR-10 training job:

1. Start Temporal, a Ray cluster, and the CIFAR-10 worker as above.
2. Use the `cifar10_workflow` main helper (or your own client) to start a `Cifar10ScalingWorkflow` that includes at least one configuration with:
   - `num_epochs` large enough to run for a few minutes.
3. Wait until you see logs from `train_cifar10_with_ray` showing active training.
4. **Kill the worker process** (Ctrl‑C).
5. Restart the worker with the same command:

   ```bash
   uv run -m src.workflows.train_tune.cifar10_scaleup.worker
   ```

6. Observe in Temporal Web and logs that:
   - The workflow resumes from its last recorded state.
   - The long-running `train_cifar10_with_ray` activity is retried according to Temporal’s retry policy.
   - Final `Cifar10TrainResult` objects are produced, and the experiment summary still reflects all requested scales.

Because all Ray and training logic lives in an activity offloaded with `asyncio.to_thread`, the workflow itself remains deterministic and safe to replay.

---

## Why Temporal (for this example)

This CIFAR-10 scale-up example is designed to highlight Temporal’s strengths for **GPU-heavy, cluster-scale training**:

- **Durable orchestration of long-running training**:
  - Training activities can run for minutes or hours.
  - Worker crashes or restarts do not lose workflow progress.
- **Code-first scaling experiments**:
  - Scaling logic is expressed as a workflow that iterates over `RayScaleConfig` entries.
  - You can easily add new scales, change epochs, or adjust batch sizes without changing any Ray wiring.
- **Clean separation of orchestration and execution**:
  - Temporal workflows stay lightweight and deterministic.
  - Ray is used purely inside activities, which can be swapped for mocks in tests.
- **Incremental path to clusters**:
  - The same workflow and activity code runs:
    - On a laptop with a local Ray head.
    - On a multi-node Ray cluster (on-prem or cloud) by changing `RAY_ADDRESS`.

Compared to ad-hoc scripts or pure Ray drivers, Temporal adds a **durable, replayable control plane** that tracks each scale configuration as a first-class workflow run.

---

## Repo map (local to this folder)

- `cifar10_workflow.py` – Temporal workflow:
  - `Cifar10ScalingWorkflow` orchestrates a list of `RayScaleConfig` entries by calling `train_cifar10_with_ray` once per scale and aggregating results.
  - `Cifar10ScalingInput` / `Cifar10ScalingOutput` Pydantic models describe experiment input/output.
- `cifar10_activities.py` – Activity and data models:
  - `RayScaleConfig` – how many workers, epochs, batch size, GPUs per worker.
  - `Cifar10TrainRequest` / `Cifar10TrainResult` – input/output for training.
  - `_train_cifar10_with_ray_sync` – synchronous implementation of Ray + PyTorch training.
  - `train_cifar10_with_ray` – Temporal activity wrapper that offloads the sync helper to a thread and logs progress.
- `worker.py` – Temporal worker that connects to `localhost:7233`, uses the Pydantic data converter, and runs:
  - `Cifar10ScalingWorkflow`.
  - `train_cifar10_with_ray` activity.
- `local_ray_cluster.py` – Utility script for starting a local Ray head node with a CLI (`--port`, `--dashboard-port`).
- `tests/` – Tests for:
  - `cifar10_activities.py` (unit-level, where present).
  - `cifar10_workflow.py` (integration-style, using mocked activities and Temporal’s test environment).

For BERT-focused examples and the broader `train_tune` portfolio, see `src/workflows/train_tune/PORTFOLIO.md`.

