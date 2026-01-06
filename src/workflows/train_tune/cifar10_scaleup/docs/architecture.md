# CIFAR-10 Ray Scale-Up – Architecture

This document explains how the `cifar10_scaleup` package applies the same Temporal patterns used in the BERT examples to a **vision** workload with **Ray-based data-parallel training**.

---

## Components

### Workflows

- `Cifar10ScalingWorkflow` (`cifar10_workflow.py`)
  - Input: `Cifar10ScalingInput`
    - `experiment_name`: human-readable label for this scaling experiment.
    - `scales`: list of `RayScaleConfig` entries (workers, epochs, batch size, GPUs per worker).
    - `dataset_root`: optional shared dataset cache root.
    - `use_gpu`: whether to request GPU devices for training.
    - `random_seed`: base seed forwarded into Ray workers.
  - Responsibilities:
    - Log experiment metadata and number of scales.
    - For each `RayScaleConfig`:
      - Construct a `Cifar10TrainRequest` with a unique `run_id`.
      - Call `train_cifar10_with_ray` as an activity with a 2-hour timeout.
      - Collect the resulting `Cifar10TrainResult`.
    - Return `Cifar10ScalingOutput` with `experiment_name` and a list of per-scale results.
  - Output: `Cifar10ScalingOutput`.

The workflow is intentionally **simple and deterministic**: it does not talk to Ray directly and makes no network calls. All non-deterministic work is delegated to activities.

### Activities

- `train_cifar10_with_ray` (`cifar10_activities.py`)
  - Public activity definition:
    - Logs the start and end of the training run.
    - Offloads `_train_cifar10_with_ray_sync` to a background thread using `asyncio.to_thread`.
  - `_train_cifar10_with_ray_sync`:
    - Imports and configures:
      - Ray (cluster connection, resource configuration).
      - PyTorch / Torchvision (model, dataset, transforms).
    - Loads CIFAR-10:
      - Caches the dataset at `dataset_root` (or `~/data` by default).
      - Splits the training set into train/validation (80/20) for a smaller, fast demo.
    - Defines `SimpleCifarNet`, a compact CNN suitable for CIFAR-10:
      - Two conv layers + pooling.
      - A small MLP head.
    - Chooses device and Ray GPU resources:
      - CUDA or MPS if available and `use_gpu=True`.
      - CPU-only fallback if no accelerators are available.
    - Implements a `Trainer` Ray actor:
      - Each actor holds a copy of the model and trains on a shard of the dataset.
      - Epoch loop:
        - Broadcasts the current state dict from worker 0 to all workers.
        - Each worker runs a `train_epoch` over its shard.
    - Evaluates on the test set using the reference model from worker 0.
    - Computes:
      - `test_accuracy`.
      - `training_time_seconds`.
      - `num_parameters`.
      - `effective_num_workers`.
    - Shuts down Ray and returns `Cifar10TrainResult`.

---

## State model

### Workflow state

`Cifar10ScalingWorkflow` maintains only local variables inside its `run` method:

- A list of `Cifar10TrainResult` values, one per scale.
- A derived `run_id` per scale, based on `experiment_name` and index.

Workflow state is small and fully reconstructible from history, which keeps workflow histories compact and replay-friendly.

### Activity-level state

State for training is held entirely inside:

- The Ray cluster (actors, processes).
- The `SimpleCifarNet` model weights.
- The CIFAR-10 dataset loaded from disk.

All of this state is **outside** the workflow, which aligns with Temporal best practices: workflows orchestrate, activities execute side effects.

---

## Determinism

- Workflows:
  - Do not import Ray, PyTorch, or Torchvision.
  - Only construct Pydantic models and call activities.
  - Are safe to replay in Temporal for debugging or inspection.

- Activities:
  - Contain all non-deterministic behavior:
    - Dataset I/O.
    - Network interaction with Ray.
    - Random seeds (inside PyTorch and Ray workers).
  - Are guarded by a clear `RAY_IMPORT_MESSAGE` for environments where Ray/torch are not installed, keeping workers healthy and errors explicit.

As a result, deterministic replay is preserved at the workflow layer, even though training is stochastic and relies on external systems.

---

## Timeouts, retries, and idempotency

- Activity options:
  - `train_cifar10_with_ray` is invoked with:
    - `start_to_close_timeout=timedelta(hours=2)`.
  - For long experiments, you can:
    - Tune the timeout based on expected training time.
    - Configure retry policies in the workflow options or at the client level.

- Idempotency:
  - **Business key** for the training run is `run_id` in `Cifar10TrainRequest`.
  - Retries of `train_cifar10_with_ray` correspond to re-running the training experiment with the same Ray configuration.
  - For demo purposes, this is acceptable; in production, you might:
    - Persist intermediate checkpoints and metrics.
    - Make retries resume from the latest checkpoint rather than starting over.

---

## Backpressure and scaling

- **Task queue**:
  - `cifar10-ray-task-queue`:
    - Hosts `Cifar10ScalingWorkflow` and `train_cifar10_with_ray` in `worker.py`.
  - You can run multiple workers on this queue to increase throughput or isolate experiments.

- **Worker concurrency**:
  - `worker.py` currently uses a default `Worker` setup without an explicit `activity_executor`.
  - For heavier workloads, you can:
    - Configure a `ThreadPoolExecutor` and `max_concurrent_activities`.
    - Run separate workers for orchestration vs heavy training activities.

- **Ray scaling**:
  - Horizontal scaling of training is handled by Ray via:
    - `num_workers` (actors).
    - `num_gpus_per_worker`.
  - Temporal sees Ray as a black box: it simply triggers the activity with the desired configuration and records the resulting metrics.

---

## Failure modes and behavior

- **Ray or training failure**:
  - If Ray fails to start or training crashes:
    - `_train_cifar10_with_ray_sync` raises an exception.
    - Temporal marks the activity as failed and can retry it according to your policies.

- **Ray not installed**:
  - Import errors for Ray/torch/torchvision result in a `RuntimeError` with `RAY_IMPORT_MESSAGE`.
  - This keeps the worker process healthy while clearly signaling misconfiguration.

- **Worker crash / restart**:
  - In-flight `train_cifar10_with_ray` activities may be retried on a restarted worker.
  - The workflow state (which scales have completed) is preserved and can be inspected in Temporal Web.

---

## Production path

To move from this demo toward a production-scale CIFAR-10 (or similar) system:

- **Temporal**:
  - Deploy Temporal Server (self-hosted or Temporal Cloud).
  - Run multiple workers for:
    - Orchestration.
    - Training activities on GPU-capable nodes.

- **Ray**:
  - Deploy a multi-node Ray cluster (Kubernetes, managed Ray, or on-prem).
  - Use `RAY_ADDRESS` in the environment to point activities at the cluster.

- **Experiment tracking**:
  - Persist `Cifar10TrainResult` metadata and Ray run IDs to a database or tracking system (e.g., MLflow, Weights & Biases).
  - Enrich `Cifar10TrainResult` with dataset and code version information.

With these additions, `cifar10_scaleup` becomes a robust pattern for scaling vision models using Temporal and Ray, consistent with the rest of the `train_tune` portfolio.

