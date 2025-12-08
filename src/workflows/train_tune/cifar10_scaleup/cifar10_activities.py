"""Activities for CIFAR-10 training using Ray."""

import asyncio
import time
from typing import Final

from pydantic import BaseModel, Field
from temporalio import activity


class RayScaleConfig(BaseModel):
    """Configuration for a single Ray training scale."""

    num_workers: int = Field(
        gt=0,
        description="Number of Ray workers to use for data-parallel training.",
    )
    num_epochs: int = Field(
        gt=0,
        le=500,
        description="Number of epochs to train for this run (capped for safety).",
    )
    batch_size: int = Field(
        gt=0,
        le=1024,
        description="Mini-batch size used by each worker.",
    )
    num_gpus_per_worker: float = Field(
        ge=0,
        le=8,
        description="GPUs reserved per worker (0 for CPU-only).",
    )
    description: str | None = Field(
        default=None,
        description="Optional human readable tag (e.g., 'baseline', '4x scale up').",
    )


class Cifar10TrainRequest(BaseModel):
    """Input to a single CIFAR-10 Ray training run."""

    run_id: str
    """Logical identifier for this training run."""

    scale: RayScaleConfig
    """Ray scale configuration for this run."""

    dataset_root: str | None = Field(
        default=None,
        description="Optional root directory for caching the CIFAR-10 dataset.",
    )

    use_gpu: bool = Field(
        default=False,
        description="Whether to use GPU devices if available.",
    )

    random_seed: int = Field(
        default=42,
        description="Random seed forwarded into the Ray workers.",
    )


class Cifar10TrainResult(BaseModel):
    """Summary metrics from a single training run."""

    run_id: str
    """Identifier echoed from the request."""

    scale: RayScaleConfig
    """Scale configuration actually used for this run."""

    test_accuracy: float = Field(
        description="Top-1 test-set accuracy from the final checkpoint (0.0–1.0).",
    )

    training_time_seconds: float = Field(
        description="End-to-end wall-clock time for the Ray job in seconds.",
    )

    num_parameters: int = Field(
        description="Total number of learnable parameters in the model.",
    )

    effective_num_workers: int = Field(
        description="Number of workers that successfully participated in training.",
    )


RAY_IMPORT_MESSAGE: Final[str] = (
    "CIFAR-10 training dependencies are not installed. "
    "Install 'ray[default]', 'torch', and 'torchvision' to execute this activity."
)


def _train_cifar10_with_ray_sync(request: Cifar10TrainRequest) -> Cifar10TrainResult:
    """Run a CIFAR-10 training job at the requested Ray scale.

    This helper is intentionally synchronous and CPU-bound. The async activity
    wrapper delegates to it via ``asyncio.to_thread`` so that Temporal can
    continue polling for other workflow tasks while the training job runs.

    The implementation uses a compact CNN and Ray for simple data-parallel
    training. It is optimized for readability rather than state-of-the-art
    performance so that it can serve as an educational example.
    """
    try:
        import os
        import statistics

        import ray
        import ray._private.ray_constants as ray_constants
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        from torch.utils.data import DataLoader, random_split
        from torchvision import datasets, transforms
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise RuntimeError(RAY_IMPORT_MESSAGE) from exc

    # Disable Ray's uv runtime environment hook when running under `uv run`.
    # The hook currently does not correctly handle `uv run -m ...` invocations,
    # which is how this worker is typically launched in this template.
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False

    start_time = time.perf_counter()

    # Initialize Ray; if a cluster is already running, this will connect to it.
    ray.init(ignore_reinit_error=True)

    dataset_root = request.dataset_root or os.path.expanduser("~/data")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ],
    )

    full_train_dataset = datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=dataset_root,
        train=False,
        download=True,
        transform=transform,
    )

    # Use a validation-style split to keep the example small and fast.
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size])

    # Simple CNN suitable for CIFAR-10 scale experiments.
    class SimpleCifarNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 256)
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            """Forward pass."""
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    # Choose device and Ray GPU resources based on what's actually available.
    if request.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        requested_num_gpus = request.scale.num_gpus_per_worker
    elif request.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon (MPS): use the GPU in PyTorch but do not request CUDA
        # GPUs from Ray, since Ray does not see MPS as "GPU" resources.
        device = torch.device("mps")
        requested_num_gpus = 0.0
    else:
        # Pure CPU fallback.
        device = torch.device("cpu")
        requested_num_gpus = 0.0

    @ray.remote(num_gpus=requested_num_gpus)
    class Trainer:
        def __init__(self, seed: int, train_dataset) -> None:
            torch.manual_seed(seed)
            # Store the training dataset passed as an argument to avoid
            # capturing it in the actor definition's closure.
            self.train_dataset = train_dataset
            self.model = SimpleCifarNet().to(device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        def train_epoch(self, shard_indices: list[int]) -> float:
            """Train for one epoch on the given index shard and return loss."""
            shard_subset = torch.utils.data.Subset(self.train_dataset, shard_indices)
            loader = DataLoader(
                shard_subset,
                batch_size=request.scale.batch_size,
                shuffle=True,
                num_workers=2,
            )
            self.model.train()
            epoch_losses: list[float] = []
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            return float(statistics.mean(epoch_losses)) if epoch_losses else 0.0

        def get_state_dict(self) -> dict[str, torch.Tensor]:
            """Return the model weights for aggregation."""
            return self.model.state_dict()

        def set_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
            """Load model weights from the coordinator."""
            self.model.load_state_dict(state_dict)

    num_workers = request.scale.num_workers
    indices_per_worker: list[list[int]] = [[] for _ in range(num_workers)]
    for idx in range(len(train_dataset)):
        indices_per_worker[idx % num_workers].append(idx)

    workers = [
        Trainer.remote(seed=request.random_seed + i, train_dataset=train_dataset)
        for i in range(num_workers)
    ]

    # Coordinate a few synchronous data-parallel epochs.
    for _epoch in range(request.scale.num_epochs):
        # Broadcast the current "best" model (from worker 0) to all workers.
        state_dict = ray.get(workers[0].get_state_dict.remote())
        ray.get([worker.set_state_dict.remote(state_dict) for worker in workers])

        # Run a local epoch on each worker's shard.
        loss_futures = [
            worker.train_epoch.remote(indices_per_worker[i]) for i, worker in enumerate(workers)
        ]
        _losses = ray.get(loss_futures)

    # Evaluate on the held-out test set using worker 0 as the reference model.
    final_state_dict = ray.get(workers[0].get_state_dict.remote())
    reference_model = SimpleCifarNet().to(device)
    reference_model.load_state_dict(final_state_dict)
    reference_model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=request.scale.batch_size,
        shuffle=False,
        num_workers=2,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = reference_model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    training_time_seconds = float(time.perf_counter() - start_time)

    num_parameters = sum(p.numel() for p in reference_model.parameters() if p.requires_grad)
    test_accuracy = correct / total if total else 0.0

    ray.shutdown()

    return Cifar10TrainResult(
        run_id=request.run_id,
        scale=request.scale,
        test_accuracy=test_accuracy,
        training_time_seconds=training_time_seconds,
        num_parameters=num_parameters,
        effective_num_workers=num_workers,
    )


@activity.defn
async def train_cifar10_with_ray(request: Cifar10TrainRequest) -> Cifar10TrainResult:
    """Train a CIFAR-10 classifier at the requested Ray scale.

    This activity is designed to be long-running and compute-heavy, making it
    a good demonstration of Temporal's strengths. The Ray job can scale from a
    single-worker laptop run to a multi-node cluster run without changing the
    workflow orchestration code.
    """
    activity.logger.info(
        "Starting CIFAR-10 Ray training run %s with %s workers",
        request.run_id,
        request.scale.num_workers,
    )
    result = await asyncio.to_thread(_train_cifar10_with_ray_sync, request)
    activity.logger.info(
        "Completed CIFAR-10 Ray training run %s with accuracy %.3f in %.1fs",
        request.run_id,
        result.test_accuracy,
        result.training_time_seconds,
    )
    return result
