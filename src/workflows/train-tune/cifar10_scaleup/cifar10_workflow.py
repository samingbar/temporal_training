"""Workflow for scaling CIFAR-10 training with Ray."""

import asyncio
from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel

    from src.workflows.cifar10.cifar10_activities import (
        Cifar10TrainRequest,
        Cifar10TrainResult,
        RayScaleConfig,
        train_cifar10_with_ray,
    )


class Cifar10ScalingInput(BaseModel):
    """Input to the CIFAR-10 scaling workflow."""

    experiment_name: str
    """Human-readable label for this experiment (e.g. 'cifar10-ray-scale-demo')."""

    scales: list[RayScaleConfig]
    """List of Ray scale configurations to sweep over."""

    dataset_root: str | None = None
    """Optional dataset cache root shared by all runs."""

    use_gpu: bool = False
    """Whether to request GPU devices from Ray workers."""

    random_seed: int = 42
    """Base random seed forwarded into the workers for reproducibility."""


class Cifar10ScalingOutput(BaseModel):
    """Summary of the full scaling experiment."""

    experiment_name: str
    """Echoed experiment name."""

    runs: list[Cifar10TrainResult]
    """Per-scale training results."""


@workflow.defn
class Cifar10ScalingWorkflow:
    """Workflow that sweeps CIFAR-10 training over multiple Ray scales.

    The orchestration logic is intentionally simple and deterministic: for each
    requested scale configuration, it kicks off a single long-running activity
    that handles the Ray cluster interaction. The activity can scale from
    laptop-only experiments to multi-node clusters without changing this
    workflow code, demonstrating how Temporal cleanly separates orchestration
    from execution.
    """

    @workflow.run
    async def run(self, input: Cifar10ScalingInput) -> Cifar10ScalingOutput:
        """Execute the scaling experiment and aggregate results."""
        workflow.logger.info(
            "Starting CIFAR-10 scaling experiment '%s' with %s scales",
            input.experiment_name,
            len(input.scales),
        )

        results: list[Cifar10TrainResult] = []

        for idx, scale in enumerate(input.scales):
            run_id = f"{input.experiment_name}-scale-{idx}-workers-{scale.num_workers}"
            workflow.logger.info(
                "Triggering Ray training run %s (%s workers, %s epochs)",
                run_id,
                scale.num_workers,
                scale.num_epochs,
            )
            request = Cifar10TrainRequest(
                run_id=run_id,
                scale=scale,
                dataset_root=input.dataset_root,
                use_gpu=input.use_gpu,
                random_seed=input.random_seed,
            )
            result: Cifar10TrainResult = await workflow.execute_activity(
                train_cifar10_with_ray,
                request,
                start_to_close_timeout=timedelta(hours=2),
            )
            results.append(result)

        workflow.logger.info(
            "Completed CIFAR-10 experiment '%s' with %s runs",
            input.experiment_name,
            len(results),
        )
        return Cifar10ScalingOutput(experiment_name=input.experiment_name, runs=results)


async def main() -> None:  # pragma: no cover
    """Execute a sample CIFAR-10 scaling workflow against a local Temporal server."""
    from temporalio.client import Client  # noqa: PLC0415
    from temporalio.contrib.pydantic import pydantic_data_converter  # noqa: PLC0415

    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    input_data = Cifar10ScalingInput(
        experiment_name="cifar10-ray-scaling-5090",
        scales=[
            RayScaleConfig(
                num_workers=1,
                num_epochs=40,
                batch_size=256,
                num_gpus_per_worker=1.0,
                description="1× GPU baseline (long, stable)",
            ),
            RayScaleConfig(
                num_workers=2,
                num_epochs=60,
                batch_size=512,
                num_gpus_per_worker=0.5,
                description="2 workers sharing a 5090",
            ),
            RayScaleConfig(
                num_workers=4,
                num_epochs=80,
                batch_size=512,
                num_gpus_per_worker=0.25,
                description="4 workers time-slicing a 5090",
            ),
        ],
        # Use the GPU if available; on a 5090 this should push
        # the longest configuration toward ~20 minutes of runtime.
        use_gpu=True,
        random_seed=42,
    )

    result = await client.execute_workflow(
        Cifar10ScalingWorkflow.run,
        input_data,
        id="cifar10-ray-scaling-demo-id",
        task_queue="cifar10-ray-task-queue",
    )

    print("\nCIFAR-10 scaling experiment results:")
    for run in result.runs:
        print(
            f"- {run.run_id}: workers={run.scale.num_workers}, "
            f"epochs={run.scale.num_epochs}, "
            f"accuracy={run.test_accuracy:.3f}, "
            f"time_s={run.training_time_seconds:.1f}",
        )


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
