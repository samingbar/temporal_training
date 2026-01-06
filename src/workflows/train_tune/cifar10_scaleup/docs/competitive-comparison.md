# Competitive Comparison – CIFAR-10 Ray Scale-Up

This document compares the `cifar10_scaleup` Temporal pattern against other orchestration options for **cluster-scale, GPU-heavy training experiments**.

The rubric mirrors the rest of `train_tune`:

- **Durability semantics**
- **Long-running processes**
- **Code-first expressiveness**
- **Deterministic replay**
- **Portability**
- **Operational ergonomics**
- **Scaling model**
- **Cost & limits** (qualitative)

---

## Temporal (this example)

- **Durability semantics**
  - Workflows (`Cifar10ScalingWorkflow`) provide exactly-once state progression.
  - Each scale configuration is a deterministic activity invocation with clear input/output models.
  - Activity failures (Ray errors, training crashes) are surfaced to workflows and can be retried.

- **Long-running processes**
  - Training activities can run for hours; Temporal keeps workflow state durable in its history store.
  - Worker restarts do not lose experiment progress; remaining scales can still be executed.

- **Code-first expressiveness**
  - Scaling experiments are regular Python code:
    - Iterate over `RayScaleConfig` entries.
    - Call `train_cifar10_with_ray` for each.
  - Easy to extend with:
    - Additional metrics.
    - Conditional logic (e.g., stop if accuracy exceeds a threshold).
    - Nested workflows for more complex patterns.

- **Deterministic replay**
  - Workflow logic is deterministic and side-effect-free.
  - Activities encapsulate all non-determinism (Ray, datasets, random seeds).
  - You can replay workflows in Temporal to understand which scales ran and in what order.

- **Portability**
  - Temporal Server can run on-prem, in the cloud, or via Temporal Cloud.
  - Workers are plain Python processes and can run anywhere with network access to Temporal and Ray.

- **Operational ergonomics**
  - Temporal Web gives a clear view of:
    - Which scale configurations ran.
    - Which activities succeeded or failed.
    - How long each run took.
  - You can cancel, retry, or inspect individual scaling experiments.

- **Scaling model**
  - Temporal scales via workers and task queues (`cifar10-ray-task-queue`).
  - Ray scales training workloads across many GPUs and nodes.
  - The two systems compose cleanly:
    - Temporal orchestrates.
    - Ray executes distributed training.

- **Cost & limits**
  - Temporal costs (if using a managed offering) are driven by the number of workflows/activities and history size.
  - Ray costs depend on the size and duration of your GPU cluster.
  - You can tune experiment design (number of scales, epochs) to fit your budget.

---

## AWS Step Functions

- **Durability semantics**
  - Strong durability and retries for state machines.
  - However, training runs typically live in external services (SageMaker, ECS, EKS), with Step Functions orchestrating them via API calls.

- **Long-running processes**
  - Long-running tasks are supported, but:
    - Per-state and payload size limits can complicate rich experiment metadata.
    - Modeling many scale configurations often leads to deeply nested or repetitive state machines.

- **Code-first expressiveness**
  - Main model is JSON/YAML state machines.
  - Expressing a dynamic list of `RayScaleConfig` entries and iterating over them is possible but more cumbersome than writing Python workflows.

- **Deterministic replay**
  - Executions are recorded, but:
    - There is no built-in concept of replaying the same code against execution history as in Temporal.

- **Portability**
  - Tied to AWS services and IAM.
  - Migrating scaling experiments to a different cloud or on-prem environment requires significant redesign.

- **Operational ergonomics**
  - Good visualizations in the AWS console.
  - But less natural for local development and debugging than Temporal’s unified worker model.

- **Scaling model**
  - Uses AWS-native scaling for services like ECS/EKS/SageMaker.
  - Integrating custom Ray clusters requires additional glue code and infrastructure.

- **Cost & limits**
  - Priced per state transition and duration.
  - Many small tasks or complex state machines can increase costs.

---

## Azure Durable Functions

- **Durability semantics**
  - Durable orchestrations with checkpoints and replay.
  - Similar in spirit to Temporal but tightly integrated with Azure Functions.

- **Long-running processes**
  - Suitable for long-running workflows, but:
    - Orchestrator code must follow Durable Functions patterns.
    - Integrating external systems like Ray clusters adds complexity.

- **Code-first expressiveness**
  - Orchestrators are code, but constrained by Durable Functions rules.
  - Scaling experiments with many configurations require careful management of function activity calls and state.

- **Deterministic replay**
  - Replay is built-in, but only within the Azure Durable Functions environment.

- **Portability**
  - Tied to Azure; not ideal if you expect to run experiments on heterogeneous infrastructure.

- **Operational ergonomics**
  - Azure tooling is solid, but:
    - Local development and test loops can be heavier than running Temporal workers directly with `uv`.

- **Scaling model**
  - Function apps scale based on triggers and load.
  - GPU training often requires separate services (AKS, VM scale sets).

- **Cost & limits**
  - Consumption or premium plans based on function invocations and runtime.
  - Long or frequent training experiments can accumulate cost.

---

## Airflow

- **Durability semantics**
  - DAG runs and task states are persisted in a metadata database.
  - Good for batch jobs, but lacks a true workflow history + replay story.

- **Long-running processes**
  - Possible via long-running tasks or external services.
  - Common pattern is to offload training to Kubernetes jobs or external scripts with callbacks.

- **Code-first expressiveness**
  - DAGs are defined in Python, but:
    - Dynamic scaling experiments become complex when you need to encode many `RayScaleConfig` values.
    - Control flow is constrained by DAG semantics.

- **Deterministic replay**
  - Re-running a DAG creates a new run; Airflow does not re-run the same code against past history as Temporal does.

- **Portability**
  - Airflow can run on many platforms, but:
    - Integrating with Ray clusters still requires custom operators or hooks.

- **Operational ergonomics**
  - Mature UI for DAG monitoring.
  - Less natural fit for interactive exploration or ad-hoc scaling experiments.

- **Scaling model**
  - Scales via schedulers and worker pools.
  - Training itself typically happens in external systems.

- **Cost & limits**
  - Infrastructure cost for the Airflow cluster plus any backing services (Kubernetes, Ray, etc.).

---

## Dagster / Prefect

- **Durability semantics**
  - Strong orchestration for tasks and jobs with retries.
  - Good handling of metadata and logs for ML experiments.

- **Long-running processes**
  - Can orchestrate long-running tasks and jobs for training workloads.
  - Still require you to design and integrate Ray-based training patterns.

- **Code-first expressiveness**
  - Python-native APIs; expressive and ergonomic.
  - Similar in spirit to Temporal for many use cases, but without Temporal’s replay semantics and long-lived workflow history model.

- **Deterministic replay**
  - Runs and logs can be inspected, but reruns are new executions rather than replays of the same code over stored event history.

- **Portability**
  - Both can be deployed on various environments (local, Kubernetes, cloud).

- **Operational ergonomics**
  - Strong UIs, especially for ML-centric flows.
  - May integrate nicely with Ray, but you will need to design the scaling pattern yourself.

- **Scaling model**
  - Use agents/executors that schedule jobs on different backends.
  - Integrating GPU-heavy Ray jobs is possible but a separate concern.

---

## When Temporal is the better fit

Temporal (as used in `cifar10_scaleup`) is especially strong when:

- You want a **durable control plane** for cluster-scale training experiments.
- You need to **coordinate many scale configurations** while preserving a clean history and replay semantics.
- You expect **worker crashes or restarts** and want guarantees that experiments will either complete or fail visibly with retry options.
- You prefer **code-first workflows** that can be tested locally and reused in production.

Other systems can orchestrate training jobs, but Temporal gives you:

- A reliable, replayable record of which configurations were run.
- Clear separation between orchestration logic and distributed training infrastructure (Ray).
- A consistent developer experience across both BERT and CIFAR-10 examples in this portfolio.
