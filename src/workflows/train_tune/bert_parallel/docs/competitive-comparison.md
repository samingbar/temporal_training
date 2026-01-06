# Competitive Comparison – BERT Parallel Training + Evaluation

This document compares the `bert_parallel` pattern—multiple checkpointed BERT training + eval runs orchestrated via Temporal—to common alternatives.

---

## Temporal (this repo’s approach)

- **Durability semantics**
  - Workflow history records the evolution of each experiment.
  - Activities (snapshotting, training, evaluation, inference) are retried automatically.
  - Checkpoint information flows via signals, enabling resumption from mid‑run state.

- **Long‑running parallel processes**
  - Many training runs can be in flight simultaneously.
  - Temporal is built for long‑running workflows; training can span minutes or hours.
  - Heartbeats and signals provide fine‑grained visibility into progress.

- **Code‑first expressiveness**
  - Coordinator logic lives in Python (`CoordinatorWorkflow`), not JSON/YAML.
  - Parallelism is expressed with loops + `execute_child_workflow` + `asyncio.gather`.

- **Deterministic replay**
  - Workflows are deterministic; all nondeterministic ML work stays in activities.
  - You can replay the coordinator and children to understand exactly how experiments unfolded.

- **Portability**
  - Temporal itself is cloud‑agnostic; workers can run across fleets of GPU and CPU nodes in any cloud or on‑prem.

- **Operational ergonomics**
  - Temporal Web shows:
    - Parent coordinator workflow.
    - Child training and eval workflows.
    - Activity histories and failures.
  - Signals and queries expose checkpoint status without direct filesystem access.

- **Scaling model**
  - Horizontal scaling via additional workers on:
    - `bert-training-task-queue` for training.
    - `bert-eval-task-queue` for eval/coordinator.
  - Clear separation of heavy GPU workloads from orchestration.

---

## AWS Step Functions

- **Durability & parallelism**
  - State machines can fan out parallel branches.
  - However, galaxy‑shaped JSON definitions become hard to maintain for many experiments.

- **Checkpoint awareness**
  - Checkpoint paths must be manually threaded through states.
  - No native integration with experiment resumption semantics like Temporal’s replay + signals model.

- **Use case fit**
  - Well‑suited for glueing AWS services together.
  - Less ergonomic for dense, experiment‑heavy ML orchestration with dozens of parallel BERT runs and mid‑run checkpoints.

---

## Azure Durable Functions

- **Durability & fan‑out**
  - Orchestrator functions can fan out across many activities.
  - Durable state is persisted in Azure storage.

- **Limitations vs this pattern**
  - Orchestration is more tightly coupled to the Azure Functions runtime.
  - Checkpoint handling and dataset snapshots must still be built manually.

---

## Apache Airflow

- **Parallelism & scheduling**
  - DAGs can map over multiple runs, and tasks can be scheduled across workers.
  - Best suited for nightly batch jobs and ETL pipelines.

- **Checkpointed ML experiments**
  - Checkpoint and snapshot management live inside operators or external services.
  - Tracking many concurrent BERT experiments and their checkpoints is possible, but visibility is spread across logs and pipelines, not a single persisted workflow history.

---

## Dagster / Prefect

- **Durability & scaling**
  - Both orchestrators handle state, retries, and concurrent runs well.

- **ML experiment ergonomics**
  - Great for data assets and pipelines.
  - Still require explicit patterns for:
    - Checkpoint reuse.
    - Dataset snapshot management.
    - Mid‑run progress signaling.

- **Comparison**
  - The `bert_parallel` pattern is intentionally narrower in scope but highly tuned to:
    - Long‑running, checkpoint‑aware training.
    - Parallel experiment orchestration.
    - Temporal’s deterministic workflow model.

---

## When to choose this Temporal pattern

Choose the `bert_parallel` + Temporal approach when:

- You want to run **multiple long‑running BERT experiments in parallel**, each with dataset snapshots and mid‑run checkpoints.
- You need **crash‑safe orchestration** and **deterministic replay** across training + evaluation.
- You want a **code‑first, Python‑native** orchestration layer that scales across heterogeneous compute (GPU/CPU) and environments.

If your needs are:

- Simple ETL jobs → Airflow/Dagster/Prefect may be simpler.
- Pure serverless glue on a single cloud → Step Functions or Durable Functions can work.

But for **parallel, checkpoint‑aware, experiment‑heavy BERT training and evaluation**, Temporal’s workflow model—as used in this `bert_parallel` package—provides stronger durability, better replay semantics, and more natural scaling and introspection.
