# Competitive Comparison – BERT Eval Coordinator

This document compares the `bert_eval` pattern (Temporal workflows + activities) to common alternatives for orchestrating BERT training and evaluation.

The comparison uses these dimensions:

- **Durability semantics**
- **Long‑running processes**
- **Code‑first expressiveness**
- **Deterministic replay**
- **Portability**
- **Operational ergonomics**
- **Scaling model**
- **Cost & efficiency (high level)**

---

## Temporal (this repo’s approach)

- **Durability semantics**
  - Workflow state is durably persisted; each decision is recorded in history.
  - Activities (training, evaluation, dataset snapshotting) are retried automatically according to policy.
  - `CoordinatorWorkflow` treats “train + eval” as a single durable experiment.

- **Long‑running processes**
  - Fine‑tuning can run for minutes or hours; Temporal is built for long‑lived workflows.
  - Heartbeats from `fine_tune_bert` let Temporal detect stalls and cancellations.
  - Checkpoints and dataset snapshots allow safe resumption after failures.

- **Code‑first expressiveness**
  - Orchestration is expressed in Python (`workflows.py`), not YAML.
  - You can use normal control flow, loops, and functions to express complex experiments.

- **Deterministic replay**
  - Workflows are deterministic; all non‑deterministic operations live in activities.
  - You can replay Coordinator / training / eval workflows from history to debug or audit experiments.

- **Portability**
  - Temporal is cloud‑agnostic (Temporal Cloud or self‑hosted in any Kubernetes/VM environment).
  - Workers are just Python processes; you choose where they run (on‑prem, cloud GPUs, mixed).

- **Operational ergonomics**
  - One control‑plane (Temporal Web, CLI, APIs) to inspect workflow state, child workflows, activities, and history.
  - Signals and queries provide live introspection (latest checkpoints, run status).

- **Scaling model**
  - Horizontal scaling by adding workers to task queues (`bert-training-task-queue`, `bert-eval-task-queue`).
  - Easy to separate GPU‑heavy work from orchestration.

- **Cost & efficiency**
  - You only pay for the compute you allocate to workers and your Temporal cluster.
  - Checkpoint‑aware retries reduce wasted GPU hours on failure.

---

## AWS Step Functions

- **Durability semantics**
  - State machine persists step transitions; retries are configurable.
  - However, you typically orchestrate via Lambda / container tasks and must layer checkpointing manually.

- **Long‑running processes**
  - Express has strict limits; Standard can handle longer durations but at higher cost and more complex billing.
  - Human‑in‑loop or multi‑day trainings require careful design (Wait states, external signaling).

- **Code‑first expressiveness**
  - Primary model is JSON/YAML state machines; code is secondary.
  - Complex branching and looping quickly become hard to maintain.

- **Deterministic replay**
  - No replay‑based debugging model like Temporal; you reason in terms of execution logs and step outputs.

- **Portability**
  - Fully tied to AWS; moving off Step Functions requires a redesign.

- **Operational ergonomics**
  - Good AWS Console integration but scattered across CloudWatch logs, Step Functions, and metrics.
  - Cross‑account or hybrid workflows add complexity.

- **Scaling model**
  - Scales well inside AWS, especially when paired with Lambda/Fargate.
  - BERT fine‑tuning workloads still require separate GPU infrastructure (e.g., EC2, ECS).

- **Cost & efficiency**
  - Charged per state transition and execution time; long‑running flows can get expensive.
  - Less natural fit for iterative experiment workflows than Temporal’s worker model.

---

## Azure Durable Functions

- **Durability semantics**
  - Durable Functions provide durable orchestrations with history in Azure storage.
  - Similar conceptual model to Temporal, but tightly coupled to Azure Functions runtime.

- **Long‑running processes**
  - Suitable for long‑running orchestrations, though scaling GPU workloads requires additional services.

- **Code‑first expressiveness**
  - Orchestrations are written in code (C#, JavaScript, Python), which is a plus.
  - However, replay rules and binding rules are less explicit than Temporal’s workflow vs. activity split.

- **Deterministic replay**
  - Durable Functions re‑executions resemble Temporal’s replay, but the programming model is more constrained.

- **Portability**
  - Tied to Azure’s Functions and storage stack; moving to another cloud or on‑prem is non‑trivial.

- **Operational ergonomics**
  - Diagnostics span Azure Portal, Application Insights, and storage logs.
  - Less unified view of worker processes than Temporal’s task queue model.

- **Scaling model**
  - Good for event‑driven workloads; scaling heavy GPU jobs still requires separate infrastructure (AKS, VMs).

---

## Apache Airflow

- **Durability semantics**
  - DAG state is stored in a relational DB; tasks can be retried.
  - Focus is on batch ETL and scheduled workflows rather than fine‑grained experiment control.

- **Long‑running processes**
  - Long tasks are possible but not the primary design goal; backfills and schedules dominate.
  - Human‑in‑the‑loop or weeks‑long workflows are awkward and often require custom operators.

- **Code‑first expressiveness**
  - DAGs are written in Python, but constrained by DAG semantics (acyclic graph, task operators).
  - Imperative experiment orchestration is less natural than in Temporal.

- **Deterministic replay**
  - Airflow reruns tasks; no notion of deterministic replay of an entire DAG from a durable event history.

- **Portability**
  - Self‑hosted is common; managed options exist, but portability across clouds requires effort.

- **Operational ergonomics**
  - Airflow UI is familiar for ETL, but tracing a single ML experiment through training + evaluation is less direct.

- **Scaling model**
  - Executor‑based: Celery/Kubernetes executors scale reasonably.
  - Mapping experiment runs to DAG instances can get noisy for high‑volume experimentation.

---

## Dagster / Prefect

- **Durability semantics**
  - Both provide solid orchestrators with state tracking and retries.
  - Durable, replayable ML pipelines are possible, but checkpoint‑aware training is a pattern you assemble yourself.

- **Long‑running processes**
  - Long‑running tasks are supported, but human‑in‑loop and multi‑day experimentation patterns are less first‑class than in Temporal.

- **Code‑first expressiveness**
  - Both are strongly code‑first, with rich type systems and graph semantics.
  - For simple DAG‑like experiments, this is ergonomic; for complex “train + eval + resume” flows, you often need custom scaffolding.

- **Deterministic replay**
  - Replay is more about re‑executing pipeline steps than fully deterministic workflow history.

- **Portability**
  - Dagster and Prefect can be self‑hosted or managed; portability is good within their ecosystems.

- **Operational ergonomics**
  - UIs are modern and ML‑friendly, but tying together many long‑running experiments with checkpoint‑aware resumption still requires explicit design.

- **Scaling model**
  - Both support distributed executors / agents; mapping GPU training vs CPU orchestration is feasible but not as separated by default as Temporal’s task queues.

---

## When to prefer this Temporal pattern

Choose the `bert_eval` + Temporal approach when:

- You need **durable, restartable experiments** where both training and evaluation are part of a single logical run.
- You care about **checkpoint‑aware retries** and avoiding wasted GPU time on failure.
- You want a **code‑first orchestration model** that lives alongside your ML code, not in a separate YAML layer.
- You need to scale GPU training and CPU orchestration **independently**, possibly across different clusters or clouds.

If your use case is mostly:

- Simple nightly ETL with short tasks → Airflow/Dagster/Prefect may be simpler.
- Pure serverless glue between cloud services in one provider → Step Functions / Durable Functions can be adequate.

But for **long‑running, experiment‑heavy, checkpointed ML workloads**, Temporal’s workflow model, as used in this repo, gives you stronger durability, clearer state modeling, and better replay/debug tooling.
