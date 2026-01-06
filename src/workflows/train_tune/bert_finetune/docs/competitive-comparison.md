# Competitive Comparison – BERT Fine-Tuning (Baseline)

This document compares the `bert_finetune` pattern to other common ways of orchestrating BERT training and inference.

Because this package is the **baseline** (no snapshots/checkpoint resumes), the focus here is on how Temporal helps even the simplest experiments.

---

## Temporal (this repo’s approach)

- **Durability semantics**
  - Workflow history durably records experiment inputs and outputs.
  - `fine_tune_bert` and `run_bert_inference` are retried automatically according to Temporal’s retry policy.

- **Long‑running processes**
  - Training can take minutes to hours; Temporal is designed for long‑lived activities.
  - Heartbeats from `fine_tune_bert` enable liveness tracking and responsive cancellation.

- **Code‑first expressiveness**
  - Workflows are plain Python, with normal control flow.
  - No need to encode hyperparameter sweeps or experiment logic in YAML or JSON.

- **Deterministic replay**
  - Workflows are deterministic; all non‑deterministic ML work lives in activities.
  - You can replay experiments for debugging or auditing without touching the external world.

- **Portability**
  - Temporal runs in any environment (Temporal Cloud or self‑hosted); workers are just Python processes.
  - You can move from laptop to on‑prem to cloud GPUs without rewriting your orchestration code.

- **Operational ergonomics**
  - Temporal Web and CLI give a unified view of:
    - Workflows (experiments).
    - Activities (training/inference runs).
    - Failures and retries.

- **Scaling model**
  - Horizontal scaling via more workers on `bert-finetune-task-queue`.
  - Easy to later split training vs. inference queues as your system grows.

---

## AWS Step Functions

- **Durability**
  - Durable state machine with retries, but experiment state is spread across:
    - Step Functions history.
    - Logs for Lambda/containers.

- **Long‑running processes**
  - Long‑running BERT training is possible but often awkward (polling, Wait states).
  - Human‑in‑the‑loop flows or multi‑day experiments require careful design.

- **Code‑first vs YAML**
  - Main orchestration surface is JSON/YAML state machines; code is secondary.
  - Complex ML experiment logic becomes verbose and harder to maintain.

- **Portability**
  - Tied to AWS; migrating orchestration off AWS requires a redesign.

---

## Azure Durable Functions

- **Durability**
  - Good durable orchestrations on Azure storage, similar to Temporal’s concept.
  - Coupled to the Azure Functions runtime and tooling.

- **Expressiveness**
  - Code‑based orchestrations, but:
    - Replay semantics and binding rules are more implicit than Temporal’s explicit workflow/activity split.

- **Portability**
  - Bound to Azure; moving to another cloud or on‑prem stack is non‑trivial.

---

## Apache Airflow

- **Durability**
  - DAG state is stored in a DB; tasks can retry.
  - Better suited to ETL, scheduling, and batch DAGs than to experiment workflows.

- **Long‑running processes**
  - Long tasks are supported but not first‑class.
  - Multi‑stage ML experiments (train + eval + inference) are possible but typically become complex DAGs.

- **Code‑first expressiveness**
  - DAGs in Python, but constrained by DAG semantics (acyclic, operator‑oriented).

---

## Dagster / Prefect

- **Durability**
  - Good state tracking and retries for pipelines.

- **ML ergonomics**
  - Strong data/asset modeling and type systems.
  - Checkpointed experiments and fine‑tuning flows are patterns you assemble on top.

- **Comparison to this repo**
  - This `bert_finetune` package is a *simpler* Temporal equivalent of a Dagster/Prefect pipeline:
    - Fewer features, but directly geared toward long‑running BERT experiments.
    - Easy to extend into checkpointed and coordinated experiments (`bert_checkpointing`, `bert_eval`).

---

## When to use this baseline Temporal pattern

`bert_finetune` is a good fit when:

- You’re starting with **simple, single‑machine experiments** and want a clear upgrade path toward more complex orchestration.
- You want **durable, replayable orchestration** from day one without pulling in a full MLOps platform.
- You prefer to keep orchestration **close to your ML code** in Python, not in a separate YAML system.

As your needs mature (checkpoint reuse, multi‑config sweeps, eval pipelines), move to:

- `bert_checkpointing` for dataset snapshots + checkpoint‑aware retry/resume.
- `bert_eval` for end‑to‑end “train + eval” experiment workflows.

