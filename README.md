# Temporal Python SDK Project Template

![GitHub CI](https://github.com/kawofong/temporal-python-template/actions/workflows/ci.yml/badge.svg)
[![Code Coverage](https://img.shields.io/codecov/c/github/kawofong/temporal-python-template.svg?maxAge=86400)](https://codecov.io/github/kawofong/temporal-python-template?branch=master)
[![GitHub License](https://img.shields.io/github/license/kawofong/temporal-python-template)](https://github.com/kawofong/temporal-python-template/blob/main/LICENSE)

## Introduction

A modern, production-ready template for building Temporal applications using the [Temporal Python SDK](https://docs.temporal.io/dev-guide/python). This template provides a solid foundation for developing workflow-based applications with comprehensive testing, linting, and modern Python tooling.

### What's Included

- Complete testing setup (pytest) with async support
- Pre-configured development tooling (e.g. ruff, pre-commit) and CI
- Comprehensive documentation and guides
- End-to-end ML orchestration examples (CIFAR-10 scaling, BERT fine-tuning)
- An inference workflow example using Ray Serve
- An LLM-powered agent workflow example (`MAKER_test`) using reusable prompt tooling
- [AGENTS.md](https://agents.md/) to provide the context and instructions to help AI coding agents work on your project

## Getting Started

### Prerequisites

- [uv](https://docs.astral.sh/uv/)
- [Temporal CLI](https://docs.temporal.io/cli#install)

### Quick Start

1. **Clone and setup the project:**

   ```bash
   git clone https://github.com/kawofong/temporal-python-template.git
   cd temporal-python-template
   uv sync --dev
   ```

1. **Install development hooks:**

   ```bash
   uv run poe pre-commit-install
   ```

1. **Run tests:**

   ```bash
   uv run poe test
   ```

1. **Start Temporal Server**:

   ```bash
   temporal server start-dev
   ```

1. **Run an example workflow** (in separate terminals):

   - CIFAR-10 Ray scaling:

     ```bash
     # Terminal 1: start the worker
     uv run -m src.workflows.train-tune.cifar10_scaleup.worker

     # Terminal 2: execute the scaling workflow
     uv run -m src.workflows.train-tune.cifar10_scaleup.cifar10_workflow
     ```

   - BERT fine-tuning:

     ```bash
     # Terminal 1: start the worker
     uv run -m src.workflows.train-tune.bert_finetune.worker

     # Terminal 2: execute the workflow
     uv run -m src.workflows.train-tune.bert_finetune.bert_workflow
     ```

   - MAKER agent demo (LLM-based x+1 with voting):

     ```bash
     # Terminal 1: start the MAKER worker
     uv run -m src.workflows.agents.MAKER_test.worker

     # Terminal 2: execute the agent workflows
     uv run -m src.workflows.agents.MAKER_test.run
     ```

## Example Workflows

### CIFAR-10 Ray Scaling Workflow

The template also includes a CIFAR-10 + Ray scaling workflow that demonstrates how
Temporal can orchestrate long-running ML training jobs as you scale from a single
worker on your laptop to many workers in a Ray cluster.

This example is aimed at applied scientists who want to see scaling behavior without
changing workflow code: the Temporal workflow orchestration stays the same while
the Ray activity scales up or down.

> Note: The CIFAR-10 activity expects `ray[default]`, `torch`, and `torchvision`
> to be installed. After adding them to your environment, run `uv sync --dev`.

To run the CIFAR-10 scaling demo:

```bash
#Start the Ray Cluster
 uv run -m src.workflows.train_tune.cifar10_scaleup.local_ray_cluster

#In a new terminal, update the RAY_ADDRESS
export RAY_ADDRESS='127.0.0.1:6379'

# Start the CIFAR-10 worker
uv run -m src.workflows.train_tune.cifar10_scaleup.worker

# In another terminal, execute the scaling workflow
uv run -m src.workflows.train_tune.cifar10_scaleup.cifar10_workflow
```

The workflow will sweep over multiple Ray scale configurations (e.g. 1, 2, and 4 workers),
each as a separate long-running activity. The activity encapsulates the Ray job and returns
compact metrics (accuracy, wall-clock time, parameter count), while Temporal provides
orchestration, retries, and durability.

### BERT Fine-Tuning Workflow

The template also includes a BERT fine-tuning workflow using Hugging Face Transformers.
This demonstrates how Temporal can orchestrate long-running NLP experiments while
delegating model training and dataset handling to external libraries.

To run the BERT fine-tuning demo:

```bash
# Start the BERT worker
uv run -m src.workflows.train_tune.bert_finetune.worker

# In another terminal, execute the workflow
uv run -m src.workflows.train_tune.bert_finetune.bert_workflow
```

By default, the workflow runs two BERT fine-tuning configurations on GLUE SST-2,
aggregates their metrics (loss, accuracy, training time), and prints a concise
summary so you can compare the impact of training duration and hyperparameters.

### Agentic MAKER Test Workflow

The `MAKER_test` workflow demonstrates an LLM-powered agent that repeatedly solves
the simple numeric task `n → n + 1` while using a MAKER-style voting loop to guard
against bad generations. It also containes a "Normal" agentic loop for comparison.

Key pieces:

- Prompt modeling and history are built with `src/resources/myprompts`
- Structured agent I/O types live in `src/workflows/agents/MAKER_test/agent_types.py`
- The core orchestration lives in `src/workflows/agents/MAKER_test/workflow.py`
- LLM and tool-calling activities live in `src/workflows/agents/MAKER_test/activities.py`

To run the demo:

```bash
# Terminal 1: start the MAKER worker
uv run -m src.workflows.agents.MAKER_test.worker

# Terminal 2: execute both the baseline and MAKER workflows
uv run -m src.workflows.agents.MAKER_test.run
```

This will:

- Run a baseline agent (`NormalAgentWorkflow`) that calls the LLM step by step
- Run the MAKER-style agent (`MakerWorkflow`) that spawns multiple LLM calls,
  filters/red-flags outputs, and performs voting over the numeric results.

You can inspect and adapt this example to build more sophisticated LLM agents that
use tools (via `src/resources/mytools`) and richer prompt assemblies (`src/resources/myprompts`).

### Next Steps

- Check out some [example prompts](./docs/example-prompts.md) to generate Temporal Workflows using your favorite tool.
- After you have built your first Temporal Workflow, read [DEVELOPERS.md](./DEVELOPERS.md) to learn about development tips & tricks using this template.
- See [`docs/temporal-patterns.md`](./docs/temporal-patterns.md) for advanced Temporal patterns
- Check [`docs/testing.md`](./docs/testing.md) for Temporal testing best practices

## License

[MIT License](LICENSE).
