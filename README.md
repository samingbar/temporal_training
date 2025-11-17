# Temporal Python SDK Project Template

![GitHub CI](https://github.com/kawofong/temporal-python-template/actions/workflows/ci.yml/badge.svg)
[![Code Coverage](https://img.shields.io/codecov/c/github/kawofong/temporal-python-template.svg?maxAge=86400)](https://codecov.io/github/kawofong/temporal-python-template?branch=master)
[![GitHub License](https://img.shields.io/github/license/kawofong/temporal-python-template)](https://github.com/kawofong/temporal-python-template/blob/main/LICENSE)

## Introduction

A modern, production-ready template for building Temporal applications using [Temporal Python SDK](https://docs.temporal.io/dev-guide/python). This template provides a solid foundation for developing Workflow-based applications with comprehensive testing, linting, and modern Python tooling.

### What's Included

- Complete testing setup (pytest) with async support
- Pre-configured development tooling (e.g. ruff, pre-commit) and CI
- Comprehensive documentation and guides
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

1. **Run the example workflow** (in a separate terminal):

   ```bash
   # Start the worker
   uv run -m src.workflows.http.worker

   # In another terminal, execute a workflow
   uv run -m src.workflows.http.http_workflow
   ```

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
# Start the CIFAR-10 worker
uv run -m src.workflows.cifar10.worker

# In another terminal, execute the scaling workflow
uv run -m src.workflows.cifar10.cifar10_workflow
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
uv run -m src.workflows.bert.worker

# In another terminal, execute the workflow
uv run -m src.workflows.bert.bert_workflow
```

By default, the workflow runs two BERT fine-tuning configurations on GLUE SST-2,
aggregates their metrics (loss, accuracy, training time), and prints a concise
summary so you can compare the impact of training duration and hyperparameters.

### Next Steps

- Check out some [example prompts](./docs/example-prompts.md) to generate Temporal Workflows using your favorite tool.
- After you have built your first Temporal Workflow, read [DEVELOPERS.md](./DEVELOPERS.md) to learn about development tips & tricks using this template.
- See [`docs/temporal-patterns.md`](./docs/temporal-patterns.md) for advanced Temporal patterns
- Check [`docs/testing.md`](./docs/testing.md) for Temporal testing best practices

## License

[MIT License](LICENSE).
