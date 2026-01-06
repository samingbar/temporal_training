"""Utility script to launch a local Ray cluster for CIFAR-10 experiments.

This script is intended for manual use during development. It starts a local
Ray head node by invoking the ``ray`` CLI and prints connection instructions
for the CIFAR-10 Temporal worker and activities.

Example:
    uv run -m src.workflows.train_tune.cifar10_scaleup.local_ray_cluster
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the local Ray cluster launcher."""
    parser = argparse.ArgumentParser(
        description="Start a local Ray head node for CIFAR-10 scale-up experiments.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6379,
        help="TCP port for the Ray head node (default: 6379).",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8265,
        help="Port for the Ray dashboard (default: 8265).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Start a local Ray head node using the Ray CLI.

    The Ray head runs in the background after this script exits. When you are
    finished, shut it down with ``ray stop``.
    """
    if shutil.which("ray") is None:
        msg = (
            "Ray CLI not found in PATH. Install 'ray[default]' in your environment "
            "to use this script."
        )
        print(msg, file=sys.stderr)
        return 1

    args = _parse_args(argv)

    command = [
        "ray",
        "start",
        "--head",
        f"--port={args.port}",
        f"--dashboard-port={args.dashboard_port}",
    ]

    print("Starting local Ray head node for CIFAR-10 experiments:\n")
    print("  " + " ".join(command) + "\n")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI failure path
        print(
            f"Failed to start Ray head node (exit code {exc.returncode}). "
            "A Ray cluster may already be running. "
            "Try running 'ray status' or 'ray stop' and then retry.",
            file=sys.stderr,
        )
        return exc.returncode

    print(
        "Ray head node started.\n\n"
        f"- Head address: 127.0.0.1:{args.port}\n"
        "- To have the CIFAR-10 Temporal worker connect to this cluster, set:\n"
        f"    export RAY_ADDRESS='127.0.0.1:{args.port}'\n"
        "  before launching the worker process.\n\n"
        "When finished, stop the cluster with:\n"
        "    ray stop\n",
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - manual utility
    raise SystemExit(main())
