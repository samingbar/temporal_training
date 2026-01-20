"""Activity for streaming LLM-style tokens back to a workflow.

This example keeps the LLM part as a simple stub so the focus stays on how
the activity and workflow communicate via signals.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from temporalio import activity
from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from src.workflows.agents.streaming_test.types import StreamLLMRequest, LLMChunk


ADDRESS = "localhost:7233"
OPENAI_MODEL = os.getenv("OPENAI_STREAMING_MODEL", "gpt-4.1-mini")


async def _openai_stream(prompt: str) -> AsyncIterator[str]:
    """Stream tokens from OpenAI Chat Completions API.

    Requires the ``openai`` package and ``OPENAI_API_KEY`` to be configured
    in the environment. Uses ``chat.completions.create(..., stream=True)``.
    """
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - runtime configuration issue
        raise RuntimeError(
            "OpenAI streaming requested but the 'openai' package is not installed."
        ) from exc

    client = OpenAI()

    # The OpenAI Python SDK returns a synchronous generator when stream=True.
    # We iterate over it directly here since activities are allowed to perform
    # blocking I/O, and this example is intended for interactive demos.
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            content = getattr(delta, "content", None)
            if not content:
                continue
            # Newer clients may return content as a plain string.
            if isinstance(content, str):
                yield content
            else:
                # Fallback for any non-string payloads (e.g., lists/parts).
                yield str(content)


@activity.defn
async def stream_llm_activity(request: StreamLLMRequest) -> None:
    """Stream tokens from an LLM-like source and signal them back to the workflow."""
    info = activity.info()
    client = await Client.connect(ADDRESS, data_converter=pydantic_data_converter)
    handle = client.get_workflow_handle(info.workflow_id, run_id=info.workflow_run_id)

    seq = 0
    buffer: list[str] = []

    async for token in _openai_stream(request.prompt):
        buffer.append(token)

        # Batch tokens to reduce history size.
        if len(buffer) >= 20:
            chunk = LLMChunk(
                request_id=request.request_id,
                seq=seq,
                text="".join(buffer),
                done=False,
            )
            await handle.signal("llm_chunk", chunk)
            seq += 1
            buffer.clear()

    # Send the final (possibly empty) chunk and mark the stream as done.
    final_text = "".join(buffer)
    final_chunk = LLMChunk(
        request_id=request.request_id,
        seq=seq,
        text=final_text,
        done=True,
    )
    await handle.signal("llm_chunk", final_chunk)


__all__ = ["stream_llm_activity"]
