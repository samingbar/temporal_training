"""Typed models for the streaming_test example.

These Pydantic models define the data contracts used between the workflow,
activity, and any external clients for the streaming_test workflow.
"""

from __future__ import annotations

from pydantic import BaseModel


class StreamLLMRequest(BaseModel):
    """Input model for the ``stream_llm_activity`` activity."""

    prompt: str
    """Prompt text to send to the (simulated) LLM."""

    request_id: str
    """Opaque identifier used by the workflow to group streamed chunks."""


class LLMChunk(BaseModel):
    """Signal payload for streaming LLM text back into the workflow."""

    request_id: str
    """Identifier tying this chunk to a particular logical request."""

    seq: int
    """Monotonic sequence number for ordering and idempotency."""

    text: str
    """Chunk of response text."""

    done: bool
    """Whether this is the final chunk for the request."""


__all__ = ["StreamLLMRequest", "LLMChunk"]
