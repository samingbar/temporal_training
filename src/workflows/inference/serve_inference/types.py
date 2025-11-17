"""Typed models for Ray Serve batch inference orchestration.

This module centralizes request/response schemas and workflow I/O models used by
the Serve inference example. Keeping these models separate from workflow/activities
allows:

- Reuse across activities, workflows, and client code
- Clear separation of concerns (data contracts vs. orchestration/side-effects)
- Easier testing and validation

Notes
- All models are Pydantic BaseModel subclasses for validation and type safety.
- Forward references are enabled to simplify intra-module type references.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, HttpUrl

class BatchInferenceItem(BaseModel):
    """A single item to be sent to the model for inference.

    Encapsulates an arbitrary JSON-serializable payload that your Ray Serve
    deployment understands. Common examples include text, token IDs, or image
    references depending on the model interface.
    """

    payload: dict
    """JSON-serializable payload for the model."""


class BatchInferenceInput(BaseModel):
    """Input model for batch inference orchestration.

    The workflow fans out one activity per item using the provided endpoint
    and route. Configure per-request timeout to match your model SLOs.
    """

    endpoint_url: HttpUrl
    """Base URL of the Ray Serve endpoint (e.g., http://localhost:8000)."""

    route: str = "/inference"
    """Relative route accepting POST requests."""

    items: list[BatchInferenceItem]
    """List of items to infer in parallel."""

    per_request_timeout_seconds: float = 5.0
    """Timeout for each underlying activity HTTP request."""

class BatchInferenceOutput(BaseModel):
    """Aggregated output for batch inference.

    Results preserve the order of the requested items for easier correlation
    on the client side.
    """

    results: list[InferenceResponse]
    """One result per requested input item, preserving order."""


class InferenceRequest(BaseModel):
    """Input model for a single inference request to Ray Serve.

    Activity code uses this to POST JSON to the configured Ray Serve route.
    """

    endpoint_url: HttpUrl
    """Base URL of the Ray Serve endpoint (e.g., http://localhost:8000)."""

    route: str = "/inference"
    """Relative route on the endpoint that accepts POST requests."""

    payload: dict[str, Any]
    """JSON payload to send to the model (must be JSON-serializable)."""

    timeout_seconds: float = 5.0
    """Request timeout for the HTTP call."""


class InferenceResponse(BaseModel):
    """Output model for a single inference response from Ray Serve.

    Either ``output`` will be populated with parsed JSON on success, or ``error``
    will contain a human-readable message on failure (including transport errors).
    """
    status_code: int
    """HTTP status code returned by the endpoint."""

    output: dict[str, Any] | None = None
    """Parsed JSON output from the model, if available."""

    error: str | None = None
    """Error message, if the request failed or response was invalid."""

__all__ = [
    "BatchInferenceItem",
    "BatchInferenceInput",
    "BatchInferenceOutput",
    "InferenceRequest",
    "InferenceResponse",
]


