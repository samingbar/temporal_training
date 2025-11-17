"""Activities for invoking Ray Serve inference endpoints via HTTP.

This module defines an activity, ``call_serve_inference``, which performs
non-deterministic network I/O against a Ray Serve deployment. Workflows
should call this activity and remain deterministic.
"""

from __future__ import annotations

from typing import Any

import aiohttp
from pydantic import ValidationError
from temporalio import activity
from src.workflows.serve_inference.types import (
    InferenceRequest,
    InferenceResponse,
)

async def _post_json(url: str, json_payload: dict[str, Any], timeout: float) -> tuple[int, Any]:
    """POST JSON to ``url`` and parse a JSON response when possible.

    Parameters
    - url: Fully qualified endpoint URL (e.g., http://localhost:8000/inference)
    - json_payload: JSON-serializable request body
    - timeout: Total timeout for the HTTP request in seconds

    Returns
    A tuple of ``(status_code, data)`` where ``data`` is the parsed JSON body
    when available, otherwise the raw response text. Kept separate for simpler
    monkeypatching in tests.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=json_payload, timeout=timeout) as resp:
            status = resp.status
            # Try JSON first; fall back to text for debugging
            try:
                data = await resp.json()
            except Exception:  # noqa: BLE001
                data = await resp.text()
            return status, data

@activity.defn
async def call_serve_inference(request: InferenceRequest) -> InferenceResponse:
    """Invoke a Ray Serve endpoint over HTTP and return a structured result.

    The activity centralizes non-deterministic I/O. Call from a workflow with
    explicit timeouts and a retry policy, as appropriate to your SLOs.
    """
    full_url = f"{str(request.endpoint_url).rstrip('/')}{request.route}"
    activity.logger.info("Calling Ray Serve at %s", full_url)

    try:
        status, data = await _post_json(full_url, request.payload, request.timeout_seconds)
        if status >= 200 and status < 300:
            # Validate JSON shape into dict[str, Any] via Pydantic
            try:
                parsed = data if isinstance(data, dict) else {"raw": data}
                return InferenceResponse(status_code=status, output=parsed)
            except ValidationError as ve:  # pragma: no cover
                return InferenceResponse(status_code=status, error=str(ve))
        return InferenceResponse(status_code=status, error=str(data))
    except Exception as exc:  # noqa: BLE001
        activity.logger.exception("Ray Serve inference call failed: %s", exc)
        return InferenceResponse(status_code=599, error=str(exc))

__all__ = ["call_serve_inference"]
