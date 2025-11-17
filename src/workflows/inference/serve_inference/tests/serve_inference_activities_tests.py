"""Unit tests for Serve inference activities with mocked dependencies."""

import pytest
from temporalio.testing import ActivityEnvironment

from src.workflows.serve_inference.serve_inference_activities import (
    InferenceRequest,
    call_serve_inference,
)


@pytest.mark.asyncio
async def test_call_serve_inference_returns_success_when_endpoint_returns_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should return parsed JSON and 200 on success."""

    async def fake_post_json(url: str, json_payload: dict, timeout: float):  # noqa: ARG001
        return 200, {"ok": True, "echo": json_payload}

    # Patch the HTTP helper
    from src.workflows.serve_inference import serve_inference_activities as mod

    monkeypatch.setattr(mod, "_post_json", fake_post_json)

    env = ActivityEnvironment()
    req = InferenceRequest(endpoint_url="http://localhost:8000", payload={"text": "hi"})
    result = await env.run(call_serve_inference, req)

    assert result.status_code == 200
    assert result.output is not None
    assert result.output["ok"] is True
    assert result.output["echo"]["text"] == "hi"


@pytest.mark.asyncio
async def test_call_serve_inference_returns_error_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should surface non-2xx as error with status and text."""

    async def fake_post_json(url: str, json_payload: dict, timeout: float):  # noqa: ARG001
        return 500, "internal error"

    from src.workflows.serve_inference import serve_inference_activities as mod

    monkeypatch.setattr(mod, "_post_json", fake_post_json)

    env = ActivityEnvironment()
    req = InferenceRequest(endpoint_url="http://localhost:8000", payload={"x": 1})
    result = await env.run(call_serve_inference, req)

    assert result.status_code == 500
    assert result.output is None
    assert result.error is not None
    assert "internal error" in result.error


@pytest.mark.asyncio
async def test_call_serve_inference_returns_599_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Should map exceptions to a 599-like transport error with message."""

    async def fake_post_json(url: str, json_payload: dict, timeout: float):  # noqa: ARG001
        raise RuntimeError("boom")

    from src.workflows.serve_inference import serve_inference_activities as mod

    monkeypatch.setattr(mod, "_post_json", fake_post_json)

    env = ActivityEnvironment()
    req = InferenceRequest(endpoint_url="http://localhost:8000", payload={"x": 1})
    result = await env.run(call_serve_inference, req)

    assert result.status_code == 599
    assert result.output is None
    assert result.error == "boom"

