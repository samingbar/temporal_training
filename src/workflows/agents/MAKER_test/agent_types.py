from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AgentInput(BaseModel):
    task: str


class LlmResponse(BaseModel):
    result: int


class AgentStepInput(BaseModel):
    messages: List[Dict[str, Any]]


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class AgentStepOutput(BaseModel):
    # Whether the model finished
    is_final: bool

    # Plain-text final answer
    output_text: Optional[str] = None

    # If tool requested:
    tool_call: Optional[ToolCall] = None

    # Raw model message for history (optional)
    model_message: Dict[str, Any]
