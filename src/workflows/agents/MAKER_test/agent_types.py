"""Pydantic models for the MAKER_test agent workflow.

Separated into a dedicated module to avoid shadowing the stdlib `types` module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AgentInput(BaseModel):
    task: str


class AgentStepInput(BaseModel):
    """Single step input to the LLM activity."""

    messages: List[Dict[str, Any]]


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class AgentStepOutput(BaseModel):
    """Structured result from a single LLM/tool step."""

    # Whether the model finished
    is_final: bool

    # Plain-text final answer
    output_text: Optional[str] = None

    # If tool requested:
    tool_call: Optional[ToolCall] = None

    # Raw model message for history (optional)
    model_message: Dict[str, Any]


class ValidateCompanyArgs(BaseModel):
    company_name: str


class IdentifySectorArgs(BaseModel):
    company_name: str


class IdentifyCompetitorsArgs(BaseModel):
    sector: str
    company_name: str


class BrowsePageArgs(BaseModel):
    url: str
    instructions: str


class GenerateReportArgs(BaseModel):
    company_name: str
    context: str


