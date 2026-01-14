"""Temporal multi-agent personal assistant workflow.

This module contains a Temporal Workflow that refactors the
``langchain_version.py`` multi-agent personal assistant example into a
fully Temporalized design:

- All LLM calls live in Activities (see ``activities.py``)
- Tools are defined and registered via the shared ``mytools`` package
  (see ``tools.py``)
- The workflow itself is deterministic and only orchestrates calls to
  Activities while keeping track of the conversation history.

The high-level behavior matches the LangChain example:

- A "supervisor" agent receives a natural language request
- The supervisor decides when to:
  - schedule events (via the ``schedule_event`` tool)
  - send emails (via the ``manage_email`` tool)
  - or do both in sequence
- The workflow keeps a structured trace of the tools used and the number
  of LLM steps taken.
"""

from __future__ import annotations

from datetime import timedelta
from typing import List

from temporalio import workflow

from src.resources.myprompts.history import PromptHistory
from src.resources.myprompts.models import (
    BasePrompt,
    ModelPrompt,
    SystemPrompt,
    TaskPrompt,
    UserPrompt,
)

from src.resources.custom_types.types import AgentInput as CompanyResearchAgentInput
from src.workflows.agents.company_research_agent.workflow import AgentLoopWorkflow

from .agent_types import (
    AgentStepInput,
    AgentStepOutput,
    ChatMessage,
    ChatResponse,
    ChatSessionConfig,
    PersonalAssistantInput,
    PersonalAssistantResult,
    ToolCall,
)
from .config import ADDRESS, PROVIDER, TASK_QUEUE


# ---------------------------------------------------------------------------
# Prompt configuration
# ---------------------------------------------------------------------------

# These prompts are adapted from the original LangChain multi-agent
# personal assistant example. In that example, the calendar and email
# agents are separate LangChain agents. In this Temporal version we keep
# the same role descriptions but expose their capabilities as tools that
# a single supervisor agent can call.

CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

CHAT_AGENT_PROMPT = (
    "You are a general-purpose conversational assistant. "
    "Answer user questions clearly and helpfully when no specialized "
    "tool (calendar, email, weather, web lookup) is required."
)

WEATHER_AGENT_PROMPT = (
    "You are a weather assistant. "
    "Given a location (and optionally units), call get_weather to "
    "retrieve current conditions and present them in a concise, "
    "user-friendly way."
)

WEB_LOOKUP_AGENT_PROMPT = (
    "You are a web lookup assistant. "
    "When the user asks for factual or background information, use "
    "web_lookup to search the web, fetch real page content, and then "
    "analyze that content to produce an accurate, concise summary."
)

SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events, send emails, chat generically, "
    "look up weather, and perform undifferentiated web lookups.\n\n"
    "Calendar agent capabilities:\n"
    f"{CALENDAR_AGENT_PROMPT}\n\n"
    "Email agent capabilities:\n"
    f"{EMAIL_AGENT_PROMPT}\n\n"
    "Chat agent capabilities:\n"
    f"{CHAT_AGENT_PROMPT}\n\n"
    "Weather agent capabilities:\n"
    f"{WEATHER_AGENT_PROMPT}\n\n"
    "Web lookup agent capabilities:\n"
    f"{WEB_LOOKUP_AGENT_PROMPT}\n\n"
    "You also have access to a long-running company research agent via "
    "the `company_research` tool, which can perform deep competitive "
    "analysis and return a structured report.\n\n"
    "You have access to tools that implement these capabilities. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence. "
    "When you are done, write a concise final answer for the user that "
    "summarizes what you did."
)


@workflow.defn
class PersonalAssistantWorkflow:
    """Workflow that orchestrates a multi-agent personal assistant.

    The workflow is responsible for:

    - Building and maintaining a prompt history using ``myprompts``
    - Calling an LLM step activity that can request tools
    - Calling a tool-execution activity when the LLM requests a tool
    - Looping until the LLM returns a final answer or a safety limit is
      reached

    All non-deterministic work (LLM calls, network I/O, etc.) is handled
    inside activities; the workflow state itself is simple and fully
    replayable.
    """

    def __init__(self) -> None:
        # Prompt history is modeled as a pure data structure so that it
        # can be safely replayed by the Temporal Workflow engine.
        self.history = PromptHistory()
        self.tool_calls: List[ToolCall] = []
        self.steps: int = 0

    @workflow.run
    async def run(self, request: PersonalAssistantInput) -> PersonalAssistantResult:
        """Entry point for the personal assistant workflow.

        Args:
            request: Structured input containing the user's natural
                language query.

        Returns:
            A ``PersonalAssistantResult`` that includes the final answer
            plus a trace of tool calls and LLM steps.
        """

        # Seed the conversation with the supervisor's system instructions
        # and the user's top-level request.
        self.history.add(SystemPrompt(text=SUPERVISOR_PROMPT.strip()))
        self.history.add(
            TaskPrompt(
                text=(
                    "The user may ask you to schedule meetings, send emails, "
                    "or perform both actions in one request. Use tools when "
                    "they are helpful, and keep the user-facing explanation "
                    "short and clear."
                ),
            )
        )
        self.history.add(UserPrompt(text=request.query))

        provider = PROVIDER
        max_steps = 8  # safety limit so demo workflows always terminate
        last_text: str = ""

        for step_index in range(max_steps):
            self.steps = step_index + 1

            # Convert the prompt history into provider-specific messages
            messages = self.history.to_messages(provider=provider)
            step_input = AgentStepInput(messages=messages)

            # 1. Ask the LLM what to do next (possibly invoking tools).
            raw_result = await workflow.execute_activity(
                "llm_step_activity",
                step_input,
                schedule_to_close_timeout=timedelta(seconds=60),
            )

            if isinstance(raw_result, dict):
                result = AgentStepOutput(**raw_result)
            else:
                result = raw_result

            # 2. If the model requested a tool, execute it via the
            #    generic tool activity and feed the result back into the
            #    conversation. For the special `company_research` tool,
            #    delegate to the long-running company_research_agent
            #    child workflow instead.
            if result.tool_call:
                self.tool_calls.append(result.tool_call)

                if result.tool_call.name == "company_research":
                    tool_response = await _run_company_research_subagent(result.tool_call)
                else:
                    tool_response = await workflow.execute_activity(
                        "tool_activity",
                        result.tool_call,
                        schedule_to_close_timeout=timedelta(seconds=30),
                    )

                # Record the tool output in the prompt history in a way
                # that is readable by both the LLM and humans.
                self.history.add(
                    UserPrompt(
                        text=f"[Tool {result.tool_call.name} output]: {tool_response}",
                    ),
                )
                workflow.logger.info(
                    "Tool %s executed with response: %s",
                    result.tool_call.name,
                    tool_response,
                )

                # Continue the loop so the LLM can observe the tool result.
                continue

            # 3. Capture any plain-text model output and treat the first
            #    assistant message as the final answer for this request.
            if result.output_text:
                last_text = result.output_text
                self.history.add(ModelPrompt(text=result.output_text))
                workflow.logger.info(
                    "LLM produced final answer after %s steps", self.steps
                )
                break

        final_message = last_text or "The assistant could not produce a response."
        return PersonalAssistantResult(
            final_response=final_message,
            tool_calls=self.tool_calls,
            steps=self.steps,
        )


@workflow.defn
class ChatPersonalAssistantWorkflow:
    """Long-lived chat workflow that talks to the assistant via signals.

    This workflow is designed to be driven by an external CLI. The CLI
    sends user messages using signals and reads the latest assistant
    response using a query. All LLM calls and tool invocations still
    happen inside activities, so the workflow remains deterministic and
    replay-safe.
    """

    def __init__(self) -> None:
        # Maintain a rolling conversation history that is shared across
        # all chat turns in the session.
        self.history = PromptHistory()
        self._pending_messages: List[ChatMessage] = []
        self._latest_response: ChatResponse | None = None
        self._turn_index: int = 0
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Signals and queries
    # ------------------------------------------------------------------

    @workflow.signal
    def submit_user_message(self, message: ChatMessage) -> None:
        """Receive a new user message from the CLI."""
        self._pending_messages.append(message)

    @workflow.signal
    def close(self) -> None:
        """Request a graceful shutdown of the chat session."""
        self._closed = True

    @workflow.query
    def get_latest_response(self) -> ChatResponse | None:
        """Return the most recent assistant response, if any."""
        return self._latest_response

    # ------------------------------------------------------------------
    # Main chat loop
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, config: ChatSessionConfig) -> None:
        """Run a long-lived chat session with the personal assistant."""

        # Seed the conversation with the same supervisor prompt used by
        # the one-shot workflow so behavior stays consistent.
        self.history.add(SystemPrompt(text=SUPERVISOR_PROMPT.strip()))
        self.history.add(
            TaskPrompt(
                text=(
                    "You are participating in an ongoing chat session. "
                    "For each user message, decide whether to schedule "
                    "events, send emails, or both using tools. Keep each "
                    "assistant reply concise and user-friendly."
                ),
            )
        )

        if config.system_note:
            self.history.add(SystemPrompt(text=config.system_note))

        provider = PROVIDER
        max_steps_per_turn = 8

        while True:
            # Wait for either a new message or a close request.
            await workflow.wait_condition(
                lambda: bool(self._pending_messages) or self._closed,
            )

            if self._closed and not self._pending_messages:
                workflow.logger.info("Chat session closed by client signal.")
                return

            # Pull the next user message and append it to the prompt
            # history as a user prompt.
            message = self._pending_messages.pop(0)
            self._turn_index += 1
            self.history.add(UserPrompt(text=message.text))

            last_text: str = ""

            for _ in range(max_steps_per_turn):
                # Convert history to provider-specific messages.
                messages = self.history.to_messages(provider=provider)
                step_input = AgentStepInput(messages=messages)

                raw_result = await workflow.execute_activity(
                    "llm_step_activity",
                    step_input,
                    schedule_to_close_timeout=timedelta(seconds=60),
                )

                if isinstance(raw_result, dict):
                    result = AgentStepOutput(**raw_result)
                else:
                    result = raw_result

                # Handle tool calls first, mirroring the one-shot workflow.
                if result.tool_call:
                    if result.tool_call.name == "company_research":
                        tool_response = await _run_company_research_subagent(
                            result.tool_call,
                        )
                    else:
                        tool_response = await workflow.execute_activity(
                            "tool_activity",
                            result.tool_call,
                            schedule_to_close_timeout=timedelta(seconds=30),
                        )

                    self.history.add(
                        UserPrompt(
                            text=f"[Tool {result.tool_call.name} output]: {tool_response}",
                        ),
                    )
                    workflow.logger.info(
                        "Chat turn %s tool %s executed with response: %s",
                        self._turn_index,
                        result.tool_call.name,
                        tool_response,
                    )
                    # Continue this turn so the LLM can see the tool output.
                    continue

                # Capture plain-text assistant output and finalize this turn
                # on the first non-tool assistant message.
                if result.output_text:
                    last_text = result.output_text
                    self.history.add(ModelPrompt(text=result.output_text))
                    break

            final_text = last_text or "The assistant could not produce a response."
            self._latest_response = ChatResponse(
                text=final_text,
                turn_index=self._turn_index,
            )


async def _run_company_research_subagent(tool_call: ToolCall) -> str:
    """Execute the company research sub-agent as a child workflow.

    The supervisor agent calls this helper when the LLM selects the
    `company_research` tool. The helper starts the
    `company_research_agent.AgentLoopWorkflow` child workflow, waits for
    it to finish, and returns a concise textual summary suitable for
    inclusion in the main conversation history.
    """
    args = tool_call.arguments or {}
    company = (
        args.get("company")
        or args.get("company_name")
        or args.get("query")
        or ""
    )

    if not company:
        return "company_research tool was called without a 'company' argument."

    child_input = CompanyResearchAgentInput(task=company)

    # Let Temporal choose a deterministic child workflow ID; we only
    # specify the workflow function and input.
    result: dict = await workflow.execute_child_workflow(
        AgentLoopWorkflow.run,
        child_input,
        task_queue="company-research-task-queue"
    )

    markdown = (result.get("markdown_report") or "").strip()
    if not markdown:
        return (
            "The company research agent completed without producing a "
            "report. No additional details are available."
        )

    # Limit the text we feed back into the supervisor history to keep
    # prompts reasonably sized, while still providing rich signal.
    max_chars = 2000
    if len(markdown) > max_chars:
        snippet = markdown[: max_chars - 3].rstrip() + "..."
    else:
        snippet = markdown

    return (
        "Company research report (markdown excerpt):\n\n"
        f"{snippet}"
    )


async def main() -> None:  # pragma: no cover
    """Convenience entry point for running the demo workflow once.

    This function assumes a worker is already running for the
    ``agent-task-queue`` task queue (see ``worker.py``). It is provided
    as a simple way to trigger the workflow without going through the
    higher-level CLI or notebook examples.
    """
    import asyncio  # noqa: PLC0415

    from temporalio.client import Client  # noqa: PLC0415
    from temporalio.contrib.pydantic import pydantic_data_converter  # noqa: PLC0415

    client = await Client.connect(ADDRESS, data_converter=pydantic_data_converter)

    input_data = PersonalAssistantInput(
        query="Schedule a team standup for tomorrow at 9am and email the team.",
    )
    result = await client.execute_workflow(
        PersonalAssistantWorkflow.run,
        input_data,
        id="multi-agent-personal-assistant",
        task_queue=TASK_QUEUE,
    )

    print("\nFinal assistant response:\n")  # noqa: T201
    print(result.final_response)  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(main())
