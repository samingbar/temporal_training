"""Agent-style workflow that assembles LLM prompts using the shared `myprompts` package."""

from __future__ import annotations

import asyncio

from temporalio import workflow

from .agent_types import AgentStepInput, AgentStepOutput
from pydantic import BaseModel
from src.resources.myprompts.models import SystemPrompt, UserPrompt, TaskPrompt, ModelPrompt
from src.resources.myprompts.history import PromptHistory
from src.workflows.agents.MAKER_test.config import PROVIDER
import json
from datetime import timedelta

SYSTEM_PROMPT = """ You are an expert that specializes in adding 1 to a provided number. Your only task is to output a JSON object containing n + 1.Output ONLY valid JSON. No quotes, no markdown, no backticks, no explanations."""
TASK_PROMPT = """ You are given an integer n and optionally conversation history. 
                        Output ONLY: {"result": <n_plus_one>}
                        Do NOT add explanations or any surrounding text."""
ACTION_PROMPT = """{n}"""
@workflow.defn
class NormalAgentWorkflow:
    def __init__(self):
        self.history = PromptHistory()
        self.step = 0

    @workflow.run
    async def run(self):
        """Baseline agent that calls the LLM step-by-step and checks n+1 correctness."""
        # Build initial history for n=0
        n = 0
        self.history.add(SystemPrompt(text=SYSTEM_PROMPT.strip()))
        self.history.add(TaskPrompt(text=TASK_PROMPT.strip()))
        self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))
        struct_hist = self.history.to_messages(provider=PROVIDER)

        next_input = AgentStepInput(messages=struct_hist)
        x = 0
        while x == n:
            response = await workflow.execute_activity(
                "llm_step_activity",
                next_input,
                schedule_to_close_timeout=timedelta(seconds=60),
            )
            # Temporal + data converter may deserialize to dict; normalize.
            if isinstance(response, dict):
                response = AgentStepOutput(**response)

            msg = response.output_text or ""

            # Parse the JSON {"result": <n_plus_one>}
            try:
                payload = json.loads(msg)
                raw_val = payload["result"]
                # Normalize to an integer so we can compare to x.
                if isinstance(raw_val, bool):
                    raise ValueError("Boolean is not a valid numeric result")
                if isinstance(raw_val, (int, float)):
                    candidate = int(raw_val)
                elif isinstance(raw_val, str):
                    candidate = int(raw_val.strip())
                else:
                    raise ValueError(f"Non-numeric result type: {type(raw_val)}")
            except Exception:  # noqa: BLE001
                workflow.logger.error("Parse failure for step %s: %s", x, msg)
                break

            # Append model message to history
            n = candidate
            self.history.add(ModelPrompt(text=msg))
            self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))
            struct_hist = self.history.to_messages(provider=PROVIDER)
            workflow.logger.info("Step %s: model returned %s", x, candidate)

            # Prepare the next step
            next_input = AgentStepInput(messages=struct_hist)

            # Update n with the candidate; loop condition x == n will break
            # once the model stops returning strict n+1 increments.
            
            x += 1

        workflow.logger.info(f"Failed after {x} steps. Incorrect result = {n}")
        return x

@workflow.defn
class MakerWorkflow:
    """Workflow that repeatedly adds 1 to a number until it renders an incorrect result"""

    def __init__(self):
        self.history = PromptHistory()


    @workflow.run
    async def run(self):  
        """Build a provider-specific message list for the given task."""

        # Build initial history for n=0
        n = 0
        self.history.add(SystemPrompt(text=SYSTEM_PROMPT.strip()))
        self.history.add(TaskPrompt(text=TASK_PROMPT.strip()))
        self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))
        struct_hist = self.history.to_messages(provider=PROVIDER)

        # Set up loop inputs
        next_input = AgentStepInput(messages=struct_hist)
        x = 0
        k = 3 
       
        # Loop as long as the MAKER consensus equals the expected n+1
        while x == n:
            workflow.logger.info("---- MAKER Step %s, input n=%s ----", x, n)

            # (A) Run MAKER voting loop for this step
            decided_value = await self._maker_vote(next_input, k)

            # (B) Check correctness (task expects n+1)
            if decided_value != n + 1:
                workflow.logger.error(
                    "Incorrect result! Expected %s, got %s",
                    n + 1,
                    decided_value,
                )
                return {
                    "last_correct_step": x,
                    "failed_with": decided_value,
                    "expected": n + 1,
                }

            # (C) Update state
            n = decided_value
            x += 1

            # (D) Rebuild history fresh for the new n
            self.history.reset()
            self.history.add(SystemPrompt(text=SYSTEM_PROMPT.strip()))
            self.history.add(TaskPrompt(text=TASK_PROMPT.strip()))
            self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))

            next_input = AgentStepInput(
                messages=self.history.to_messages(provider=PROVIDER),
            )

        # If we ever fall out of the loop without a mismatch,
        # report how far we progressed.
        return {
            "status": "completed_loop_exit",
            "last_n": n,
            "last_step": x,
        }
    
    async def _maker_vote(self, next_input: AgentStepInput, k: int) -> int:
        vote_counts = {}

        while True:
            # --------------------------------------------------------
            # Start a batch of activities (parallel micro-agents)
            # --------------------------------------------------------
            batch = []
            for _ in range(5):
                fut = workflow.start_activity(
                    "llm_step_activity",
                    next_input,
                    schedule_to_close_timeout=timedelta(seconds=20),
                )
                batch.append(fut)

            batch_results = [await item for item in batch]

            # --------------------------------------------------------
            # Process results
            # --------------------------------------------------------
            for raw in batch_results:
                if isinstance(raw, dict):
                    raw = AgentStepOutput(**raw)
                print(raw)
                msg = raw.output_text

                # Red-flag check
                if self.red_flag(msg):
                    workflow.logger.info(f"RED-FLAGGED: {msg}")
                    continue

                # Safe parse
                try:
                    js = json.loads(msg)
                    raw_val = js["result"]
                    # Normalize to an integer for voting/comparison.
                    if isinstance(raw_val, bool):
                        raise ValueError("Boolean is not a valid numeric result")
                    if isinstance(raw_val, (int, float)):
                        val = int(raw_val)
                    elif isinstance(raw_val, str):
                        val = int(raw_val.strip())
                    else:
                        raise ValueError(f"Non-numeric result type: {type(raw_val)}")
                except Exception:
                    workflow.logger.info(f"PARSE FAIL (red-flag implicitly): {msg}")
                    continue

                # Count vote
                vote_counts[val] = vote_counts.get(val, 0) + 1

                # MAKER first-to-be-ahead-by-k rule
                current = vote_counts[val]
                max_other = max(
                    (ct for v2, ct in vote_counts.items() if v2 != val),
                    default=0,
                )

                if current >= max_other + k:
                    workflow.logger.info(
                        f"VOTED: {val} with votes={vote_counts}"
                    )
                    return val
            
    def red_flag(self, message: str) -> bool:
        """
        Very simple MAKER-style red flagging:
        - too long
        - not JSON
        - contains explanation (heuristic)
        """

        # 1. Length-based red flag (very strong signal in paper)
        if len(message) > 120:
            return True

        # 2. Format must look like JSON
        if not message.strip().startswith("{"):
            return True

        # 3. Contain forbidden text
        lowered = message.lower()
        if "explain" in lowered or "because" in lowered:
            return True

        # 4. Validate JSON
        try:
            js = json.loads(message)
            return "result" not in js
        except:
            return True

    
  
