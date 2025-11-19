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

@workflow.defn
class NormalAgentWorkflow:
    def __init__(self):
        self.history = PromptHistory()
        self.step = 0

    @workflow.run
    async def run(self):
        
        #Define Prompt Templates
        SYSTEM_PROMPT = """ You are an expert that specializes in adding 1 to a provided number. You must always output ONLY valid JSON, with no explanation, no prose, and no additional fields. """
        TASK_PROMPT = """ You are given a single integer input (`n`) and optionally a conversation history.
                            Your ONLY task is to output a JSON object containing the value of n + 1.
                            The output MUST follow this exact format:
                            {"result": <n_plus_one>}
                            Examples:
                            Input: 0 → {"result": 1}
                            Input: 41 → {"result": 42}
                            Do NOT add explanations, text, or additional fields. Output only the JSON object. """
        ACTION_PROMPT = """{n}"""

        #Build Initial History
        self.history.add(SystemPrompt(text = SYSTEM_PROMPT.strip())) # Add system message
        self.history.add(TaskPrompt(text= TASK_PROMPT.strip())) # Add task 
        self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=0))) 
        struct_hist = self.history.to_messages(provider=PROVIDER) #build provider-formatted history

        next_input = AgentStepInput(messages=struct_hist)
        x = 0
        n = 0
        while x == n:
            response = await workflow.execute_activity(
                "llm_step_activity",
                next_input,          
                schedule_to_close_timeout=timedelta(seconds=60)
            )
            # Temporal + data converter may deserialize to dict; normalize.
            if isinstance(response, dict):
                response = AgentStepOutput(**response)
            # Grab the new value and set it to n; append to the front of the history
            self.history.add(ModelPrompt(text=response))
            struct_hist = self.history.to_messages(provider=PROVIDER)
            workflow.logger.info(f"Step {x}: Result = {n}")

            #Prepare the next step
            next_input = AgentStepInput(messages=struct_hist)

            n = response["result"]
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
        
        #Build Prompt Templates
        SYSTEM_PROMPT = """ You are an expert that specializes in adding 1 to a provided number. You must always output ONLY valid JSON, with no explanation, no prose, and no additional fields. """ 
        TASK_PROMPT = """ You are given a single integer input (`n`) and optionally a conversation history.
                            Your ONLY task is to output a JSON object containing the value of n + 1.
                            The output MUST follow this exact format:
                            {"result": <n_plus_one>}
                            Examples:
                            Input: 0 → {"result": 1}
                            Input: 41 → {"result": 42}
                            Do NOT add explanations, text, or additional fields. Output only the JSON object. """
        ACTION_PROMPT = """{n}"""

        #Build Initial History
        self.history.add(SystemPrompt(text = SYSTEM_PROMPT.strip())) # Add system message
        self.history.add(TaskPrompt(text= TASK_PROMPT.strip())) # Add task 
        self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=0))) 
        struct_hist = self.history.to_messages(provider=PROVIDER) #build provider-formatted history

        #Set Up Initial Loop Inputs
        next_input = AgentStepInput(messages=struct_hist)
        x = 0
        n = 0
        k = 3 
       
        workflow.logger.info(f"---- MAKER Step {x}, input n={n} ----")

        # --------------------------------------------------------
        # (A) Run MAKER voting loop for this step
        # --------------------------------------------------------
        decided_value = await self._maker_vote(next_input, k)

        # --------------------------------------------------------
        # (B) Check correctness (task expects n+1)
        # --------------------------------------------------------
        if decided_value != n + 1:
            workflow.logger.error(
                f"Incorrect result! Expected {n+1}, got {decided_value}"
            )
            return {
                "last_correct_step": x,
                "failed_with": decided_value,
                "expected": n + 1,
            }

        # --------------------------------------------------------
        # (C) Update state
        # --------------------------------------------------------
        n = decided_value
        x += 1

        # --------------------------------------------------------
        # (D) Rebuild history fresh (MAKER extreme decomposition)
        #     Avoid unbounded growth which increases drift risk
        # --------------------------------------------------------
        self.history.reset() 
        self.history.add(SystemPrompt(text=SYSTEM_PROMPT.strip()))
        self.history.add(TaskPrompt(text=TASK_PROMPT.strip()))
        self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))

        next_input = AgentStepInput(
            messages=self.history.to_messages(provider=PROVIDER)
        )

        # Should never hit this
        return {"status": "unexpected_exit"}
    
    async def _maker_vote(self, next_input: AgentStepInput, k: int) -> int:
        vote_counts = {}

        while True:
            # --------------------------------------------------------
            # Start a batch of activities (parallel micro-agents)
            # --------------------------------------------------------
            batch = []
            for _ in range(3):  # batch size B; tune freely
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
                    val = js["result"]
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

    
  
