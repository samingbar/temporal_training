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
        while self.step == n:
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
            except Exception: 
                workflow.logger.error("Parse failure for step %s: %s", self.step, msg)
                break

            # Append model message to history
            n = candidate
            self.history.add(ModelPrompt(text=msg))
            self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))
            struct_hist = self.history.to_messages(provider=PROVIDER)
            workflow.logger.info("Step %s: model returned %s", self.step, candidate)

            # Prepare the next step
            next_input = AgentStepInput(messages=struct_hist)

            # Update n with the candidate; loop condition x == n will break
            # once the model stops returning strict n+1 increments.
            self.step += 1

        workflow.logger.info(f"Failed after {self.step} steps. Incorrect result = {n}")
        return self.step

@workflow.defn
class MakerWorkflow:
    """Workflow that repeatedly adds 1 to a number until it renders an incorrect result"""

    def __init__(self):
        self.history = PromptHistory()
        self.step = 0


    @workflow.run
    async def run(self):  
        """Build a provider-specific message list for the given task."""

        # Build initial prompt history
        n = 0
        self.history.add(SystemPrompt(text=SYSTEM_PROMPT.strip()))
        self.history.add(TaskPrompt(text=TASK_PROMPT.strip()))
        self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))
        struct_hist = self.history.to_messages(provider=PROVIDER)

        # Set up loop inputs
        next_input = AgentStepInput(messages=struct_hist)
        k = 3 # K is the 'win value' during response voting
       
        # Loop as long as the MAKER consensus equals the expected n+1
        while self.step == n:
            workflow.logger.info("MAKER Step %s, input n=%s", self.step, n)

            #1. Run MAKER voting loop for this step
            decided_value = await self._maker_vote(next_input, k)

            #2. Check correctness (task expects n+1)
            if decided_value != n + 1:
                workflow.logger.error(
                    "Incorrect result! Expected %s, got %s",
                    n + 1,
                    decided_value,
                )
                return {self.step}

            #3. Update state
            n = decided_value
            self.step += 1

            #4. Rebuild history fresh for the new n
            self.history.reset() #In MAKER, we reset the history and isolate the next operation into a tiny, atomic unit. 
            self.history.add(SystemPrompt(text=SYSTEM_PROMPT.strip()))
            self.history.add(TaskPrompt(text=TASK_PROMPT.strip()))
            self.history.add(UserPrompt(text=ACTION_PROMPT.format(n=n)))

            next_input = AgentStepInput(messages=self.history.to_messages(provider=PROVIDER))
        
        #Raise exception if we exit the loop in an unexpected way
        raise Exception("Unhandled exit of MAKER loop")
    
   
    async def _maker_vote(self, next_input: AgentStepInput, k: int) -> int: #Method for running MAKER voting rounds
        vote_counts = {}

        #Open a loop that will close when the win threshold is reached
        while True:
            
            #1. Start a batch of activities (parallel micro-agents)
            batch = []
            for _ in range(5):
                fut = workflow.start_activity(
                    "llm_step_activity",
                    next_input,
                    schedule_to_close_timeout=timedelta(seconds=20),
                )
                batch.append(fut)

            batch_results = [await item for item in batch]

           
            #2. Process the batch results
            for raw in batch_results:
                if isinstance(raw, dict):
                    raw = AgentStepOutput(**raw)
                print(raw)
                msg = raw.output_text

                # Red-flag check
                if self.red_flag(msg):
                    workflow.logger.info(f"RED-FLAGGED: {msg}")
                    continue

                # Safe parse -- this may be partly rendundant due to the red flag check
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

                # Count votes
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
   
    #Function for red-flag checking in the MAKER style
    def red_flag(self, message: str) -> bool:

        # 1. Length-based red flag (very strong signal in paper)
        if len(message) > 120:
            return True

        # 2. Format must look like JSON -- this can be adapted to more specific formatting checks, differentiating it from #4
        if not message.strip().startswith("{"):
            return True

        # 3. Contain forbidden text -- This can be extended to cover broader style & rambling tests. Usual tests will include lenght checks (exlcude long answers) and rambling checks (model based assessment)
        lowered = message.lower()
        if "explain" in lowered or "because" in lowered:
            return True

        # 4. Validate JSON -- Validate output is valid JSON
        try:
            js = json.loads(message)
            return "result" not in js
        except:
            return True

    
  
