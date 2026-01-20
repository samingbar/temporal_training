from datetime import timedelta

from temporalio import workflow

# Import non-deterministic / external dependencies via the unsafe guard so
# Temporal can safely replay this workflow.
with workflow.unsafe.imports_passed_through():
    from src.workflows.agents.streaming_test.activity import stream_llm_activity
    from src.workflows.agents.streaming_test.types import StreamLLMRequest, LLMChunk


@workflow.defn
class TestWorkflow:
    def __init__(self):
        # request_id -> {"next_seq": int, "text": list[str], "done": bool}
        self.buffers = {}

    @workflow.signal
    def llm_chunk(self, chunk: LLMChunk) -> None:
        buf = self.buffers.setdefault(chunk.request_id, {"next_seq": 0, "text": [], "done": False})

        # idempotency / ordering: accept only the next expected sequence
        if chunk.seq != buf["next_seq"]:
            return

        buf["text"].append(chunk.text)
        buf["next_seq"] += 1
        if chunk.done:
            buf["done"] = True

    @workflow.query
    def get_llm_text(self, request_id: str) -> str:
        buf = self.buffers.get(request_id)
        return "".join(buf["text"]) if buf else ""

    @workflow.run
    async def run(self, prompt: str) -> str:
        info = workflow.info()
        request_id = info.workflow_id + ":" + info.run_id  # or generate your own stable id

        request = StreamLLMRequest(prompt=prompt, request_id=request_id)

        await workflow.execute_activity(
            stream_llm_activity,
            request,
            start_to_close_timeout=timedelta(seconds=60),
        )
        await workflow.wait_condition(lambda: self.buffers.get(request_id, {}).get("done", False))
        return self.get_llm_text(request_id)
