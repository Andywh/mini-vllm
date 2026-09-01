"""引擎 — 对应 LLMEngine + EngineCore 的合并简化版。"""

from __future__ import annotations

from model_runner import MiniModelRunner
from output_processor import MiniOutputProcessor, MiniRequestOutput
from request import MiniRequest
from scheduler import MiniScheduler


class MiniEngine:
    def __init__(self) -> None:
        self.scheduler = MiniScheduler()
        self.model_runner = MiniModelRunner()
        self.output_processor = MiniOutputProcessor()
        self._id = 0

    def add_request(self, prompt_token_ids: list[int], max_tokens: int = 16) -> str:
        """
        对应 LLMEngine.add_request：
          - output_processor 建档
          - scheduler 进 waiting
        """
        self._id += 1
        req_id = str(self._id)
        req = MiniRequest(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            max_tokens=max_tokens,
        )
        self.output_processor.add_request(req_id, prompt_token_ids)
        self.scheduler.add_request(req)
        return req_id

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished_requests()

    def step(self) -> list[MiniRequestOutput]:
        """
        schedule → execute_model → sample_tokens
          → update_from_output → process_outputs
        """
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output.num_scheduled_tokens:
            return []

        self.model_runner.bind_requests(self.scheduler.requests)
        self.model_runner.execute_model(scheduler_output)
        model_output = self.model_runner.sample_tokens()
        new_tokens = self.scheduler.update_from_output(model_output)
        finished = {
            rid
            for rid, req in self.scheduler.requests.items()
            if req.is_finished() and rid in new_tokens
        }
        return self.output_processor.process_outputs(new_tokens, finished)
