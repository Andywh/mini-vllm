"""模型执行 — 对应 GPUModelRunner.execute_model + sample_tokens（mock）。"""

from __future__ import annotations

from request import MiniRequest
from scheduler import ModelOutput, SchedulerOutput


class MiniModelRunner:
    """
    阶段 1：不加载真模型。
    策略：每次生成 (last_token + 1)，生成满 3 个后吐 eos=2。
    """

    def __init__(self, eos_token_id: int = 2) -> None:
        self.eos_token_id = eos_token_id
        self._pending: SchedulerOutput | None = None
        self._requests: dict[str, MiniRequest] = {}

    def bind_requests(self, requests: dict[str, MiniRequest]) -> None:
        self._requests = requests

    def execute_model(self, scheduler_output: SchedulerOutput) -> None:
        """对应 execute_model：前向。这里只暂存调度结果。"""
        self._pending = scheduler_output

    def sample_tokens(self) -> ModelOutput:
        """对应 sample_tokens：从「logits」采样。"""
        assert self._pending is not None, "call execute_model first"
        sampled: dict[str, list[int]] = {}
        for req_id, n in self._pending.num_scheduled_tokens.items():
            req = self._requests[req_id]
            tokens = [self._mock_next_token(req) for _ in range(n)]
            sampled[req_id] = tokens
        self._pending = None
        return ModelOutput(sampled_token_ids=sampled)

    def _mock_next_token(self, req: MiniRequest) -> int:
        if len(req.output_token_ids) >= 3:
            return self.eos_token_id
        last = req.all_token_ids[-1] if req.all_token_ids else 0
        return last + 1
