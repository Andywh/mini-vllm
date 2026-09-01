"""调度器 — 对应 vLLM Scheduler.schedule / update_from_output（极简）。"""

from __future__ import annotations

from dataclasses import dataclass

from request import MiniRequest, RequestStatus


@dataclass
class SchedulerOutput:
    """本 step 要跑的请求及各自分到的 token 数。"""

    num_scheduled_tokens: dict[str, int]


@dataclass
class ModelOutput:
    """对应 ModelRunnerOutput.sampled_token_ids（简化）。"""

    sampled_token_ids: dict[str, list[int]]


class MiniScheduler:
    def __init__(self) -> None:
        self.requests: dict[str, MiniRequest] = {}
        self.waiting: list[MiniRequest] = []
        self.running: list[MiniRequest] = []

    def add_request(self, request: MiniRequest) -> None:
        """对应 scheduler.add_request → waiting。"""
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def has_unfinished_requests(self) -> bool:
        return any(not r.is_finished() for r in self.requests.values())

    def schedule(self) -> SchedulerOutput:
        """
        阶段 1：最多调度 1 个请求，本步 1 个 token。

        阶段 2 TODO：waiting 里多个请求，仍然每步只选 1 个。
        阶段 3 TODO：一步里给多个 running 请求各分 1 个 decode token。
        """
        if self.waiting and not self.running:
            req = self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            if req.num_computed_tokens == 0:
                req.num_computed_tokens = len(req.prompt_token_ids)
            self.running.append(req)

        num_scheduled: dict[str, int] = {}
        for req in list(self.running):
            if req.is_finished():
                continue
            num_scheduled[req.request_id] = 1
            break  # 阶段 1/2：一步只跑一个请求

        return SchedulerOutput(num_scheduled_tokens=num_scheduled)

    def update_from_output(self, model_output: ModelOutput) -> dict[str, list[int]]:
        """把采样结果写回 Request，判停，维护 running。"""
        emitted: dict[str, list[int]] = {}
        still_running: list[MiniRequest] = []

        for req in self.running:
            new_ids = model_output.sampled_token_ids.get(req.request_id, [])
            for tid in new_ids:
                req.append_output(tid)
                req.num_computed_tokens += 1
            if new_ids:
                emitted[req.request_id] = new_ids
            if not req.is_finished():
                still_running.append(req)

        self.running = still_running
        return emitted
