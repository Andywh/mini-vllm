from dataclasses import dataclass

from request import MiniRequest
from request import RequestStatus

@dataclass
class SchedulerOutput:
    num_scheduled_tokens: dict[str, int]

@dataclass
class ModelOutput:
    sampled_token_ids: dict[str, list[int]]

class MiniScheduler:
    def __init__(self):
        self.requests: dict[str, MiniRequest] = {}
        self.waiting: list[MiniRequest] = []
        self.running: list[MiniRequest] = []

    def add_request(self, request: MiniRequest) -> None:
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def has_unfinished_request(self):
        return any(not r.is_finished() for r in self.requests.values())

    def schedule(self) -> SchedulerOutput:
        # 如果只有 waiting 队列里有请求
        # 就取出一个 waiting 队列里的请求，并且设置 reqest 状态为 RUNNING
        if self.waiting and not self.running:
            req = self.waiting.pop(0) # req = self.waiting[0]
            req.status = RequestStatus.RUNNING
            # num_computed_tokens 已经计算的 token
            # req.num_computed_tokens == 0 这条请求还没算过任何 token（刚入队）
            # 因为目前的代码，假设我们直接走 decode 了，
            # 所以这里就给 num_computed_tokens 赋值 prompt 长度
            if req.num_computed_tokens == 0:
                req.num_computed_tokens = len(req.prompt_token_ids)
            self.running.append(req)

        # 目前是第一阶段，阶段 1 的设计就是：每一步只调度 1 个请求、只算 1 个 token。
        num_scheduled: dict[str, int] = {}
        for req in list(self.running):
            if req.is_finished():
                continue
            num_scheduled[req.request_id] = 1
            break

        return SchedulerOutput(num_scheduled_tokens=num_scheduled)

    def update_from_output(self, model_output: ModelOutput) -> dict[str, list[int]]:
        ### 把采样结果写回 Request，判停，维护 running
        # 本步「发出去」的新 token
        # emitted 就是本步新产生的 token
        # 已在 output_token_ids 里的旧 token 不会再进 emitted；只装这一步刚 append 的。
        emitted: dict[str, list[int]] = {}
        # 本步结束后还没结束的请求，用来替换 self.running
        still_running: list[MiniRequest] = []

        for req in self.running:
            new_ids = model_output.sampled_token_ids.get(req.request_id, [])
            for tid in new_ids:
                req.append_output_token(tid)
                req.num_computed_tokens += 1
            if new_ids:
                emitted[req.request_id] = new_ids
            if not req.is_finished():
                still_running.append(req)

        self.running = still_running
        return emitted

