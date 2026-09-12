"""调度器 — 对应 vLLM Scheduler.schedule / update_from_output（极简）。"""

from __future__ import annotations

from dataclasses import dataclass, field

from kv_cache import MiniKVCacheManager
from request import MiniRequest, RequestStatus


@dataclass
class SchedulerOutput:
    """本 step 要跑的请求及各自分到的 token 数。"""

    num_scheduled_tokens: dict[str, int]
    # 本步新分配的物理块（教学用，真 vLLM 也会带 new blocks 信息）
    new_blocks: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ModelOutput:
    """对应 ModelRunnerOutput.sampled_token_ids（简化）。"""

    sampled_token_ids: dict[str, list[int]]


class MiniScheduler:
    def __init__(self, kv_cache_manager: MiniKVCacheManager | None = None) -> None:
        self.requests: dict[str, MiniRequest] = {}
        self.waiting: list[MiniRequest] = []
        self.running: list[MiniRequest] = []
        # 对应真 vLLM：scheduler 持有 kv_cache_manager
        self.kv_cache_manager = kv_cache_manager or MiniKVCacheManager(
            num_gpu_blocks=32, block_size=4
        )

    def add_request(self, request: MiniRequest) -> None:
        """对应 scheduler.add_request → waiting。"""
        self.requests[request.request_id] = request
        self.waiting.append(request)

    def has_unfinished_requests(self) -> bool:
        return any(not r.is_finished() for r in self.requests.values())

    def schedule(self) -> SchedulerOutput:
        """
        阶段 1：最多调度 1 个请求，本步 1 个 token。
        调度时按需 allocate_slots（对应真 vLLM schedule 里调 KVCacheManager）。
        """
        new_blocks: dict[str, list[int]] = {}

        if self.waiting and not self.running:
            req = self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            if req.num_computed_tokens == 0:
                # 假 prefill：prompt 视为已算完，同时为 prompt 分配 KV blocks
                req.num_computed_tokens = len(req.prompt_token_ids)
                allocated = self.kv_cache_manager.allocate_slots(
                    req.request_id, req.num_computed_tokens
                )
                if allocated:
                    new_blocks[req.request_id] = allocated
            self.running.append(req)

        num_scheduled: dict[str, int] = {}
        for req in list(self.running):
            if req.is_finished():
                continue
            # 本步还要再算 1 个 decode token → KV 需要覆盖到 computed+1
            allocated = self.kv_cache_manager.allocate_slots(
                req.request_id, req.num_computed_tokens + 1
            )
            if allocated:
                new_blocks.setdefault(req.request_id, []).extend(allocated)
            num_scheduled[req.request_id] = 1
            break  # 阶段 1/2：一步只跑一个请求

        return SchedulerOutput(
            num_scheduled_tokens=num_scheduled, new_blocks=new_blocks
        )

    def update_from_output(self, model_output: ModelOutput) -> dict[str, list[int]]:
        """把采样结果写回 Request，判停，维护 running；结束则 free KV。"""
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
            else:
                # 对应真 vLLM：finished 后归还 blocks
                self.kv_cache_manager.free(req.request_id)

        self.running = still_running
        return emitted
