"""请求状态 — 对应 vLLM scheduler 侧 Request（简化版）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class MiniRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int = 16
    # 已经「算过」的 token 数（含 prompt）
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    eos_token_id: int = 2

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    def append_output(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)
        if (
            token_id == self.eos_token_id
            or len(self.output_token_ids) >= self.max_tokens
        ):
            self.status = RequestStatus.FINISHED
