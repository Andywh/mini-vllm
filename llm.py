"""用户 API — 对应 vllm.LLM 的极简版。"""

from __future__ import annotations

from engine import MiniEngine
from output_processor import MiniRequestOutput


class MiniLLM:
    def __init__(self) -> None:
        self.engine = MiniEngine()

    def generate(
        self,
        prompt_token_ids_list: list[list[int]],
        max_tokens: int = 16,
    ) -> list[MiniRequestOutput]:
        """
        入队 → while step → 收集 finished。
        chat template 省略，直接收 token ids。
        """
        for ids in prompt_token_ids_list:
            self.engine.add_request(ids, max_tokens=max_tokens)

        results: list[MiniRequestOutput] = []
        while self.engine.has_unfinished_requests():
            for out in self.engine.step():
                if out.finished:
                    results.append(out)

        results.sort(key=lambda x: int(x.request_id))
        return results
