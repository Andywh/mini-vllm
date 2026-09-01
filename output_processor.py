"""输出处理 — 对应 OutputProcessor.process_outputs（极简 detokenize）。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MiniRequestOutput:
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finished: bool = False


class MiniOutputProcessor:
    """真 vLLM 用 tokenizer decode；这里用 T{id} 拼接。"""

    def __init__(self) -> None:
        self._states: dict[str, MiniRequestOutput] = {}

    def add_request(self, request_id: str, prompt_token_ids: list[int]) -> None:
        """对应 output_processor.add_request：前端建档。"""
        self._states[request_id] = MiniRequestOutput(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
        )

    def process_outputs(
        self,
        new_tokens: dict[str, list[int]],
        finished_ids: set[str],
    ) -> list[MiniRequestOutput]:
        outputs: list[MiniRequestOutput] = []
        for req_id, tids in new_tokens.items():
            st = self._states[req_id]
            st.token_ids.extend(tids)
            st.text += "".join(f"T{t}" for t in tids)
            st.finished = req_id in finished_ids
            outputs.append(
                MiniRequestOutput(
                    request_id=st.request_id,
                    prompt_token_ids=list(st.prompt_token_ids),
                    token_ids=list(st.token_ids),
                    text=st.text,
                    finished=st.finished,
                )
            )
        return outputs
