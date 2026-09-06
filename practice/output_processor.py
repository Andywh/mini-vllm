from dataclasses import dataclass, field


@dataclass
class MiniRequestOutput:
    # 哪个请求
    request_id: str
    # 输入 prompt（编码后）
    prompt_token_ids: list[int]
    # 累计生成的 output token（会一步步变长)
    token_ids: list[int] = field(default_factory=list)
    # 假 detokenize 后的文本
    text: str = ""
    # 是否整段生成结束
    finished: bool = False

class MiniOutputProcessor:

    def __init__(self):
        self._states: dict[str, MiniRequestOutput] = {}

    def add_request(self, request_id: str, prompt_token_ids: list[int]) -> None:
        self._states[request_id] = MiniRequestOutput(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
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

