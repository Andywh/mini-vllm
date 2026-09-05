from scheduler import SchedulerOutput, ModelOutput
from request import MiniRequest


class MiniModelRunner:

    def __init__(self, eos_token_id: int = 2) -> None:
        self.eos_token_id = eos_token_id
        self._pending: SchedulerOutput | None = None
        self._requests: dict[str, MiniRequest] = {}

    # 把 scheduler 的 requests 字典挂上来，
    # 采样时才能用 req_id 找到 MiniRequest（读 all_token_ids 等）
    def bind_requests(self, requests: dict[str, MiniRequest]) -> None:
        self._requests = requests

    # 对应「前向」。真系统跑模型；这里只把 SchedulerOutput 存进 _pending
    def execute_model(self, scheduler_output: SchedulerOutput) -> None:
        self._pending = scheduler_output

    # 按 _pending.num_scheduled_tokens，
    # 为每个 req 采 n 个 token，返回 ModelOutput
    def sample_tokens(self) -> ModelOutput:
        assert self._pending is not None, "call execute_model first"
        sampled: dict[str, list[int]] = {}
        # n 确实是「采几次」
        # 但当前 mock 只适合 n=1（阶段 1）
        # 若要支持 n=2 且得到递增 token，
        # 要在 mock 循环里临时更新，或采完一个就改本地副本——阶段 1 不必做
        for req_id, n in self._pending.num_scheduled_tokens.items():
            req = self._requests[req_id]
            tokens = [self._mock_next_token(req) for _ in range(n)]
            sampled[req_id] = tokens
        self._pending = None
        return ModelOutput(sampled_token_ids=sampled)

    def _mock_next_token(self, req: MiniRequest) -> int:
        # >= 3 可以改成 5、10；eos=2 也可以改成别的
        # 这个 output_token_ids 表示新生成的 token 列表
        if len(req.output_token_ids) >= 3:
            return req.eos_token_id
        last = req.all_token_ids[-1] if req.all_token_ids else 0
        return last + 1
