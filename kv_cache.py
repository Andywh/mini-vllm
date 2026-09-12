"""分页 KV 管理（玩具版）— 对应 vLLM 的 BlockPool + KVCacheManager。

不做 CUDA Attention kernel；只演示：
  - 物理块池（BlockPool）
  - 每请求一张 block table（逻辑块下标 → 物理块 id）
  - 按需 allocate / free

对照真 vLLM：
  - vllm/v1/core/block_pool.py          → MiniBlockPool
  - vllm/v1/core/kv_cache_manager.py    → MiniKVCacheManager.allocate_slots
  - scheduler.schedule 里调用 allocate_slots
"""

from __future__ import annotations

from dataclasses import dataclass, field


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class MiniBlockPool:
    """物理 KV 块池。每个 block_id 代表一块固定大小的「显存页」。"""

    num_gpu_blocks: int
    free_block_ids: list[int] = field(init=False)
    # 引用计数：多请求共享同一物理块时用（prefix / parallel sample）；玩具里一般是 0/1
    ref_counts: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.free_block_ids = list(range(self.num_gpu_blocks))
        self.ref_counts = [0] * self.num_gpu_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def allocate_block(self) -> int:
        if not self.free_block_ids:
            raise RuntimeError("KV OOM: no free blocks in MiniBlockPool")
        block_id = self.free_block_ids.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def free_block(self, block_id: int) -> None:
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_block_ids.append(block_id)


@dataclass
class MiniKVCacheManager:
    """
    管理每个 request 的 block table，并按需从 BlockPool 要物理块。

    block_tables[req_id] = [phys0, phys1, ...]
      下标 i = 逻辑块号；值 = 物理块 id（可不连续）
    """

    num_gpu_blocks: int = 32
    block_size: int = 4  # 玩具用小块，便于观察；真 vLLM 常用 16
    block_pool: MiniBlockPool = field(init=False)
    block_tables: dict[str, list[int]] = field(default_factory=dict)
    # 该请求已经为多少 token 准备了 KV 槽位（通常跟 num_computed_tokens 对齐）
    num_cached_tokens: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.block_pool = MiniBlockPool(self.num_gpu_blocks)

    def get_block_table(self, request_id: str) -> list[int]:
        return list(self.block_tables.get(request_id, []))

    def allocate_slots(self, request_id: str, num_tokens: int) -> list[int]:
        """
        保证 request 至少能放下 num_tokens 个已计算 token 的 KV。

        对应真 vLLM：KVCacheManager.allocate_slots
        返回：本调用新分配的物理块 id 列表。
        """
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")

        if request_id not in self.block_tables:
            self.block_tables[request_id] = []
            self.num_cached_tokens[request_id] = 0

        current = self.num_cached_tokens[request_id]
        if num_tokens <= current:
            return []

        need_blocks = ceil_div(num_tokens, self.block_size)
        have_blocks = len(self.block_tables[request_id])
        new_block_ids: list[int] = []
        for _ in range(need_blocks - have_blocks):
            new_block_ids.append(self.block_pool.allocate_block())
            self.block_tables[request_id].append(new_block_ids[-1])

        self.num_cached_tokens[request_id] = num_tokens
        return new_block_ids

    def free(self, request_id: str) -> None:
        """请求结束：归还所有物理块。对应真 vLLM free。"""
        table = self.block_tables.pop(request_id, [])
        self.num_cached_tokens.pop(request_id, None)
        for block_id in table:
            self.block_pool.free_block(block_id)

    def debug_summary(self, request_id: str | None = None) -> str:
        lines = [
            f"pool: free={self.block_pool.num_free_blocks}/{self.num_gpu_blocks} "
            f"block_size={self.block_size}"
        ]
        ids = [request_id] if request_id else list(self.block_tables)
        for rid in ids:
            if rid is None:
                continue
            table = self.block_tables.get(rid, [])
            cached = self.num_cached_tokens.get(rid, 0)
            lines.append(
                f"  req={rid} cached_tokens={cached} "
                f"block_table={table}  # 逻辑下标→物理块"
            )
        return "\n".join(lines)
