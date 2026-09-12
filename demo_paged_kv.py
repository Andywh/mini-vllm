"""单独演示分页 KV：不跑完整 generate，只看 allocate / block_table / free。

运行：
  cd /Users/andy/Documents/zelda/mini-vllm
  python demo_paged_kv.py
"""

from __future__ import annotations

from kv_cache import MiniKVCacheManager


def main() -> None:
    # 物理池很小，便于观察；block_size=4 → 每 4 个 token 用 1 块
    kv = MiniKVCacheManager(num_gpu_blocks=8, block_size=4)
    print("=== 初始 ===")
    print(kv.debug_summary())

    # 假 prefill：prompt 有 5 个 token → 需要 2 个 block（4+1）
    new = kv.allocate_slots("1", num_tokens=5)
    print("\n=== allocate prompt 5 tokens ===")
    print(f"new blocks: {new}")
    print(kv.debug_summary("1"))
    print("含义: block_table=[p0,p1] 物理上可以不连续，逻辑上覆盖 0..4")

    # decode 再生成到第 8 个 token（index 0..7）→ 正好 2 块，不再分配
    new = kv.allocate_slots("1", num_tokens=8)
    print("\n=== grow to 8 tokens ===")
    print(f"new blocks: {new}  (应为空：2 块已够)")
    print(kv.debug_summary("1"))

    # 再到 9 → 需要第 3 块
    new = kv.allocate_slots("1", num_tokens=9)
    print("\n=== grow to 9 tokens ===")
    print(f"new blocks: {new}")
    print(kv.debug_summary("1"))

    # 另一请求同时占块
    kv.allocate_slots("2", num_tokens=4)
    print("\n=== second request 4 tokens ===")
    print(kv.debug_summary())

    kv.free("1")
    print("\n=== free req 1 ===")
    print(kv.debug_summary())

    kv.free("2")
    print("\n=== free req 2 ===")
    print(kv.debug_summary())


if __name__ == "__main__":
    main()
