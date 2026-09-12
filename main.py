"""入口：跑通阶段 1，并打印 KV block_table（PagedAttention 管理侧）。"""

from llm import MiniLLM


def main() -> None:
    llm = MiniLLM()
    kv = llm.engine.scheduler.kv_cache_manager

    print("--- before generate ---")
    print(kv.debug_summary())

    outputs = llm.generate([[10, 11, 12]], max_tokens=8)
    for o in outputs:
        print(f"id={o.request_id} finished={o.finished}")
        print(f"  tokens={o.token_ids}")
        print(f"  text={o.text!r}")

    print("--- after generate (blocks should be freed) ---")
    print(kv.debug_summary())


if __name__ == "__main__":
    main()
