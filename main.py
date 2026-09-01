"""入口：跑通阶段 1。"""

from llm import MiniLLM


def main() -> None:
    llm = MiniLLM()
    outputs = llm.generate([[10, 11, 12]], max_tokens=8)
    for o in outputs:
        print(f"id={o.request_id} finished={o.finished}")
        print(f"  tokens={o.token_ids}")
        print(f"  text={o.text!r}")


if __name__ == "__main__":
    main()
