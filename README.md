# mini-vllm

对照 vLLM offline 主链路的学习用迷你引擎。

## 对应关系

| 本项目 | vLLM |
|--------|------|
| `MiniLLM.generate` | `vllm.LLM` |
| `MiniEngine.step` | `LLMEngine.step` + `EngineCore.step` |
| `MiniScheduler` | `v1/core/sched/scheduler.py` |
| `MiniModelRunner` | `GPUModelRunner`（mock） |
| `MiniOutputProcessor` | `v1/engine/output_processor.py` |

## 目录（扁平，无子包）

```
mini-vllm/
  main.py              # 入口
  llm.py               # MiniLLM
  engine.py            # MiniEngine.step
  scheduler.py         # MiniScheduler
  model_runner.py      # MiniModelRunner
  output_processor.py  # MiniOutputProcessor
  request.py           # MiniRequest
```

## 阶段

1. **已实现**：单请求；每 step 1 token；mock 采样
2. **待填**：多请求 waiting，每步只调度 1 个
3. **待填**：一步 batch 多个 decode

## 运行

```bash
cd /Users/andy/Documents/zelda/mini-vllm
source .venv/bin/activate
python main.py
```

## 练习顺序

1. 跑通 `main.py`，读 `engine.py` 的 `step`
2. 改 `scheduler.py` 支持多 waiting（阶段 2）
3. 一步多个 running 各 1 token（阶段 3）
