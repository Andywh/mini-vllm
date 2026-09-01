# practice — 自己仿写区

根目录是**参考实现**（能跑）。这里是空壳，自己填。

## 怎么练

1. 先尽量不看根目录实现
2. 按顺序填下面的 `NotImplementedError`
3. 在本目录跑：`python main.py`
4. 卡住再看根目录**同名文件的同一个函数**

## 填写顺序

1. `request.py` — `is_finished` / `append_output`（`all_token_ids` 已给参考）
2. `scheduler.py`
3. `model_runner.py`
4. `output_processor.py`
5. `engine.py`
6. `llm.py`
7. 跑通 `main.py`

## 阶段 1 目标

```bash
cd practice
python main.py
```

期望类似：`finished=True`，tokens 类似 `[13, 14, 15, 2]`。
