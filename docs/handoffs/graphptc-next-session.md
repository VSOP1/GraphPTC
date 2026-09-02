# GraphPTC 新会话交接

## 开始顺序

1. 阅读根目录 `AGENTS.md`、`README.md` 和 `docs/benchmark-results.md`。
2. 查看 `git status` 与当前 diff，不回滚未知改动。
3. 根据任务阅读 `docs/benchmarks/<name>.md` 和相应配置。
4. 使用 `.\.venv\Scripts\python.exe` 与 `.\.venv\Scripts\graphptc.exe`。

## 当前代码入口

- Agent：`src/graphptc/agents/`
- 持久执行与 telemetry：`src/graphptc/runtime/`
- Research Graph：`src/graphptc/graph/`
- 检索：`src/graphptc/retrieval/`
- Benchmark adapter：`src/graphptc/benchmarks/<name>/`
- CLI：`src/graphptc/cli/`
- 外部准备和 launcher：`scripts/<benchmark>/`

BrowseComp-Plus 中的 `source_dataset.py` 负责下载、校验和解密其上游问题/答案数据。

## 评测边界

- 当前本地主要指标 lead：BrowseComp-Plus、AppWorld、ToolSandbox、Agent-Diff、FanOutQA、FRAMES。
- 完整但无总体 lead：ALFWorld、APIFlow、ToolHop、InterCode。
- DeepPlanning adapter 可用，但当前没有保留可汇总的全量结果。
- 历史 `runs/` 只读；源码重构后的新运行必须使用新输出目录。

## 修改纪律

- 不原地修改已评测配置、manifest、数据选择或阈值。
- 不把 graph 写入、结构成功或工具调用减少表述为 benchmark 提升。
- 不把失败日志等同于在线修复；声称 repair 必须有 `failure -> PATCH -> re-execution` 证据。
- 修改模块路径时同步更新 import、worker command、实现哈希源文件、脚本、文档和测试。
- 付费运行前先通过对应 `inspect-*` / `probe-*`，并核对 agent 与 grader 凭据。

## 基线验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\graphptc.exe --help
```
