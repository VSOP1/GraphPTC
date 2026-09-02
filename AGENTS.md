# GraphPTC Codex 交接规则

当用户要求切换模型或启动评测时，Codex 必须遵守以下步骤。

## 开始前

1. 从仓库根目录工作，先阅读 `README.md`、`docs/server-evaluation.md`、
   `docs/benchmark-results.md` 和目标 benchmark 文档。
2. 获取模型 ID、OpenAI-compatible base URL、API key 的环境变量名和新的 profile 名称；不要请求、
   输出或写入明文密钥。
3. 检查 `git status`。不要覆盖用户未提交的改动，也不要修改历史 `runs/`。

## 新模型配置

1. 不得原地修改已经产生结果的 TOML。使用 `scripts/evaluation/full_suite.py create-profile`
   生成六个领先 benchmark 的 GraphPTC、Fewshot PTC 与 Direct Tool Calling 配置。
2. 三个 arm 使用相同模型、endpoint、温度、预算、数据、任务数和 grader。
3. 只修改 `[model]`、新的输出目录以及环境安装路径；其他差异必须在运行前报告。
4. 标准 OpenAI-compatible API 不要保留供应商专属 `thinking` 字段，除非目标接口明确支持。
5. 默认保留冻结 grader。若用户明确要求换 grader，使用新的结果标签并说明不可直接横向比较。
6. ToolSandbox 的 `[user_model]` 是冻结 user simulator；只切换 `[model]`，不得同步替换它。

## 默认完整重测集合

默认重测以下六个完整 benchmark 家族，并同时运行 GraphPTC、Fewshot PTC 与 Direct Tool Calling：

- BrowseComp-Plus：直接运行 `questions.jsonl` 中的完整 830 题，不拆分 split；
- AppWorld：test-normal、test-challenge；
- ToolSandbox：full 1,032 scenarios；
- Agent-Diff：224 tasks × 3 trials；
- FanOutQA：dev 310；
- FRAMES：test 824。

复制配置时以以下 matched 三组配置为模板：

| Benchmark | GraphPTC 模板 | Fewshot PTC 模板 | Direct Tool Calling 模板 |
| --- | --- | --- | --- |
| BrowseComp-Plus | `browsecomp_plus.graphptc-full.toml` | `browsecomp_plus.fewshot-ptc-full.toml` | `browsecomp_plus.direct-tools-full.toml` |
| AppWorld | `appworld.graphptc-test-{normal,challenge}.toml` | `appworld.fewshot-ptc-test-{normal,challenge}.toml` | `appworld.direct-tools-test-{normal,challenge}.toml` |
| ToolSandbox | `graphptc.toml` | `fewshot-ptc.toml` | `direct-tools.toml` |
| Agent-Diff | `graphptc.toml` | `fewshot-ptc.toml` | `direct-tools.toml` |
| FanOutQA | `graphptc-dev.toml` | `fewshot-ptc-dev.toml` | `direct-tools-dev.toml` |
| FRAMES | `graphptc-test.toml` | `fewshot-ptc-test.toml` | `direct-tools-test.toml` |

表中省略目录前缀，配置均位于 `configs/<benchmark>/`。BrowseComp-Plus 的历史结果曾按四个分片
生成，但新模型正式重测必须使用上表的单个 830 题配置，不能继续沿用历史分片流程。

ALFWorld、APIFlow、ToolHop 和 InterCode 虽有完整结果，但当前没有总体 GraphPTC lead；只有在用户
明确要求“全部已完成 benchmark”时加入。DeepPlanning 没有保留可汇总的全量结果，不得自动加入
完整重测集合或宣传为已完成评测。

## 预检与运行

1. 使用 `.venv/bin/python -m graphptc --help` 确认服务器命令入口。
2. 先执行 `full_suite.py preflight --profile <name>`，再执行 `all --dry-run` 并向用户报告 21 份配置。
3. BrowseComp-Plus 必须检查 retriever `/metadata`；`/health` 不能代替运行签名元数据检查。
4. 检查所有 agent、grader、user simulator 和 search API 环境变量。
5. 输出运行计划：配置组、任务数、输出目录、grader 和准确命令。
6. 预检通过后才允许发起付费模型请求。
7. 中断后重复同一命令续跑。不得擅自添加 `--restart`、删除 checkpoint、只重试选中的失败任务，
   或把新代码续接到历史运行目录。

## 打包

- 只在完整测试、CLI、TOML、路径和密钥扫描通过且 Git 工作树干净后运行
  `scripts/release/build_package.py`。
- 不得把 `.env`、`.mcp_env`、`.venv`、`external/`、缓存或大型本地数据加入源码包。
- 历史结果只有在用户明确要求时才使用 `--include-results` 单独打包。

## 报告

- “GraphPTC lead”只表示同模型 matched 本地比较，不表示外部 SOTA。
- 分开报告主要/次要指标、runner failure、工具与检索调用、重复、时延/token 和 graph 机制证据。
- `correct -> wrong` 是评分器状态变化，不是执行状态；独立轨迹漂移不能归因于 repair/reuse。
- 保存配置、数据与 manifest hash、响应、grade、report、checkpoint、graph artifact 和日志。
