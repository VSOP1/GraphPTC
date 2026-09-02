# GraphPTC

GraphPTC 是一个面向深度检索与工具使用任务的 Programmatic Tool Calling（PTC）评测仓库。
模型通过 `programmatic_tool_call` 在持久 Python 运行时中组织多次工具调用；GraphPTC 在相同
Agent 主循环上维护执行与依赖图，用于记录任务进展、工具效果和后续适应动作。

本仓库的重点不是提供一个通用聊天框架，而是让 GraphPTC 与 matched Fewshot PTC 在冻结数据、
相同模型、相同预算和相同评分器下进行可复现比较。历史配置、响应、评分、报告和运行签名应视为
评测证据，不应原地改写。

项目的论文方法主线、创新点和组件边界见
[GraphPTC 创新点与方法总结](docs/architecture/GraphPTC创新点与方法总结.md)。

## 当前评测结论

“领先”仅表示本仓库内 GraphPTC 相对 matched Fewshot PTC 的本地结果，不代表外部排行榜 SOTA。

| Benchmark | 范围与主要指标 | GraphPTC | Fewshot PTC | 差值 |
| --- | --- | ---: | ---: | ---: |
| BrowseComp-Plus | 全部 830 题，accuracy | 35.66% | 29.40% | +6.27 pp |
| AppWorld test-normal | 168 tasks，TGC / SGC | 78.0 / 67.9 | 67.3 / 41.1 | +10.7 / +26.8 pp |
| AppWorld test-challenge | 417 tasks，TGC / SGC | 69.8 / 54.0 | 52.5 / 30.2 | +17.3 / +23.8 pp |
| ToolSandbox | 1,032 scenarios，official similarity | 74.16 | 43.67 | +30.48 pp |
| Agent-Diff | 224 tasks × 3 trials，pass rate | 67.86% | 67.11% | +0.74 pp |
| FanOutQA | dev 310，official-local loose | 72.13% | 68.09% | +4.04 pp |
| FRAMES | test 824，paper-style judge | 69.54% | 66.38% | +3.16 pp |

ALFWorld、APIFlow-Bench、ToolHop 和 InterCode 已完成 paired 全量评测，但 GraphPTC 没有取得
总体领先。DeepPlanning adapter 仍可使用，但当前没有保留可汇总的全量结果。
完整指标、次要指标、限制和证据路径见
[评测结果汇总](docs/benchmark-results.md)。

## 目录结构

```text
GraphPTC/
├── configs/                     # 按 benchmark 保存冻结配置
├── data/                        # 小型数据、选择清单和 provenance；大型数据通常忽略
├── docs/
│   ├── architecture/            # 项目定位与架构文档
│   ├── benchmarks/              # benchmark 安装、环境和评测说明
│   ├── development/             # 研究日志
│   └── handoffs/                # 会话与交接说明
├── external/                    # 服务器隔离环境；由 setup 脚本生成，不入 Git
├── infra/                       # 外部 retriever、Docker 和隔离环境依赖
├── runs/                        # 本地响应、评分、报告、图和日志；默认不入 Git
├── scripts/                     # 按 benchmark 组织的数据、服务、运行和诊断脚本
├── src/graphptc/
│   ├── cli/                     # 参数解析与命令分发
│   ├── agents/                  # Original PTC、CodeAct 和 direct-tools Agent
│   ├── runtime/                 # 持久运行时、worker、telemetry 和 JSONL 工具
│   ├── graph/                   # 执行图、投影、适应策略和工具效果
│   ├── retrieval/               # 联网搜索与本地语料检索
│   └── benchmarks/              # 每个 benchmark 的 adapter/runtime/worker/prompt
└── tests/                       # 与源码职责和 benchmark 结构对应
```

各目录说明：

- [configs](configs/README.md)
- [data](data/README.md)
- [docs](docs/README.md)
- [infra](infra/README.md)
- [runs](runs/README.md)
- [scripts](scripts/README.md)

## 架构边界

```text
benchmark task
  → benchmark adapter
  ├─ GraphPTC / Fewshot PTC → OriginalPTCAgent / CodeActPTCAgent
  │  → persistent program runtime → execution events and Research Graph
  └─ Direct Tool Calling → DirectToolAgent → native benchmark functions
  → benchmark tools or isolated official worker
  → prediction, official/local evaluator, report
```

- `agents/` 分别实现 PTC 协议和原生 function-calling 协议，不包含 benchmark 数据或评分逻辑。
- `runtime/` 执行生成代码并保持单个任务内的 Python 状态。
- `graph/` 记录依赖、效果和诊断；图写入或结构成功不等于 benchmark 提升。
- `benchmarks/` 只处理任务加载、官方环境桥接、结果持久化和评分。
- 外部 benchmark worker 属于受信评测环境，不是生产安全沙箱。

### Direct Tool Calling baseline

`DirectToolAgent` 是 benchmark 无关的原生 function-calling 循环。它只接收模型、运行预算、系统与
任务提示、`工具名 → Python callable` 映射、对应的 OpenAI function tool schema，以及可选的最终
回答提示。核心不依赖 `search`、`fetch` 或某个 benchmark runtime，并统一记录工具名、成功状态、
错误类型、耗时和 observation 大小。

benchmark adapter 仍负责工具/runtime 的创建与关闭、任务提示、最终答案解析、官方评分和领域指标。
BrowseComp-Plus、AppWorld、ToolSandbox、Agent-Diff、FanOutQA 和 FRAMES 均已接入该 baseline。
其中 ToolSandbox 为保留官方多角色对话和动态工具 schema，在官方 worker 内复用相同的原生
function-calling 协议；其余五项直接使用通用 `DirectToolAgent` 主循环。
这些 direct 配置用于老师后续重测，仓库当前没有把它们标记为已有正式结果。

## 安装

主环境要求 Python 3.11+。老师的 Linux 服务器建议直接执行：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,browsecomp-plus,agent-diff,fanoutqa]"
cp .env.example .env
.venv/bin/python -m graphptc --help
```

AppWorld、ToolSandbox 和 Agent-Diff 使用仓库下互相隔离的 `external/` 环境，可一次准备：

```bash
bash scripts/setup/server.sh
```

Git clone 不包含完整 benchmark 数据和本地 Wikipedia/BM25 服务。六项正式重测的官方来源、固定
revision、下载命令、目标目录与验收标准统一见
[数据集下载与准备](docs/dataset-setup.md)；不要从非官方镜像补齐缺失文件。

Windows 仍可作为本地控制端：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,browsecomp-plus,agent-diff,fanoutqa]"
Copy-Item .env.example .env
```

检查入口：

```powershell
.\.venv\Scripts\graphptc.exe --help
.\.venv\Scripts\python.exe -m graphptc --help
```

主模型通过支持 function calling 的 OpenAI-compatible Chat Completions 接口调用。密钥只写入
本地 `.env`，配置中只保存环境变量名。非 OpenAI-compatible 的原生 Anthropic/Gemini 接口需要先
部署兼容网关；目标接口不支持 `thinking` 扩展时不要在 `[model]` 中填写该字段。

## 配置模型 API

每个 TOML 的 `[model]` 控制被评测模型：

```toml
[model]
model = "your-model-id"
base_url = "https://api.example.com/v1"
api_key_env = "TEACHER_MODEL_API_KEY"
```

`[grader]` 是独立评分器。只切换被评测模型时不要同时更换 grader；若确实更换，结果必须标记为
新的开发期评分协议，不能与当前表格直接比较。

不要原地修改已经产生历史结果的配置。使用统一入口创建 GraphPTC、Fewshot PTC 和 Direct Tool
Calling 三组 profile；生成的 21 份配置保存在 `runs/profiles/<profile>/configs/`，不会覆盖历史配置：

```bash
.venv/bin/python scripts/evaluation/full_suite.py create-profile \
  --profile teacher-model-v1 \
  --model your-model-id \
  --base-url https://api.example.com/v1 \
  --api-key-env TEACHER_MODEL_API_KEY
```

ToolSandbox 的 `[user_model]` 是冻结 user simulator，不会随 `[model]` 一起切换。仓库级 Codex 操作
规则见 [AGENTS.md](AGENTS.md)。

## 单个 benchmark 快速运行

先用对应的 `inspect-*` 或 `probe-*` 命令检查环境，再生成和评分。例如 FRAMES：

```powershell
.\.venv\Scripts\graphptc.exe inspect-frames `
  --config configs/frames/graphptc-test.toml
.\.venv\Scripts\graphptc.exe probe-frames-wikipedia `
  --config configs/frames/graphptc-test.toml
.\.venv\Scripts\graphptc.exe run-frames `
  --config configs/frames/graphptc-test.toml
.\.venv\Scripts\graphptc.exe evaluate-frames `
  --config configs/frames/graphptc-test.toml
```

再次执行相同命令会按各 adapter 的 checkpoint 规则继续。除非明确要创建一轮全新实验，否则不要
传入 `--restart`，也不要删除已成功任务或只重试挑选出的失败样本。

所有保留命令可通过以下方式查看：

```powershell
.\.venv\Scripts\graphptc.exe --help
```

## 全量三组评测

当前建议交给新模型重测的完整集合是六个已完成全量、且主要指标显示本地 GraphPTC lead 的
benchmark：BrowseComp-Plus、AppWorld、ToolSandbox、Agent-Diff、FanOutQA 和 FRAMES。

每个 benchmark 必须同时运行 GraphPTC、Fewshot PTC 与 Direct Tool Calling 配置。正式配置总表见
[configs/README.md](configs/README.md)。BrowseComp-Plus 直接使用
`data/browsecomp_plus/questions.jsonl` 完成单次 830 题评测，不再拆分 fold。Codex 在开始任何
付费请求前必须检查：

- agent API key 与冻结 grader key；
- BrowseComp-Plus retriever `/metadata`，不能只检查 `/health`；
- AppWorld、ToolSandbox 和 Agent-Diff 的隔离官方环境；
- FanOutQA 的固定 Kiwix Wikipedia；
- FRAMES 的固定 Pyserini/BM25 Wikipedia 快照；
- 新旧配置的任务数、数据选择、预算、prompt variant 和输出目录是否 matched。

具体配置组和执行纪律见 [AGENTS.md](AGENTS.md)。

若服务器是从 GitHub clone 开始准备，先按
[数据集下载与准备](docs/dataset-setup.md)完成六项数据和本地检索服务，再执行统一预检。

服务器数据和服务就绪后，统一预检不会发起付费 agent 请求：

```bash
.venv/bin/python scripts/evaluation/full_suite.py preflight --profile teacher-model-v1
.venv/bin/python scripts/evaluation/full_suite.py all --profile teacher-model-v1 --dry-run
```

确认 9 项预检和 21 组命令无误后启动完整生成与评分：

```bash
.venv/bin/python scripts/evaluation/full_suite.py all --profile teacher-model-v1
```

中断后重复同一命令即可按 adapter checkpoint 续跑，统一入口不会自动添加 `--restart`。完整服务器
交接步骤见 [服务器评测指南](docs/server-evaluation.md)。

## 外部环境文档

- [AppWorld](docs/benchmarks/appworld.md)
- [BrowseComp-Plus](docs/benchmarks/browsecomp_plus.md)
- [Agent-Diff](docs/benchmarks/agent_diff.md)
- [ALFWorld](docs/benchmarks/alfworld.md)
- [ToolSandbox](docs/benchmarks/toolsandbox.md)
- [FanOutQA](docs/benchmarks/fanoutqa.md)
- [FRAMES](docs/benchmarks/frames.md)
- [DeepPlanning](docs/benchmarks/deepplanning.md)

`scripts/<benchmark>/` 保存数据与服务脚本；`scripts/setup/`、`scripts/evaluation/` 和
`scripts/release/` 分别负责服务器环境、统一评测和安全打包。

## 复现纪律

- 保留配置、数据选择、hash、响应、评分、报告、checkpoint、图事件和运行签名。
- smoke、pilot、开发集、完整本地评测和官方 leaderboard 结果必须分开标注。
- `runs/` 只保留当前汇总采用的最终全量运行；开发期产物不进入交接包。
- GraphPTC 与对照分别报告准确率、执行失败、检索/工具调用、重复、时延和 token。
- 结构图生成、`GRAPH_DELTA` 或工具调用减少不能单独视为任务结果提升。
- 源码重构会产生新的 implementation hash；不要使用重构后的代码续跑旧 `runs/`。
- `.env`、`.mcp_env`、受限数据和虚拟环境不得进入交接包；`runs/` 只交付
  [最终结果清单](runs/README.md)中的目录。
- `runs/` 和大型本地数据默认被 Git 忽略，因此 `git archive` 或普通 clone 不是可直接全量运行的
  完整包。若老师需要开包即跑，应从工作区打包这些依赖，或按 benchmark 文档另行准备数据与官方环境。

## 安全打包

提交并确认工作树干净后执行：

```bash
.venv/bin/python scripts/release/build_package.py
```

命令会在 `dist/` 生成不含 `.env`、虚拟环境、缓存、外部环境和大型数据的 source ZIP、可直接
`git clone` 的 Git bundle 及 SHA-256 清单。只有明确需要交付约 5.8 GB 历史证据时才增加
`--include-results`。本地语料和官方环境体积约 142 GB，应按
[服务器评测指南](docs/server-evaluation.md)单独同步或重建，不能和源码盲目压成一个包。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

任何目录迁移都必须同步更新模块 import、配置中的 worker 路径、实现哈希源文件列表、脚本路径和
测试。仅能在完整测试、CLI 冒烟和路径扫描通过后交付。
