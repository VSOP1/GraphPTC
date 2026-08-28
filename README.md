# GraphPTC Stage 0

本目录实现论文实验的第一阶段：在 DeepSearchQA 上建立可复现的 Original PTC
baseline。当前阶段不包含执行图、故障归因、patch、失效传播或选择性重执行。

## 实现边界

每道题由 Agent 自主决定生成多少个 PTC block：

1. MiMo 以 `tool_choice=auto` 接收唯一的直接工具
   `programmatic_tool_call`。
2. 每次调用提交一个自包含 Python 程序。
3. `search_web`、`search_web_batch`、`fetch_url` 和 `fetch_urls`
   只注册到 PTC runtime，不会出现在模型的 tools 列表中。
4. 程序可以连续调用这些工具，中间结果留在程序内，只有 stdout 返回模型。
5. 模型查看 stdout 后，自主决定继续生成 PTC block 或提交最终答案。

Agent 使用 Anthropic 公开复现实验中的最小问题模板：可以自由规划和多次搜索，
最终答案写入 `<result>` 标签。完整回复保留在执行记录中，只有最后一个非空
`<result>` 的内容作为 benchmark prediction。`tool_choice=auto` 允许 Agent 在不需要
搜索时直接作答，也允许按任务复杂度生成零个、一个或多个 PTC block。

`max_turns=100`、`max_ptc_blocks=100` 和 `max_tool_calls=1000` 是防失控的资源
上限，不是每题固定调用数。它们有意设置得远高于初始测试中的实际用量，避免
harness 过早强制收尾。每题另有 `task_timeout_seconds=3600` 的 wall-clock
保险预算。

实现采用薄封装方案：模型和 benchmark 适配由本项目实现，本地 Python
子进程、AST 检查和双向工具 IPC 复用固定版本的
`ToolRegistry 0.14.0 + codecell 0.2.1`。没有 Claude 依赖，也没有 Claude Code
container。

## 安装

需要 Python 3.11 或更高版本。在 PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

在本地 `.env` 中配置：

```dotenv
MIMO_API_KEY=...
TAVILY_API_KEY=...
```

`MIMO_API_KEY` 同时用于当前的 `mimo-v2.5` Agent 和迭代阶段 grader，
`TAVILY_API_KEY` 用于搜索与网页提取。

## 运行

下载并校验官方 Kaggle v4 数据：

```powershell
.\.venv\Scripts\graphptc.exe download-data
```

先运行少量样本验证 API 和日志：

```powershell
.\.venv\Scripts\graphptc.exe run --limit 3
```

确认后断点续跑完整 900 题：

```powershell
.\.venv\Scripts\graphptc.exe run
```

runner 默认跳过已经成功写入 `responses.jsonl` 的 example ID，并自动重试
所选范围内的失败记录。只有显式传入 `--restart` 才会清空整个响应文件并重跑
所选样本。每条记录包含模型、prompt、搜索、runtime、依赖版本和实现源码指纹；
配置或实现变化时会拒绝把新结果混入旧文件。

使用官方 judge prompt 和当前配置的 MiMo grader 评分：

```powershell
.\.venv\Scripts\graphptc.exe evaluate
```

默认输出位于 `runs/deepsearchqa/official-style/`，与早期 prompt 调试结果隔离：

- `responses.jsonl`：提取后的 `<result>` prediction、完整模型答案、成功/失败、
  每个 PTC 的代码和 stdout、token、模型请求数、搜索调用与耗时。
- `grades.jsonl`：官方逐题 precision、recall、F1 和判分详情。
- `report.json`：宏平均指标、无效样本计数、数据哈希和模型版本。

评分过程也支持断点恢复：每道题判完立即写入 `grades.jsonl`；相同预测、judge
模型和官方 prompt 对应的有效 grade 会被复用，缺失、API 错误或无效 JSON 会在下次
执行时单独重试。报告同时聚合成功/失败、PTC 数量、模型 token、搜索调用和各层耗时。

## 分数说明

项目不设置硬编码分数阈值。`evaluate` 正常输出宏平均 precision、recall、
F1 以及空响应、grader 错误和无效 JSON 数量，再结合相同配置下的样本表现判断
baseline 是否明显异常。

迭代阶段默认使用 `mimo-v2.5` 充当 grader，因此该分数适合开发回归，但不能冒充
DeepSearchQA 官方可比结果。正式复现实验时可将 `[grader]` 的 `provider` 改为
`gemini`、模型改为 `gemini-2.5-flash`，安装 `.[gemini]` 并配置
`GOOGLE_API_KEY`；judge prompt 和指标实现保持不变。

## 后续接入 GPT

模型层使用 OpenAI-compatible Chat Completions。接入 GPT 时只需更换
`[model]` 的 `model`、`base_url` 和 `api_key_env`；不使用 MiMo
thinking 扩展时删除 `thinking` 配置，PTC runtime 和评测代码无需修改。

## BrowseComp

BrowseComp 复用相同的官方式 Agent prompt 和 Original PTC runtime。官方加密
数据集只以密文保存在 `data/browse_comp_test_set.csv`；runner 使用每行 `canary`
在内存中解密问题和答案，不会把 ground truth 或 canary 写入响应记录。

```powershell
.\.venv\Scripts\graphptc.exe download-browsecomp
.\.venv\Scripts\graphptc.exe run-browsecomp --example-id 0 --restart
.\.venv\Scripts\graphptc.exe evaluate-browsecomp
```

BrowseComp 使用 Anthropic 复现代码公开的 A/B/C judge：`A` 为正确，`B` 为
错误，`C` 为弃答，最终指标为全部 1,266 题上的 accuracy。当前配置仍以
`mimo-v2.5` 作为开发期 grader，因此不能冒充 Anthropic 使用固定 Claude grader
得到的官方可比结果。

Anthropic 的 BrowseComp 配置使用服务端 compaction 和 3M task budget。MiMo 没有
对应协议，baseline v2 在完整 assistant/tool-result 边界使用 tools-disabled MiMo 摘要，
摘要成功后才删除旧历史，并单独记录压缩 token、延迟和失败。触发阈值与资源预算是
MiMo 配置，不照搬 Claude 参数，因此仍不是 Anthropic 精确复现。

## BrowseComp-Plus

BrowseComp-Plus 是当前 Agent 架构迭代的主 benchmark。它使用固定的 830 个问题和
100,195 篇本地语料，不调用 Tavily。baseline 使用冻结的官方 Pyserini/Lucene BM25、
Qwen tokenizer 512-token snippet、top-5 和独立本地 retriever 服务。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[browsecomp-plus,dev]"
.\.venv\Scripts\graphptc.exe download-browsecomp-plus
.\.venv\Scripts\graphptc.exe run-browsecomp-plus --example-id 769 --restart
.\.venv\Scripts\graphptc.exe evaluate-browsecomp-plus
```

## ALFWorld adapter

GraphPTC 与 matched Fewshot PTC baseline 已通过独立 worker 接入官方 ALFWorld 0.4.2
文本环境，并为 `valid_seen` / `valid_unseen` 提供成对配置。本地全量开发评测已经完成；
这些结果不是官方排行榜结果。环境隔离、对齐项、命令、结果和验证边界见
[`docs/alfworld-evaluation.md`](docs/alfworld-evaluation.md)。

Original PTC 的冻结 20 题 pilot 由 SHA-256 对 query ID 排序选取，排除早期诊断题；
选择依据、源数据哈希和题号保存在 `data/browsecomp_plus/pilot20.manifest.json`。对应
配置为 `configs/browsecomp_plus.original-ptc-v1-turn30-pilot20.toml`。该配置显式要求
20 条数据，不能用于 830 题全量评测；默认 `browsecomp_plus.example.toml` 指向全量输出。

数据固定到 `Tevatron/browsecomp-plus-corpus` revision
`b27b02bc3e45511b8b82a13e6f90ce761df726f6`，7 个 Parquet 分片均按官方 LFS SHA-256
校验。问题由 BrowseComp-Plus qrel 的一基 `query_id` 映射到官方 BrowseComp 加密数据，
避免重复下载包含相同文档的 2.78 GB query 数据。

当前 MiMo grader 只用于开发回归。完成 Original PTC baseline 与 GraphPTC 的本地全量
评测后，再切换官方 Qwen3-32B judge，并在 DeepSearchQA/BrowserComp 上对齐 Anthropic
官方复现配置。

## 安全限制

当前执行后端是本地子进程隔离，不是安全沙箱。AST 检查不能作为恶意代码防线，
子进程也可能继承宿主环境。本阶段只适合固定 benchmark 和受信模型输出，不应运行
不受信代码或暴露生产凭据。真正的安全隔离不属于本次“无 container”实现范围。
