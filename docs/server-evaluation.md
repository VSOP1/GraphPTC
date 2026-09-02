# Linux 服务器交接与全量评测

本文是老师拿到仓库后的最短操作路径。默认正式集合包含 BrowseComp-Plus、AppWorld、
ToolSandbox、Agent-Diff、FanOutQA 和 FRAMES，并对每项运行 GraphPTC、Fewshot PTC、Direct Tool
Calling 三个 matched arm；AppWorld 包含 normal/challenge，因此一共生成 21 份配置。

## 1. 获取源码

推荐使用交付包中的 Git bundle，以保留源码版本：

```bash
git clone GraphPTC-<commit>.bundle GraphPTC
cd GraphPTC
```

source ZIP 适合直接解压；其中的 `RELEASE.json` 为无 `.git` 环境提供 commit provenance。两种包都
不会包含 `.env`、`.venv`、`external/`、本地大数据或历史 `runs/`。

## 2. 安装环境

```bash
bash scripts/setup/server.sh
cp .env.example .env
```

脚本创建主 `.venv`，并在 `external/{appworld,toolsandbox,agent_diff}/` 创建三个隔离环境。默认需要
`python3.11` 和 `python3.10`；可分别通过 `GRAPHPTC_PYTHON311`、`GRAPHPTC_PYTHON310` 指定命令。

在 `.env` 中填写：

- `TEACHER_MODEL_API_KEY`：新被评测模型；
- `MIMO_API_KEY`：冻结 grader，以及 ToolSandbox 的冻结 user simulator；
- `RAPID_API_KEY`：ToolSandbox 官方搜索场景；
- `AGENT_DIFF_API_KEY`、`AGENT_DIFF_BASE_URL`：Agent-Diff 服务。

不要把 `.env` 发回、提交或复制进结果包。

## 3. 准备大型数据与服务

源码包不包含约 142 GB 的本地检索数据。可以从原机器安全同步 `data/` 中对应正式文件，也可以按
官方来源重建。完整下载地址、固定 revision、服务器命令、目标目录和逐项验收标准见
[六项正式评测的数据下载与准备](dataset-setup.md)。开始前必须满足：

- BrowseComp-Plus：完整 830 题、加密上游 CSV、qrels、BM25 corpus/index，且 retriever
  `/metadata` 与配置一致；
- FanOutQA：固定 `wikipedia_en_all_nopic_2023-09` Kiwix ZIM 服务；
- FRAMES：固定 `wikipedia/20230601.en` Pyserini/BM25 服务；
- Agent-Diff：执行一次 `.venv/bin/python -m graphptc download-agent-diff` 下载冻结 224 题数据。

各 benchmark 的方法与运行边界仍见 `docs/benchmarks/`。AppWorld 下载数据受其官方许可约束，
不进入 GraphPTC 源码包。

## 4. 创建新模型 profile

```bash
.venv/bin/python scripts/evaluation/full_suite.py create-profile \
  --profile MODEL_PROFILE \
  --model MODEL_ID \
  --base-url https://provider.example/v1 \
  --api-key-env TEACHER_MODEL_API_KEY
```

profile 名称必须唯一。脚本从冻结模板生成 21 份配置，将输出隔离到
`runs/<benchmark>/MODEL_PROFILE/<arm>/`，同时为 AppWorld 使用新的 experiment name。它只修改
`[model]` 和输出位置，不改数据、预算、prompt 或 grader。只有供应商明确支持时才传
`--thinking MODE`。

ToolSandbox 的 `[user_model]` 会继续使用冻结配置，因此更换被评测模型不会改变对话用户角色。

## 5. 预检与启动

```bash
.venv/bin/python scripts/evaluation/full_suite.py preflight --profile MODEL_PROFILE
.venv/bin/python scripts/evaluation/full_suite.py all --profile MODEL_PROFILE --dry-run
.venv/bin/python scripts/evaluation/full_suite.py all --profile MODEL_PROFILE
```

`preflight` 检查环境变量、数据规模、外部 worker 和两个 Wikipedia 服务，不调用付费 agent。`all`
按固定顺序执行 21 个 generation 命令，再执行对应 evaluator。任一命令失败即停止；修复外部问题后
重复同一命令继续，不要添加 `--restart`、删除 checkpoint 或只重试挑出的失败样本。

如需分阶段执行，可分别使用 `preflight`、`run` 和 `evaluate` 子命令。

## 6. 结果交接

保留生成的 profile manifest、21 份 resolved 配置、响应、评分、报告、checkpoint、日志、graph
artifact 和运行签名。不要用新源码续接旧结果目录。历史结果的交付范围以 `runs/README.md` 为准。
