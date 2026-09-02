# 六项正式评测的数据下载与准备

本文是从一个不含大型数据的 Git clone 准备 GraphPTC 正式评测环境的统一入口。默认范围仅包括
BrowseComp-Plus、AppWorld、ToolSandbox、Agent-Diff、FanOutQA 和 FRAMES；每项完成后都必须运行
本文给出的 `inspect-*` 或 `probe-*` 命令，不能只凭“文件已下载”判断环境可用。

所有命令都从仓库根目录执行。不要把数据下载到配置之外的临时路径后静默修改 TOML；如需使用挂载盘，
应通过符号链接保持下表中的仓库路径不变，并在运行记录中保存真实路径、版本和文件数量。

## Codex 执行规则

当用户要求准备数据或启动全量评测时，Codex 应按以下顺序执行：

1. 阅读本文件、`docs/server-evaluation.md` 和目标 benchmark 文档。
2. 检查目标配置中的数据路径、revision、预期任务数和服务地址，不要自动改成最新版数据。
3. 只从下表所列官方来源下载；官方来源不可达时停止并报告，不使用来源不明的网盘或镜像代替。
4. 下载命令可以断点续传，但不得跳过 adapter 内置的数量、revision、hash 或 `/metadata` 检查。
5. 数据和服务全部通过单项检查后，再运行 `full_suite.py preflight`；预检通过前不得发起付费评测。
6. 不要把 `.env`、AppWorld 受保护数据、题目答案、Wikipedia 语料、索引或缓存提交到 Git。

建议为源码、外部环境和三套本地检索数据预留至少 160 GB 可用空间。FanOutQA 的 ZIM 和 FRAMES 的
TFRecord/JSON/Lucene 中间文件应放在 Linux 原生文件系统；不要放在 WSL 的 `/mnt/c` 或 `/mnt/d`。

## 来源与目标一览

| Benchmark | 官方来源 | 本仓库使用的位置 | 完整范围 |
| --- | --- | --- | ---: |
| BrowseComp-Plus | [官方仓库](https://github.com/texttron/BrowseComp-Plus)、[语料](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)、[BM25 索引](https://huggingface.co/datasets/Tevatron/browsecomp-plus-indexes) | `data/browsecomp_plus/` | 830 题、100,195 篇文档 |
| AppWorld | [官方仓库与下载说明](https://github.com/StonyBrookNLP/appworld) | `external/appworld/data/` | test-normal 168、test-challenge 417 |
| ToolSandbox | [Apple 官方仓库](https://github.com/apple/ToolSandbox) | `external/toolsandbox/` | 1,032 scenarios |
| Agent-Diff | [官方仓库](https://github.com/agent-diff-bench/agent-diff) | `data/agent_diff/` | 224 tasks × 3 trials |
| FanOutQA | [官方仓库](https://github.com/zhudotexe/fanoutqa)、[固定 Wikipedia ZIM](https://datasets.mechanus.zhu.codes/fanoutqa/wikipedia_en_all_nopic_2023-09.zim) | 问题随 Python 包安装；ZIM 建议放 `/var/lib/fanoutqa/` | dev 310 |
| FRAMES | [固定 Hugging Face revision](https://huggingface.co/datasets/google/frames-benchmark/tree/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef) | `data/frames/` | test 824、6,672,479 篇 Wikipedia 文档 |

## 0. 安装基础环境

Linux 服务器需要 Python 3.11、Python 3.10、Git、curl/wget 和足够磁盘空间。仓库统一脚本会安装
GraphPTC 主环境，并准备 AppWorld、ToolSandbox 与 Agent-Diff 的隔离环境：

```bash
bash scripts/setup/server.sh
```

该命令会下载 AppWorld 的官方受保护数据，但不会下载 BrowseComp-Plus、Agent-Diff 任务 JSONL、
FanOutQA ZIM 或 FRAMES Wikipedia。某个隔离环境已准备好时，也可以只执行对应的
`scripts/setup/<benchmark>.sh`。

## 1. BrowseComp-Plus

正式配置使用完整 830 题和固定本地 BM25 检索，不再拆分 fold。准备过程分为“问题/标注/语料”和
“正式 BM25 服务”两部分，二者都需要完成。

### 1.1 下载问题、qrels 和语料

```bash
.venv/bin/python -m graphptc download-browsecomp-plus \
  --config configs/browsecomp_plus/browsecomp_plus.graphptc-full.toml
```

该命令会：

- 下载 OpenAI 官方加密 `browse_comp_test_set.csv`；
- 从固定 BrowseComp-Plus revision 下载 gold/evidence qrels；
- 生成 `questions.jsonl` 中的完整 830 题；
- 下载固定的 7 个 Parquet 语料分片，并建立 adapter 的本地 SQLite 索引；
- 使用仓库内置期望值验证下载内容，不需要手工维护第二份 hash 清单。

主要输出应为：

```text
data/browsecomp_plus/
├── browse_comp_test_set.csv
├── questions.jsonl
├── qrel_golds.txt
├── qrel_evidence.txt
├── corpus_parquet/
└── corpus.sqlite3
```

### 1.2 下载正式 BM25 索引和 tokenizer

仓库的正式检索服务还需要官方预构建 BM25 索引和固定 Qwen3 tokenizer。创建独立环境：

```bash
sudo apt-get update
sudo apt-get install -y openjdk-21-jre-headless
python3.11 -m venv external/browsecomp_plus_retriever/.venv
external/browsecomp_plus_retriever/.venv/bin/python -m pip install --upgrade pip
external/browsecomp_plus_retriever/.venv/bin/python -m pip install \
  "pyserini==1.2.0" "transformers==4.53.2" "huggingface-hub[cli]"

external/browsecomp_plus_retriever/.venv/bin/hf download \
  Tevatron/browsecomp-plus-indexes \
  --repo-type dataset \
  --revision b3f37f70c33829eb09d04784a54277a31871fd63 \
  --include 'bm25/*' \
  --local-dir data/browsecomp_plus/official_indexes

external/browsecomp_plus_retriever/.venv/bin/hf download \
  Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --include 'tokenizer*' 'vocab*' 'merges.txt' 'config.json' \
  --local-dir data/browsecomp_plus/qwen3-tokenizer
```

在一个长期运行的终端或服务管理器中启动 retriever：

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"
external/browsecomp_plus_retriever/.venv/bin/python \
  scripts/browsecomp_plus/retriever.py \
  --index-path data/browsecomp_plus/official_indexes/bm25 \
  --tokenizer-path data/browsecomp_plus/qwen3-tokenizer \
  --index-revision b3f37f70c33829eb09d04784a54277a31871fd63 \
  --tokenizer-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --top-k 5 \
  --snippet-max-tokens 512
```

验收命令：

```bash
.venv/bin/python -m graphptc inspect-browsecomp-plus \
  --config configs/browsecomp_plus/browsecomp_plus.graphptc-full.toml
```

必须看到 830 题，并成功读取 retriever `/metadata`。`/health` 返回成功不能替代该检查。

## 2. AppWorld

AppWorld 的任务、数据库和 evaluator 数据由官方 CLI 下载，不应复制进 GraphPTC 的 `data/`：

```bash
bash scripts/setup/appworld.sh
```

脚本固定安装 `appworld==0.1.3.post1`，依次执行官方 `install`、`download data`、测试验证和任务验证，
并把根目录固定为 `external/appworld/`。不要改用当前 main 或最新版 PyPI 后继续复用历史配置。

验收两个正式 split：

```bash
.venv/bin/python -m graphptc inspect-appworld \
  --config configs/appworld/appworld.graphptc-test-normal.toml
.venv/bin/python -m graphptc inspect-appworld \
  --config configs/appworld/appworld.graphptc-test-challenge.toml
```

预期分别为 168 和 417 个任务。AppWorld 要求受保护数据及其衍生内容公开分发时保持加密；不要把
`external/appworld/data/`、原始任务、数据库或逐题 evaluator 输出提交到 GitHub。

## 3. ToolSandbox

ToolSandbox 没有需要另行下载的数据压缩包。全部 1,032 个 scenario 与 evaluator 定义包含在固定的
官方仓库 checkout 中：

```bash
bash scripts/setup/toolsandbox.sh
```

脚本会克隆到 `external/toolsandbox/`，检出 commit
`165848b9a78cead7ca7fe7c89c688b58e6501219`，创建 Python 3.10 隔离环境并安装兼容依赖。

验收命令：

```bash
.venv/bin/python -m graphptc inspect-toolsandbox \
  --config configs/toolsandbox/graphptc.toml
```

必须解析出 1,032 个 scenario。`RAPID_API_KEY` 是部分正式搜索场景的运行凭据，不是数据下载凭据；
ToolSandbox 的冻结 `[user_model]` 也不能在切换被评测模型时一起替换。

## 4. Agent-Diff

先安装隔离 SDK，再用 GraphPTC 内置命令从固定官方 commit 下载三份任务文件：

```bash
bash scripts/setup/agent_diff.sh
.venv/bin/python -m graphptc download-agent-diff \
  --config configs/agent_diff/graphptc.toml
```

输出目录为：

```text
data/agent_diff/
├── train.jsonl
├── test.jsonl
├── all_numbered.jsonl
└── manifest.json
```

下载器固定使用 commit `3bb9c40707df23d89e5dbc0e40c424ba38c69ff8`，验证每个文件并确认
179/45/224 的 train/test/all 数量。不要用 Hugging Face 上的最新版文件覆盖这些冻结文件。

配置好 `AGENT_DIFF_API_KEY` 和 `AGENT_DIFF_BASE_URL` 后验收：

```bash
.venv/bin/python -m graphptc inspect-agent-diff \
  --config configs/agent_diff/graphptc.toml
```

正式范围必须显示 224 题，运行时每题 3 个 trial。若使用 self-hosted Agent-Diff，服务端模板和 seed
也属于执行环境，必须记录服务 commit；不能只下载任务 JSONL 就宣称预检完成。

## 5. FanOutQA

### 5.1 问题数据

`scripts/setup/server.sh` 安装 `pyproject.toml` 中固定 commit 的 `fanoutqa` 包；dev/test 问题由
`fanoutqa.load_dev()` 和 `fanoutqa.load_test()` 读取，不需要再复制 JSON 到 `data/`。确认版本和
dev 数量：

```bash
.venv/bin/python - <<'PY'
import importlib.metadata
import fanoutqa
print("fanoutqa", importlib.metadata.version("fanoutqa"))
print("dev", len(fanoutqa.load_dev()))
PY
```

正式配置要求 `fanoutqa>=1.3.0` 且 dev 为 310 题。

### 5.2 固定 Wikipedia 与 Kiwix

不要使用实时 Wikipedia。下载官方 2023-09 ZIM，并放在 Linux 原生文件系统：

```bash
sudo install -d -o "$USER" -g "$(id -gn)" /var/lib/fanoutqa
wget -c \
  https://datasets.mechanus.zhu.codes/fanoutqa/wikipedia_en_all_nopic_2023-09.zim \
  -O /var/lib/fanoutqa/wikipedia_en_all_nopic_2023-09.zim

mkdir -p external
wget -c \
  https://download.kiwix.org/release/kiwix-tools/kiwix-tools_linux-x86_64-3.8.2.tar.gz \
  -O external/kiwix-tools_linux-x86_64-3.8.2.tar.gz
tar -xzf external/kiwix-tools_linux-x86_64-3.8.2.tar.gz -C external

external/kiwix-tools_linux-x86_64-3.8.2/kiwix-serve \
  --daemon --port 8888 --threads 16 \
  /var/lib/fanoutqa/wikipedia_en_all_nopic_2023-09.zim
```

非 x86_64 服务器必须从 [Kiwix 官方目录](https://download.kiwix.org/release/kiwix-tools/) 选择对应架构，
不能直接使用上面的 x86_64 压缩包。

验收命令：

```bash
.venv/bin/python -m graphptc inspect-fanoutqa \
  --config configs/fanoutqa/graphptc-dev.toml
.venv/bin/python -m graphptc probe-fanoutqa-wikipedia \
  --config configs/fanoutqa/graphptc-dev.toml
```

## 6. FRAMES

### 6.1 下载 824 题

正式配置固定 Hugging Face revision `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`：

```bash
mkdir -p data/frames
curl --fail --location --retry 3 \
  'https://huggingface.co/datasets/google/frames-benchmark/resolve/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/test.tsv?download=true' \
  --output data/frames/test.tsv
```

不要改为 `main` revision。`test.tsv` 含答案和 gold Wikipedia 链接，adapter 只在生成结束后使用它们；
不得把整行数据或 gold 链接放进模型提示。

### 6.2 下载并建立固定 Wikipedia 索引

仓库脚本直接从 Google TFDS 公共 bucket 下载 `wikipedia/20230601.en/1.0.0` 的 256 个 TFRecord
分片，转换为 JSONL 并建立 Pyserini/Lucene BM25 索引：

```bash
sudo bash scripts/frames/setup_retriever.sh
bash scripts/frames/prepare_wikipedia.sh
```

`prepare_wikipedia.sh` 可以安全重复执行：完整分片、转换计数和已有索引会被复用。完成后应存在：

```text
data/frames/wikipedia-20230601/
├── tfrecord/
├── json/
├── counts/
├── index/
└── manifest.json
```

在一个长期运行的终端或服务管理器中启动服务：

```bash
bash scripts/frames/run_retriever.sh
```

然后在另一个终端验收：

```bash
.venv/bin/python -m graphptc inspect-frames \
  --config configs/frames/graphptc-test.toml
.venv/bin/python -m graphptc probe-frames-wikipedia \
  --config configs/frames/graphptc-test.toml
```

检查必须确认 824 题、`wikipedia/20230601.en` 和 6,672,479 篇文档。

## 7. 六项统一验收

六项单独准备完成后，先创建新模型 profile，再执行统一预检：

```bash
.venv/bin/python scripts/evaluation/full_suite.py preflight --profile MODEL_PROFILE
.venv/bin/python scripts/evaluation/full_suite.py all --profile MODEL_PROFILE --dry-run
```

预检应依次通过 9 项检查，并由 dry-run 打印 21 份 matched 配置命令。任一数据数量、revision、服务
metadata 或环境凭据不匹配时应停止处理；不要通过改小 `expected_tasks`、拆分数据或跳过失败检查来
启动正式评测。
