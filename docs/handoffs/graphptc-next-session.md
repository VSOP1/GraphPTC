# GraphPTC 新会话交接 Prompt

你正在 `D:\GraphPTC` 继续实验。先完整阅读根目录的 `AGENTS.md`（若存在）、
`GraphPTC完整实现框架.md`、当前 Git 状态和相关测试，再提出短计划并逐步执行。

## 当前目标

Stage 0 的 `original-ptc-v1` 已冻结。不要继续调 baseline prompt，不要加入 few-shot、
最少调用次数、跨 block 新颖度账本或其他策略规则。现在正式迭代 GraphPTC，但必须先保证
GraphPTC 的 Stage 1 只是离线观测层，不改变 baseline 的模型可见上下文、工具协议、执行语义
或检索行为。

## 冻结 Original PTC 契约

- 模型：`mimo-v2.5`，thinking disabled，单次 completion 上限 2048，episode 上限 30 turns。
- 模型直接看到并可调用的唯一工具是 `programmatic_tool_call`；`search` 和 `fetch` 只允许从
  Python runtime 内调用，完整 schema 同时展示给模型。
- `tool_choice=auto`，一个 API turn 可产生多个 PTC block，Agent 自主决定 block 数量。
- 使用 `CodeActPTCAgent`、`PersistentIpcRuntime`、`OfficialCorpusSearchTools`；Python 状态在同一
  task 内持久、task 间重置。
- 模型只看到程序实际 stdout 或执行错误；`structured_observation=False`，不得注入
  `PTC_OBSERVATION`、调用统计、docid 新颖度等 harness 元数据。
- telemetry、程序分析和图事件只能离线记录，不反馈给模型。
- 正式 baseline variant 是 `original-ptc-v1`。`fewshot-ptc-v1` 和
  `phase-planning-v1` 仅是已拒绝晋级的历史诊断，不是 GraphPTC 起点。

核心实现位于：

- `src/graphptc/codeact_agent.py`
- `src/graphptc/browsecomp_plus_benchmark.py`
- `src/graphptc/persistent_runtime.py`
- `configs/browsecomp_plus.original-ptc-v1-turn30-pilot20.toml`

## 从头实现 Stage 1

仓库已删除早期 Stage 1 原型，包括 GraphPTC Agent、observability runtime、benchmark
adapter、CLI/config、launcher 和测试。不要从 Git 历史恢复这些文件；该原型基于旧的
`OriginalPTCAgent`、字面代码 prompt、`search_local/search_local_batch`、SQLite adapter
和非持久 block 状态，与冻结 baseline 不等价。

第一项任务是从当前冻结 baseline 提取最小、透明的 observability adapter。优先让 Original
PTC 和 GraphPTC 复用同一 Agent loop、prompt、tool spec、retriever、persistent runtime 和
config 构造路径，只通过明确的执行 hook 旁路记录事件，避免复制一套会继续漂移的执行路径。
不要为未来 Stage 2 预先设计复杂抽象。

验收条件：

1. 用确定性 fake model/search runtime 做 matched test：Original PTC 与 Stage 1 GraphPTC 的
   model requests、生成 code、stdout、最终答案和 search/fetch 调用完全一致。
2. GraphPTC 只额外产生 append-only 的 episode/block/tool 事件；事件不得进入 messages。
3. 覆盖多 block、单 block 多工具调用、runtime error、stdout truncation、task 间状态重置。
4. 使用相同 Original config 在 1 至 2 个 BrowseComp-Plus 样本做 smoke test；先确认协议与事件，
   不因随机模型输出差异宣称效果提升。
5. 上述 conformance gate 通过后，再依据框架规划 Stage 2 Dynamic Dependency Graph。

## 冻结评测事实

固定 100 题上，MiMo 同时充当生成模型和开发 grader：

- Original PTC：39/100；平均 1.592 runtime calls/PTC block。
- Positive few-shot：26/100；虽增至 4.081 calls/block，但重复结果更多，未晋级。
- Direct tools：23/100。
- Direct tools 全 830 题：208/830，25.06%。该 run 对 qid 584 有一次记录在案的失败重试，
  只能按报告中的 provenance 解读。

汇总报告：`runs/browsecomp_plus/pilot100-comparison/report.json`。
Original pilot20 的 grades/report 完整，但 `responses.jsonl` 曾被中断的 restart 截断为 2 条；
100 题分数由完整 grades 合并，轨迹指标来自 pilot20/extra80 的 component reports。不要把这份
缺失轨迹的 pilot20 当作完整 raw-response artifact。

这些是开发结果，不是官方 leaderboard 分数。暂时不要运行新的 830 题全量评测；先使用独立
合成任务和固定小样本完成结构与行为等价验证。

## 当前目录约定

- `src/graphptc/`：当前 baseline 核心实现；GraphPTC 将从这里建立新的独立模块。
- `src/graphptc/experiments/`：被保留的受控诊断实现。
- `scripts/data|services|launchers|analysis|experiments/`：数据、服务、启动、分析和消融脚本。
- `tests/core|agents|benchmarks|experiments/`：对应测试类别。
- `configs/` 保持扁平，避免破坏冻结报告中的路径和实验溯源。

已删除 `fewshot-replan-v1`、`mincalls-control-v1` 和 `fewshot-mincalls-v1` 的配置、
代码入口、专用分支与运行产物。不要恢复它们。

## 工作纪律

- 当前 worktree 含未提交的 baseline/实验整理变更；先阅读 `git diff`，不要回滚未知修改。
- 每次改动保持最小，先写行为等价测试，再改实现。
- 分离并报告 protocol、prompt、model、budget、runtime 和 GraphPTC harness 的影响。
- 不拿 BrowseComp-Plus 测试题调 prompt，不因 cached token 便宜而忽略轨迹长度和重复检索。
- 使用 `.\.venv\Scripts\python.exe -m pytest -q` 验证；所有真实 API 运行保留 config、签名、
  response、grade、report 和事件文件。

完成阅读后，先基于当前 baseline 标出可插入离线观测 hook 的最小位置，再给出只针对透明
observability adapter 的实现计划，随后从头实现并验证。不要恢复旧 Stage 1，也不要直接开始
归因、修复或重执行。
