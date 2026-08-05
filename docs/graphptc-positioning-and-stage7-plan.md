# GraphPTC 当前定位与 Stage 7 实验计划

状态：当前决策基线，供后续实现和实验预注册使用

日期：2026-08-05

适用仓库：`D:\GraphPTC`

## 1. 文档目的

本文档统一当前 GraphPTC 的目标、边界和后续实验顺序。它取代口头讨论作为 Stage 7 的执行依据，但不覆盖历史冻结基线、已有 Gate 配置或运行产物。

本文档必须帮助后续执行者区分：

- 已经由产物验证的事实；
- 尚待实验验证的假设；
- GraphPTC 通用核心与 benchmark adapter；
- 离线分析、shadow 决策和会改变 agent 行为的 active 策略；
- 结构正确性、局部机制收益和最终 benchmark outcome。

## 2. 当前决策

GraphPTC 的定位从“运行时错误自动修复器”调整为：

> 基于执行依赖图的 PTC 诊断、研究进展监控、资源控制与局部恢复层。

依赖图的首要作用不再局限于异常后的代码 patch。它应统一承载执行 provenance、数据和状态依赖、artifact 使用关系、重复计算识别、失效传播以及安全复用。Repair 是图驱动动作之一，不是唯一价值出口。

近期目标优先级为：

1. 保持答案准确率非劣；
2. 降低重复检索、无进展调用和长尾成本；
3. 提高运行稳定性和可解释性；
4. 在明确失败现场中安全恢复；
5. 证据图成熟后再检验准确率提升。

当前不应宣称 GraphPTC 已提高 BrowseComp-Plus 准确率，也不应将局部 repair 成功等同于最终答案收益。

## 3. 已验证事实

### 3.1 当前 matched pilot20

最终 Gate 使用同一份 `data/browsecomp_plus/pilot20.questions.jsonl`：

- 数据集 SHA-256：`4edbb70bd09c97ec33292141f913ed5c1b1cac5cd84022ea02796e10803def87`
- prompt：`fewshot-ptc-v1`
- model：`mimo-v2.5`
- thinking：disabled
- temperature：`0.0`
- budget：30 turns，最多 29 个 PTC blocks
- control 与 active 的 20 个 example ID 集合完全一致
- 这是开发 grader 结果，不是官方 BrowseComp-Plus leaderboard 结果

历史 `fewshot-ptc-v1-turn30-pilot20` 为 `6/20`，其中 19 个有效回答；本次为匹配当前实现而重新运行的 control 为 `7/20`，20 个回答均有效。两次运行签名不同，不得把 `30%` 和 `35%` 当成同一运行的分数修订。

### 3.2 Active outcome Gate

报告：`runs/graphptc-active-v1/pilot20-comparison-report.json`

- control：`7/20`，35%
- active：`5/20`，25%
- 差值：`-2`，超过预注册允许损失 `-1`
- 13 项 Gate 中 12 项通过
- 唯一失败项：active accuracy non-inferiority
- 因此 Active 不晋级，不运行 pilot100，不设为默认路径

### 3.3 Repair 与 selective replay

- 20 题中 8 次进入 repair attribution
- 5 次提交 active repair
- 3 次判定 `not_repairable`
- repair error：0
- 复用工具调用：62
- 重新执行工具调用：14
- 5 个实际修复样本中，control 和 active 都是 `1/5`

在发生 repair 的局部范围内，62/76 个调用被复用，说明 selective replay 的安全复用机制具有局部效率价值。但触发覆盖率低，且没有观察到最终答案收益。

三个 `correct -> wrong` 样本均不是已提交 repair 导致：

- `qid 772`：`no_repairable_failure`
- `qid 896`：`no_repairable_failure`
- `qid 1234`：`not_repairable`

因此当前证据不能把准确率下降因果归因于 patch 或 selective replay。两个独立 API 运行在 repair 前已经可能产生不同轨迹。

### 3.4 主要失败形态

当前更显著的问题是正常执行中的检索漂移和调用膨胀：

- live tool calls：control `1622`，active `2596`
- repeated searches：control `237`，active `586`
- candidate recall：control `0.3554`，active `0.3148`
- fetched evidence recall：control `0.2562`，active `0.2389`
- 中位时延：control `196.1s`，active `261.7s`

代表性案例：

- `qid 152`：87 -> 592 live calls；发生 repair，但 replay 只重新执行 7 个调用，大部分额外调用来自后续 agent 轨迹。
- `qid 1204`：155 -> 927 live calls；没有发生 repair，是基础轨迹漂移和重复检索的直接案例。

## 4. 当前能力与边界

### 4.1 当前执行链路

```text
fewshot-ptc-v1 + CodeActPTCAgent
        -> Persistent Python Runtime + runtime tools
        -> append-only execution events
        -> execution dependency graph
        -> failure attribution
        -> local patch
        -> invalidation
        -> selective replay / commit
        -> shadow or active integration
```

### 4.2 当前依赖图表达的内容

当前 Stage 2 图主要包含：

- episode、block、tool、transform、state、output 节点；
- `DATA`、`CONTROL`、`STATE`、`RESULT_OF` 等因果边；
- 工具结果、stdout 和最终答案 artifact；
- 源码位置、运行状态和跨 block 变量依赖；
- replay action 与 artifact provenance。

它适合回答：

- 哪个 block 或 tool 失败；
- 哪段代码和哪些上游节点与失败有关；
- patch 后哪些节点和 artifact 必须失效；
- 哪些只读、幂等结果可安全复用。

它尚不能直接回答：

- 查询或结果是否重复；
- 新调用是否增加信息；
- fetch 结果是否被后续 transform、stdout 或答案消费；
- 问题的哪些约束仍缺少证据；
- 当前研究是否已经停滞或进入循环。

当前图是 execution provenance graph，不是完整的 research progress graph。

## 5. Agent 与图的交互定位

### 5.1 当前模式

基础 agent 不能查询图。agent 只生成 PTC block，并在 Python runtime 中调用 benchmark 提供的工具。Observer 在旁路记录事件。

Active 仅在第一个显式 block error 后自动：

1. 从当前事件构建失败现场图；
2. 选择失败 anchor 和有界 context；
3. 向独立 repair model 请求局部 patch；
4. 分析失效范围并选择性回放；
5. 用 repaired stdout 替换错误 observation。

agent 只看到最终 stdout 或错误，看不到 node、edge、artifact 或 repair context。

### 5.2 目标模式

采用“自动监控为主，少量只读查询为辅”的混合模式：

```text
每个已完成 block
    -> GraphSession 更新 completed-prefix snapshot
    -> deterministic progress monitor
       -> 正常：不打扰 agent
       -> 命中 trigger：提供有界摘要
          -> agent 必要时执行少量只读查询
```

不得直接向 agent 暴露原始 `node()`、`predecessors()`、`successors()`、全图 dump 或任意 artifact 全文。原始图 ID 和遍历属于内部实现，直接暴露会增加调用、上下文和策略漂移。

候选只读接口限定为：

#### `graph_progress()`

返回 benchmark-neutral、有界、确定性的进展摘要：

- 已用和剩余预算；
- 新增与重复调用数量；
- 新增与重复 result fingerprint 数量；
- 最近 block 的 artifact 新颖度；
- 没有观测到通往后续输出 lineage 的 artifact；
- `STAGNATION`、`LOOP`、`TRUNCATION`、`BUDGET_RISK` 等 trigger。

#### `graph_evidence(resource_ids=None)`

返回有界 provenance：

- resource 来自哪个 tool call、block 和参数 fingerprint；
- 是否已获取完整内容；
- 是否进入 transform、stdout 或后续 state；
- artifact 的短 preview 和 hash；
- 不返回 gold、grader 信息、任意全图或无界全文。

首个在线 challenger 中，每题最多允许 2 次图查询，单次输出字节上限必须在配置中冻结。具体数值在 online Gate 前预注册。该 challenger 必须使用 control/placebo/graph 三臂设计，不能把新增接口协议的影响归因于图信息。

Patch、invalidate、replay、commit、stop 等写操作继续由 harness 控制，不作为 agent 可调用接口。

## 6. 通用核心与 benchmark adapter

新定位本身不是 BrowseComp-Plus 专属，但当前 Stage 5/6 实现仍有 benchmark 耦合：

- Active/Shadow 直接导入 `BROWSECOMP_PLUS_RUNTIME_TOOL_MANIFEST`；
- invalidation/selective replay 默认将 `search`、`fetch` 视为只读工具；
- 检索统计假定 `query`、`docid`、`snippet`、`content` 等字段。

Stage 7 必须把以下通用语义放入核心：

工具副作用、状态范围、确定性和回放权限是正交属性，不能压缩成一个互斥枚举。通用描述至少包含：

```text
ToolDescriptor
  side_effect: NONE | READ | WRITE
  state_scope: NONE | EPISODE | EXTERNAL
  determinism: DETERMINISTIC | VERSIONED | VOLATILE
  replay_policy: REUSE_RESULT | REEXECUTE | RESET_REQUIRED
  version_key: optional
  resource_key: optional
  argument_fingerprint: deterministic
  result_fingerprint: deterministic
  cost_class: optional
```

`idempotent` 不能单独决定安全复用：只读外部工具可能返回随时间变化的结果，幂等写操作仍然具有副作用。`REUSE_RESULT` 必须同时满足 descriptor、版本和 source artifact 完整性要求。

benchmark adapter 只负责：

- 将具体工具 schema 映射为 `ToolDescriptor`，并提供必要的版本键；
- 从具体结果提取 resource ID 和 fingerprint；
- 声明工具可复用性和副作用；
- 提供 benchmark 专属评分与报告，不改变核心图语义。

核心模块不得导入 BrowseComp benchmark，不得按 `search/fetch/docid` 名称分支。

适用范围必须明确：

- execution graph、failure attribution 和 read-only replay 应能泛化到检索、数据库、计算器等多工具任务；
- retrieval progress 是研究型任务的可插拔扩展；
- 对 `WRITE`、`VOLATILE` 或缺少可验证版本键的外部工具，未实现快照、补偿或事务协议前必须禁止结果复用；
- Stateful Tool Support 当前不是 BrowseComp-Plus Stage 7 的前置项，但若扩展到交互式环境则必须单独实现和验证。

## 7. 反过拟合规则

1. `qid 152`、`1204` 仅作为开发诊断案例，不得用于最终阈值验收。
2. 不得用 gold answer、qrels、grader label 或 evidence qrels 作为在线 trigger 特征。
3. BrowseComp 可用于 adapter 开发；schema、阈值和策略冻结后，必须在第二个 benchmark 上验证，不得继续调参。
4. 每个图策略必须与简单 counter baseline 比较。若简单调用计数器效果相同，不应把收益归于依赖图。
5. benchmark 专属代码只能存在于 adapter 和 evaluator。
6. prompt、model、budget、runtime、adapter 和 graph policy 影响必须分别报告。
7. GraphPTC 实验继续使用 `fewshot-ptc-v1` 作为 prompt 基础，不修改历史 `original-ptc-v1`。
8. 任何模型可见的图摘要或接口都必须定义为独立 challenger，不能混入 control。

## 8. Stage 7 实验计划

### Stage 7.0A：Replayability Audit

目标：先确认现有产物能否支持确定性失败现场重建，再讨论 repair/no-repair 因果效果。

当前 runtime trace 只保存变量名称和类型，没有保存可直接恢复的 Python 对象快照。现有 selective replay 会在新 runtime 中重新执行 prefix blocks。因此本阶段使用 `frozen-prefix reconstruction`，不得称为 runtime snapshot replay。

对已有 8 个失败现场逐一审计：

- 冻结 source events、block code、tool result artifacts、已生成 patch 和所有输入 hash；
- 列出重建前缀中的每个工具调用及预期 replay action；
- 验证所有 `REUSE_RESULT` 都有完整 artifact；
- 验证所有 `REEXECUTE/EXECUTE_NEW` 是否仍依赖 live tool；
- 检查代码中的时间、随机数、外部模块状态和未持久化对象等非确定性来源；
- 将案例分类为 `RECONSTRUCTABLE_OFFLINE`、`LIVE_DEPENDENT` 或 `UNREPLAYABLE`。

Gate：

- 8 个案例全部完成分类并给出机器可读原因；
- 只有 `RECONSTRUCTABLE_OFFLINE` 可进入确定性 micro-gate；
- 确定性重建期间 live tool 调用数必须为 0；
- 重建的未修改 prefix stdout、tool arguments、tool results 和 source event hash 与原始产物一致；
- 使用 live retriever 或新模型请求的结果不得标记为确定性证据。

### Stage 7.0B：Repair Micro-Gate 与 Graph Utility Audit

目标：关闭现有 repair 的 block 级问题，同时建立独立的进展失败标签协议。

任务 A：仅对 Stage 7.0A 判定为 `RECONSTRUCTABLE_OFFLINE` 的现场执行 frozen-prefix repair/no-repair micro-gate。

- no-repair 与 repair 从相同 source event hash 分叉；
- 比较原始失败、patch 后 block 成功、stdout 有效性、复用边界和 artifact 泄漏；
- 不运行新的完整 20 题 benchmark；
- 只形成 block 级结论，不宣称最终答案准确率收益。

任务 B：建立失败形态与标签协议。

- pathology type：runtime failure、repeated call/result、retrieval stagnation、stdout truncation、no observed lineage to output、budget exhaustion risk；
- earliest actionable block：最早允许 trigger 的 block；
- allowed trigger window：可接受的触发区间；
- required causal calls/nodes：trigger 必须覆盖的节点；
- forbidden triggers：高效正确轨迹中不得触发的位置。

标签来源分为：

- 具有确定真值的 synthetic cases；
- 在查看 trigger 输出前冻结的人工 trajectory labels；
- `qid 152/1204` 等 pilot20 案例仅作为 development diagnostics，不作为 heldout Gate。

Gate：

- repair/no-repair 使用相同 source event hash，且离线重建无 live tool；
- synthetic truth 和人工标签文件在 trigger 阈值之前冻结并记录 hash；
- graph 与 simple-counter baseline 分别报告 precision、recall、lead time、误报和漏报；
- pilot20 结果只用于可行性和错误分析，不作为 Graph Utility 晋级证据；
- 如果图在 synthetic truth 和冻结人工标签上均无增量价值，停止 agent-facing Progress Graph，保留现有 provenance/repair 图并考虑简单 runtime guard。

### Stage 7.1：Generality Gate 与核心解耦

目标：在新增 Progress Graph 前移除 Stage 5/6 的 BrowseComp 硬编码。

工作项：

- 注入 runtime tool manifest；
- 注入副作用、状态范围、确定性、版本和 replay policy 描述；
- selective replay 根据 descriptor 判定，不根据工具名判定；
- Active/Shadow 不再导入 BrowseComp benchmark；
- 保持现有 BrowseComp 行为和产物 schema 等价。
- 在进入 Stage 7.2 前冻结第二个检索 benchmark 的数据来源、adapter contract、主指标和 provenance；无法获得可信资产时停止跨 benchmark 晋级声明。

合成案例至少覆盖：

- read-only search/fetch；
- calculator 或数据库查询；
- 同 block 多工具和跨 block state；
- mutating tool 负例；
- 同参数同结果、同参数异状态、异参数同结果。

Gate：

- GraphPTC 核心无 `browsecomp` import；
- 核心无 `search/fetch/docid` 名称分支；
- `NONE/READ + DETERMINISTIC/VERSIONED` 工具的 replay 与当前允许行为等价；
- `WRITE`、`VOLATILE` 或无版本外部工具默认 `RESET_REQUIRED`；
- frozen control 路径 model-visible payload 不变。

### Stage 7.2：GraphSession 与 Retrieval Progress Graph

目标：离线支持每个 completed block 的增量图快照和研究进展分析。

新增最小组件：

- `GraphSession`：消费 append-only events，产生不可变 completed-prefix snapshot；
- `ProgressProjector`：从通用 tool/artifact 事件派生调用和结果 fingerprint；
- `ProgressSummary`：产生有界确定性摘要；
- retrieval adapter：仅在检索 benchmark 中提取 resource IDs。

当前图只提供静态 AST lineage 与 executed spans 的组合证据，不具备对象级动态 taint tracking。因此第一版只能产生 `NO_OBSERVED_LINEAGE_TO_OUTPUT`，不能断言 artifact 实际未被消费。

Gate：

- prefix snapshot 不包含未来事件或未完成 block；
- 增量构建与对相同前缀全量重建字节等价；
- 输出顺序和 hash 确定；
- 20/20 当前轨迹可构建；
- 无 gold/qrels/grader 字段进入图；
- `qid 152/1204` 的膨胀在离线报告中可定位到具体 block、调用和重复结果。
- list/dict mutation、alias、helper function、闭包和跨 block state 均有正负对照；未通过精确 Gate 的 lineage 类型不得用于在线 trigger。

### Stage 7.3A：BrowseComp Development Shadow Gate

目标：在不改变 agent 行为的条件下评估自动 trigger。

Shadow 只记录本来会产生的：

- `STAGNATION`；
- `REPEATED_CALL_LOOP`；
- `REPEATED_RESULT_LOOP`；
- `NO_OBSERVED_LINEAGE_TO_OUTPUT`；
- `BUDGET_RISK`。

Gate 必须使用 Stage 7.0B 已冻结的标签协议，至少包括：

- frozen forbidden positions 误触为 0；
- trigger 在预注册 allowed window 内命中；
- trigger 精确指向 causal blocks/calls；
- graph 相对 simple-counter baseline 的 precision、recall 或 lead time 增量为正，并报告样本数；
- shadow 对 response、tool calls 和 model messages 的影响为 0。

该阶段只允许选择 trigger schema 和阈值。所有选择完成后冻结 config/hash，后续 portability shadow 不得继续调参。

### Stage 7.3B：Cross-Benchmark Portability Shadow Gate

目标：在任何模型可见或在线控制改动前，验证 schema、descriptor 和 trigger 不依赖 BrowseComp 字段与阈值。

- 使用 Stage 7.1 预注册的第二个检索 benchmark；
- 复用冻结的 core schema、agent-facing schema 和 trigger 阈值；
- 只允许新增 benchmark adapter，不允许修改核心或阈值；
- 在一个非检索多工具合成套件上验证 execution graph、invalidation 和 replay；
- 明确标记不适用的 retrieval progress 指标。

Gate：

- 核心代码和 schema 无 benchmark 专属分支；
- 第二 benchmark 的全部 episode 可构建 completed-prefix graph；
- shadow 对模型请求、答案和原始工具调用影响为 0；
- 通用 trigger 按预注册标签报告 precision/recall/lead time；
- 若 portability 失败，返回 Stage 7.1/7.2 修正通用抽象，但不得用第二 benchmark 调整 BrowseComp 阈值后继续声称 heldout。

### Stage 7.4A：Exact-Result Reuse Gate

目标：先独立验证最低风险、无需 agent 查询图的自动复用动作。

`REUSE_EXACT_RESULT` 只适用于参数 fingerprint、tool version、state scope 和 source artifact 均匹配，且 descriptor 明确允许 `REUSE_RESULT` 的调用。缓存键必须包含工具版本或等价 snapshot revision。

Gate：

- model-visible payload、logical tool calls、返回值、异常语义和顺序与未缓存 control 等价；
- `WRITE`、`VOLATILE`、版本不匹配和 artifact 不完整案例零复用；
- eligible calls、eligible savings、realized savings 分别报告；
- 主要效率指标为 `realized_savings / eligible_savings`，不使用与可节省上限无关的固定总调用降幅；
- BrowseComp 真实 1 至 2 题 smoke 只验证协议，不形成准确率收益声明。

### Stage 7.4B：Agent-Facing Graph Interface Gate

目标：隔离“新增接口协议”与“graph-derived information”的效果。

每个接口单独实验。先测试 `graph_progress()`；只有它通过后才单独测试 `graph_evidence()`。不得同时加入停止策略、搜索重定向、evidence compaction 或 prompt 的其他修改。

必须使用三臂 matched control：

```text
A: fewshot-ptc-v1 原始 control
B: 相同接口、说明、调用额度和近似输出大小，但返回 telemetry/placebo summary
C: 与 B 相同协议，但返回 graph-derived summary
```

- `A vs B` 测量新增接口与提示本身的影响；
- `B vs C` 才能归因依赖图信息的增量价值；
- placebo 不得包含 C 独有的 causal lineage 或未来信息；
- 三臂必须使用相同 model、budget、tool adapter、grader 和题集。

结构 Gate：

- control payload 和工具语义不变；
- agent 图查询次数和字节上限受控；
- 图查询不产生 retriever 调用；
- 禁止访问未来节点、gold 和无权限 artifact；
- 图接口在 completed-prefix snapshot 上执行，禁止读取当前未完成 block；
- `WRITE`、`VOLATILE` 和无版本外部工具不受 agent 接口改变。

该阶段是能力和行为 Gate，不直接晋级默认路径。候选 outcome 指标包括：

- A/B/C 有效 response 和 grade 完整性；
- `A vs B` 与 `B vs C` 的 paired win/loss/tie；
- logical/runtime/live calls 和 repeated calls；
- candidate 和 fetched evidence recall 不下降；
- latency、token、median/P90 和异常案例单独报告。

任何数值阈值必须根据 Stage 7.3 冻结的可达范围和 heldout 样本量预注册。不得继续使用与动作覆盖上限无关的统一 `25%/50%` 阈值。

### Stage 7.5：Repeated Paired Heldout Outcome Gate

只有 Stage 7.3B portability 和对应 Stage 7.4 行为 Gate 均通过后执行。

- 在运行前冻结题集、轮数、停止规则、primary outcome、非劣界值和效率阈值；
- 使用多轮 matched A/B/C 配对，或对不涉及 agent 接口的动作使用 matched control/action 配对；
- 报告每题、每轮和汇总的 win/loss/tie；
- 报告正确率差值分布、调用和时延分布以及跨轮方差；
- repair、cache、graph interface 和非触发样本分别报告；
- 只有汇总 heldout non-inferiority 与动作专属效率 Gate 同时通过才允许扩大样本。

不得用一次配对运行消除模型轨迹漂移，不得用 development shadow 设置 heldout 后再回调阈值，也不得将开发 grader 结果标记为官方 benchmark 成绩。

## 9. 统一指标

### 结构与安全

- exact node/edge/artifact provenance；
- forbidden node/artifact leakage；
- prefix isolation；
- byte stability 和 artifact hash；
- side effect、state scope、determinism、version 和 replay policy 分类正确率；
- replay reuse/execute provenance。
- frozen-prefix reconstruction 完整性与 live-dependency 分类。

### 机制覆盖

- triggerable episode rate；
- attribution coverage；
- repairable rate；
- progress trigger precision/recall；
- trigger lead time；
- graph 相对简单 counter 和 placebo summary 的增量收益；
- `NO_OBSERVED_LINEAGE_TO_OUTPUT` 在各类 alias/mutation 场景中的精度。

### 效率

- logical/runtime/live tool calls；
- calls per block；
- repeated calls、queries 和 result slots；
- unique resource/artifact growth；
- reused 与 executed replay calls；
- eligible savings、realized savings 和二者比率；
- model requests、tokens、latency median/P90。

### Outcome

- 有效 response/grade 数；
- accuracy 或 benchmark 原生主指标；
- A/B/C 或 control/action paired win/loss/tie；
- 每题、每轮和汇总的 outcome 差值及跨轮方差；
- candidate/fetched evidence recall（仅适用时）；
- repair、cache、graph interface、placebo 和非触发子集分别报告。

## 10. 实验纪律与产物

每个真实运行必须冻结并保留：

- config 和预注册 Gate；
- dataset/prompt/model/runtime/tool manifest hashes；
- ToolDescriptor、adapter、tool version/snapshot hashes；
- 统一 run signature；
- responses、grades、report、events；
- graph/trigger/action sidecar；
- frozen trajectory labels、allowed trigger windows 和 arm assignment；
- artifact SHA-256；
- 开发、heldout、官方结果标签。

运行原则：

- 优先复用签名完全匹配的冻结产物；
- 不因期待结果而修改已注册阈值；
- 每个 challenger 只改变一个主要变量；
- synthetic/fixture Gate 不能替代真实 smoke；
- 真实 smoke 不能替代 heldout outcome Gate；
- Gate 失败后不扩大样本；
- `original-ptc-v1` 保持冻结；GraphPTC prompt 迭代使用 `fewshot-ptc-v1`；
- Stateful Tool Support 在当前 BrowseComp-Plus 阶段继续暂缓。

## 11. 立即执行顺序

下一次实现会话从 Stage 7.0A 开始，顺序不可跳过：

1. 写入 Stage 7.0A Replayability Audit 预注册 Gate；
2. 将 8 个已有失败现场分类为 offline reconstructable、live dependent 或 unreplayable；
3. 只对 offline reconstructable 案例执行 Stage 7.0B repair/no-repair micro-gate；
4. 在查看 trigger 输出前冻结 synthetic truth 和人工 trajectory labels；
5. 完成 Graph Utility Audit，并加入 simple-counter control；
6. 通过增量价值 Gate 后执行 Stage 7.1，同时冻结第二 benchmark 及其 adapter contract；
7. 完成 Stage 7.2 GraphSession 和 lineage 精度 Gate；
8. 依次通过 Stage 7.3A BrowseComp development shadow 与 Stage 7.3B portability shadow；
9. 先执行 Stage 7.4A exact-result reuse，再用三臂设计执行 Stage 7.4B agent interface；
10. 最后执行 Stage 7.5 repeated paired heldout Gate，通过后才允许扩大样本。

未通过 Stage 7.3B 前，不向 agent 暴露模型可见图接口。未通过 Stage 7.5 前，不晋级默认策略或运行更大 pilot。

本阶段的首要决策点不是“如何增加更多图功能”，而是：

> 依赖图是否在 benchmark-neutral 条件下，比简单 telemetry 更早、更准确地识别可行动的执行和研究失败，并能以非劣准确率换取稳定的效率收益。
