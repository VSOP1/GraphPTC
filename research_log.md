# GraphPTC Research Log

## v11 final100

- 实验假设：将 Research Graph 上下文暴露给 Agent，可使其根据已有检索、证据和候选状态选择下一动作。
- 失败现象：39/100；61 个错误中，33 个候选检索未命中、11 个候选命中但未 fetch、17 个证据已 fetch 但答案错误。2164 次 `CONTINUE`，仅 2 次 `INSPECT`、5 次 `PATCH`，artifact reuse 为 0。
- 原因判断：v11 主要记录并展示图，目标取当前或首个未解决 constraint；缺少稳定任务分解、需求状态、明确 diagnosis，以及 expected/actual graph delta 闭环。
- 修改内容：无（作为后续迭代起点）。
- 结果变化：最终准确率 39%。
- 是否保留：保留为 v12 的开发参照，不再继续增强 v11 prompt。

## v12.0 requirement-state（开发中）

- 实验假设：一次性需求分解、依赖状态归约和 target-specific diagnosis 能让依赖图指出具体证据缺口，而不只是展示调用历史。
- 失败现象：v11 constraint 粒度漂移；`CONTINUE` 执行即算 aligned；target context 未验证预期状态变化。首次20题生成有 1 题因 graph delta 超过 3200 字符失败，因此该轮不计结果。
- 原因判断：缺少跨 block 稳定的 requirement DAG，以及 `expected graph delta -> actual graph delta` 校验。
- 修改内容：增加一次性 `task_graph`、requirement 依赖和状态机；为每个候选 target 生成 diagnosis/expected delta；按目标验证实际图变化；模型仍负责语义判断，未加入题目、关键词或固定搜索次数规则。对超长观察仅保留紧凑 actual-delta 摘要并分级裁剪，避免运行时因图增长中止。
- 结果变化：修复观察边界后20/20成功；3/20（15%），相对 v11 的6/20下降。candidate recall 37.51%，fetched-evidence recall 34.55%；fetch 1111 次、重复 fetch 708 次；20/20 初始化 task graph，但仍全部选择 `CONTINUE`，expected delta 实现 258/474。
- 是否保留：不保留该行为版本。保留任务图/状态/expected-delta骨架，修正具体图动作暴露与复用后再测。

## v12.1 actionable-target（开发中）

- 实验假设：诊断必须携带可执行的 target-local 操作，且 telemetry 只能统计模型实际看到的压缩后上下文；图内精确 fetch 复用可减少重复抓取而不改变内容语义。
- 失败现象：v12.0 的 `UNFETCHED_CANDIDATE_DOCUMENT` 出现281次，但 known-document fetch 仅36次；压缩可能移除 docid，Agent转而大范围 fetch，产生708次重复 fetch。
- 原因判断：diagnosis 与 action affordance 分离；内部完整 contract 被错误计为模型可见；重复 fetch 没有走已有 artifact。
- 修改内容：主 target 的 diagnosis 增加具体 `suggested_operations`；次要 target 只暴露状态摘要；后续消费统计改用实际序列化 contract；相同 docid 的 fetch 从 Research Graph artifact 返回。
- 结果变化：8/20（40%），相对 v11 的6/20和 v12.0 的3/20提高；新增3个 `wrong→correct`、1个 `correct→wrong`。candidate recall 45.69%，fetched-evidence recall 39.00%；实际 fetch 从1111降至220，重复 fetch 从708降至0，图内 fetch artifact reuse 268次。剩余12错为7个 retrieval miss、5个 evidence 后语义/答案错误。
- 是否保留：保留，作为当前最佳开发版本。

## v12.2 adaptive-target-selection（开发中）

- 实验假设：基于每个 requirement 的实际 delta 历史在多个目标间排序，可在当前分支不再产生图变化时转向尚未探索或更有产出的依赖，而无需搜索次数阈值。
- 失败现象：v12.1 仍有254/500次 expected delta 未实现、7个 retrieval miss；5/20未显式提供 `task_graph`，但在首 block 提供了 model-authored constraints。
- 原因判断：当前 target 排序仍偏向 current target；首 block 的既有 constraint decomposition 没有统一登记为一次性任务图。
- 修改内容：把首 block 已声明的 constraints 作为无显式依赖的 fallback 一次性任务图；target 排序依次考虑依赖是否就绪、是否有具体可执行缺口、是否未探索、历史 actual-delta 实现率，并向模型暴露选择特征。未增加题目规则或固定停搜阈值。
- 结果变化：7/20（35%），低于 v12.1 的8/20；candidate recall 45.01%，fetched-evidence recall 34.89%。task graph 初始化20/20，但搜索增至1648、expected delta 实现降至225/500，没有新增 `wrong→correct`。
- 是否保留：不保留 adaptive ranking；保留首 block constraint 到一次性任务图的兼容初始化。

## v12.3 typed-fetch-adapt（开发中）

- 实验假设：图动作必须改变真实 PTC 执行；当 Agent 选择带具体 docid 的 target 时，由 Typed Adapt Executor 在 block 开头落实该 fetch，能缩小“诊断正确但程序未执行建议”的差距。
- 失败现象：v12.1 有302次 `UNFETCHED_CANDIDATE_DOCUMENT`，但仅30次消费已曝光 docid；多数 block 在收到 fetch diagnosis 后仍继续 search。
- 原因判断：v12.1 的 suggested operation 仍是提示，不是控制动作。
- 修改内容：撤回 v12.2 target ranking；当模型选择的 CONTINUE opportunity 含 graph-backed fetch 时，在原代码前注入该 fetch，将完整结果置于 `_graph_target_document` 并输出有界摘要；记录 program override 和 actual delta。保留模型对 action/target 的语义选择。
- 结果变化：5/20（25%）；相对 v12.1 有4个 `correct→wrong`、1个 `wrong→correct`。221次 program override 将 known-document fetch 提高到221、actual delta 实现提高到386/469，但 candidate recall 降至41.42%、fetched-evidence recall 降至34.07%。
- 是否保留：不保留。强制落实结构上可行的 fetch 不等于语义上有价值，验证了 Runtime 不应替 Agent选择文档。

## v12.4 atomic-claim（开发中）

- 实验假设：保留 v12.1 的模型语义裁量，同时降低 candidate/evidence 入图摩擦，可使已获取证据真正进入 requirement 状态和答案上下文。
- 失败现象：v12.1 剩余12错中5个已 fetch gold evidence 仍答错；许多 episode 有大量 fetched documents，但 verified evidence 和 candidate 很少。
- 原因判断：现有接口要求先建 candidate、再用 exact quote 建 evidence，模型经常只在本地代码中推理而不提交 Research Graph，导致 Assess 无法区分已支持候选和未验证猜测。
- 修改内容：恢复 v12.1 行为，撤销 fallback ranking 和 program override；新增 source-verified `graph_add_claim`，一次提交 candidate、constraint 和 supports/refutes evidence；仍要求 quote 来自已 fetch 文档。
- 结果变化：6/20（30%）；claim commit 11次，但 verified evidence 从 v12.1 的53降至21，candidate recall 35.49%、fetched-evidence recall 31.78%。相对 v12.1 有3个 `correct→wrong`、1个 `wrong→correct`。
- 是否保留：不保留。恢复 v12.1 作为最终候选；原子接口可用不等于 Agent 会形成更好的研究轨迹。

## 最终候选选择

- 候选结果：v11 6/20；v12.0 3/20；v12.1 8/20；v12.2 7/20；v12.3 5/20；v12.4 6/20。
- 选择：v12.1 actionable-target。保留一次性 task graph、requirement 状态、target-specific diagnosis/expected delta、真实模型可见性统计、具体 suggested fetch 和精确 fetch artifact reuse；不保留 target ranking、program override、atomic claim。
- pilot100 结果：100/100生成和评分有效，35/100（35%）；candidate retrieval recall 41.93%，fetched-evidence recall 29.47%。65错中34个 retrieval miss、16个 candidate命中但未fetch、15个证据已fetch后的语义/答案错误。
- 机制结果：88/100初始化 task graph；171个 verified evidence；694次精确 fetch artifact reuse，重复 fetch 为0；但2450次 `CONTINUE` 对比3次 `INSPECT`、4次 `PATCH`，且 actual delta 1112次实现、1355次未实现。
- 结论：完成最终评测并保留 v12.1 代码。它证明了任务图、具体缺口暴露和依赖感知复用可以进入在线执行，但35%未超过历史 v11 final100 的39%，不能宣称端到端能力提升。下一轮若继续，应围绕 query/requirement覆盖和 evidence 后推理，而不是继续强化 fetch 或增加任务规则。

## v13 prompt-v2（开发中）

- 实验假设：将 Graph Adapt 的追加式说明改为明确的 `INITIALIZE → ASSESS → DECIDE → EXECUTE → VERIFY` 协议，并补充正常、冲突检查和真实 `failure → PATCH → re-execution` few-shot，可使模型依据图状态切换动作，而不是惯性 `CONTINUE`。
- 修改边界：新增独立 `fewshot-ptc-graph-v2`；只改变 system/user prompt 与 synthetic few-shot，不改变 tool schema、runtime、retriever、模型、预算或固定20题。旧 `fewshot-ptc-v1` 与既有产物保持不变。
- 本轮结果：20/20执行成功，开发 grader 为7/20（35%）；candidate retrieval recall 41.74%，fetched-evidence recall 37.49%。13个错误分为8个 retrieval miss、2个候选命中但未 fetch、3个已 fetch evidence 后答案错误。
- 机制结果：442个 PTC blocks 中399次 `CONTINUE`、41次有效 `PATCH`、20次 `ANSWER`，没有 `INSPECT` 或显式 `REUSE_REPLAY`。47个失败 block 中41个下一步请求 `PATCH`，对应41个 repair block 成功；说明示例使 PATCH 从标签变成了实际错误后重执行。actual delta 361次实现、81次未实现，精确 fetch artifact reuse 464次，重复 fetch 为0。
- 失败现象：仅10/20显式初始化 task graph；verified evidence 28个；8/13错误仍是 retrieval miss。结构化 prompt 改善了执行错误后的动作响应，但没有稳定建立 requirement graph，也没有让依赖图改变查询覆盖或 evidence 后推理。
- 是否保留：保留为开发 challenger，不替换当前 v12.1。下一轮 prompt 应优先强化首 block decomposition 和 `missed delta → requirement/query branch change`，不再增加 PATCH 示例或 BrowseComp 专用规则。

## Generic Graph Core v0（架构迭代）

- 问题：现有在线 Adapt 虽然进入了 Agent loop，但 `OnlineGraphAdaptation`、`ResearchGraphState`、控制诊断和 benchmark builder 共同硬编码了 search/fetch、document/evidence/candidate 语义。离线 execution graph 与在线 research graph 也是两套状态，图尚不是可迁移的 Agent 组件。
- 修改：新增 domain-neutral `EpisodeGraph`，统一保存 action、observation、artifact、state version 和任意 projection 节点；新增 `ToolEffectContract` 与 `ToolGraphRuntime`，通过 pure/read/write、determinism、cacheability 和 artifact kind 描述任意工具。现有 retrieval graph 改名为 `RetrievalGraphProjection`，在同一 Episode Graph 上增加 query/document/evidence 语义；retrieval diagnosis 和 expected-delta 解释从在线控制器移入 projection。新增 benchmark-neutral `GraphAgentHooks`、prompt contract 和 PTC metadata schema 组合器，BrowseComp-Plus 改由该接口接入。
- 通用性 sanity check：用 `lookup_rows`、`aggregate_values`、`update_inventory` 三个非检索工具验证了 read cache、artifact consumption、write state version 与 supersedes 关系；进一步用 domain-neutral `GoalGraphAdaptation` 完成“声明目标→表数据读取→聚合→artifact 依赖→目标完成→ANSWER readiness”的非检索闭环。核心代码不包含 search/fetch 名称或文档语义。必要的现有 online adaptation、CodeAct 和 BrowseComp-Plus 测试保持通过。
- 当前边界：该轮完成的是通用图内核和接入边界，不宣称任务性能提升。在线控制器仍使用 retrieval projection，Stage 2 离线 DependencyGraph 尚未完全迁移到同一增量内核；非检索验证目前是确定性执行 sanity check，还不是独立 benchmark 的端到端 Agent 评测。
- 下一步：把 Agent 的 goal、declared inputs/expected outputs 与实际 tool effects 建立通用 postcondition 闭环，让图负责 dependency frontier、复用和失效传播；随后接入一个非检索 Agent 任务集验证只增加 tool/domain adapter 即可运行，再回到固定20题评测行为影响。

## 补记：未完成的 mainline v13.0-v13.3

- 仓库中存在未写入日志的四轮产物。v13.0 decomposer 为6/20，v13.1 独立 decision 为3/20，v13.2 document executor 为1/20；v13.3 candidate-matrix 已生成20题但没有正式 grades/report，恢复后用同一开发 grader 重新判定为3/20。
- 共同问题：把 decomposition、decision 和 document execution 拆成更多模型调用或强制动作后，结构执行率提高，但检索覆盖和答案正确率持续下降。v13.2/13.3 尤其把图目标解释成逐文档 fetch/inspect 调度，造成窄分支反复执行，验证了“让控制器替 Agent 选择具体检索动作”不是主线。
- 处理：不恢复这些 transient 实现，也不把它们作为当前起点。固定20题上仍以 v12.1 的8/20为当前已记录最佳；Generic Graph Core 从通用依赖执行层重新迭代。

## Generic Graph Core v0：固定20题

- 结果：20/20生成和评分有效，6/20（30%）；candidate retrieval recall 41.62%，fetched-evidence recall 35.10%。相对 v12.1 的8/20丢失 qid 1234、315、991，新增 qid 255。
- 机制变化：PTC blocks 508；search 1950、fetch 298，重复 exact query 1031。对比 v12.1 的 search 1467、fetch 220、重复 query 503，执行成本和重复检索明显上升；verified evidence 从53降为31，realized/missed delta 为208/300。
- 原因：架构重构错误地同时改变了模型可见协议。通用 PTC schema 改变字段顺序和描述并增加 `input_artifacts`；在线 delta 又直接暴露 BLOCK/TOOL_ACTION/OBSERVATION/STATE_VERSION 以及 program/stdout artifact。虽然这些节点对 runtime 有用，但不属于 retrieval projection 的最小语义视图，增加了上下文噪声。这一结果不能用于否定通用内核。
- 处理：保留 EpisodeGraph、ToolEffectContract、execution projection 和非检索 GoalGraph；retrieval projection 恢复与 v12.1 相同的 PTC metadata schema，并只向模型投影原有 TASK/CONSTRAINT/CANDIDATE/EVIDENCE/QUERY/DOCUMENT/ACTION 及 search/fetch artifact。通用执行节点仍在同一图中供依赖、归因和复用使用，但不默认展示。下一轮记为 v0.1。

## Generic Graph Core v0.1：投影等价

- 结果：20/20生成和评分有效，7/20（35%）；candidate retrieval recall 39.83%，fetched-evidence recall 35.70%。相对 v0 新增 qid 896，仍低于 v12.1 的8/20。
- 机制：search 1463、fetch 373、重复 exact query 599，已从 v0 的1950/298/1031回到与 v12.1 相近的检索规模；verified evidence 46，realized/missed delta 235/206；出现5次显式 REUSE_REPLAY和1次INSPECT。说明协议等价修复有效，通用内核可以保留，但单次20题结果仍有轨迹波动且未改善候选覆盖。
- 失败分类：9个 retrieval miss、1个 candidate命中但未fetch、3个已fetch evidence后答案错误。主要瓶颈仍是候选发现，不是异常恢复或精确fetch复用。
- 反思：当前 requirement DAG 把每次 CONTINUE 限制到一个叶节点。对于多个约束共同识别对象的任务，这会把本应共享的候选发现动作拆成逐约束窄搜；图的依赖结构反而限制了多目标动作。
- 下一步：增加通用 composite-goal frontier。当多个ready子目标共享父目标且尚无候选时，父目标也成为可执行target，并向Agent提供子目标摘要；一个action可以先发现共享artifact/candidate，再分流到叶目标验证。该机制来自图的共同父节点和fan-out结构，不依赖题目关键词或检索次数，记为v0.2。

## Generic Graph Core v0.2：Composite Goal

- 结果：20/20生成和评分有效，5/20（25%）；candidate retrieval recall 30.98%，fetched-evidence recall 27.60%。search 1695、fetch 330、重复 exact query 706；verified evidence 30。29次 `COMPOSITE_GOAL_COVERAGE` 没有改善候选发现。
- 原因：把共享父目标自动放到frontier首位，相当于用图结构替Agent作语义粒度选择。父目标只携带有界子目标摘要，生成的查询更宽泛；同时宽目标的artifact难以更新叶目标状态。图能表示一个action关联多个目标，不代表Runtime应自动选择父目标。
- 处理：撤销父目标自动调度，保留通用图的多目标/共同父节点表达能力。下一轮只在声明postcondition未由实际图delta实现时释放当前target的优先权；正常产生新依赖结果时保持Agent选择的分支。SEARCH_TARGET也改为要求新增document或首次状态迁移，而不是仅执行一次search就算realized。记为v0.3。

## Generic Graph Core v0.3：Postcondition Branch Release

- 结果：固定20题生成和评分均有效，4/20（20%）；candidate retrieval recall 37.85%，fetched-evidence recall 18.00%。search 1669、fetch 229、重复exact query 774；verified evidence仅11，realized/missed delta为197/307。
- 原因：把“后置条件未实现”直接解释为“当前分支应降权”仍然是调度启发式。长链任务中的一次无新document并不代表目标无价值，轮换分支打断了语义连续性，导致已fetch evidence覆盖显著下降。
- 处理：撤销postcondition驱动的target轮换并恢复v0.1语义。后续停止增加retrieval排序规则，改为通用的依赖图工作记忆：完整轨迹仍保存在EpisodeGraph中，模型上下文只内联最近观察及当前依赖子图相关观察，其余替换为可通过artifact ID恢复的引用。

## Generic Graph Core v0.4：Dependency-Aware Working Memory（执行中）

- 问题：当前20题单个episode的模型上下文最高达到约29万字符、7.8万输入token。EpisodeGraph虽保留了artifact与依赖，但模型仍反复携带扁平的全部历史stdout；图尚未承担工作记忆选择职责。
- 技术点：新增通用GraphContextProjector。每个PTC block的完整stdout持久化为artifact；最近block和与当前目标在依赖图上相连的block保持原文，旧且无关的tool observation替换为GRAPH_MEMORY_REF，必要时由graph_load_artifact精确恢复。只改变上下文投影，不删除图数据、不改写Agent动作，也不依赖search/fetch或BrowseComp题目规则。
- 验证目标：先观察固定20题正确率是否超过当前8/20，同时比较最大上下文、输入token、候选/证据覆盖与失败类型。若只降成本而不提分，则保留为通用架构能力，但继续寻找能够改变跨域规划或验证行为的图机制。

- 结果：5/20（25%），candidate retrieval recall 34.04%，fetched-evidence recall 28.02%。最大上下文仅从v0.1的283117降至266413字符，最大输入从80592降至74546 token；search升至1750、fetch降至138、重复exact query升至1103。
- 结论：未保留为当前在线策略。当前显式依赖边不足以覆盖尚未提交为candidate/evidence、但仍影响后续语义判断的stdout；因此投影器既没有大幅压缩上下文，又使Agent更倾向重新search而不是消费既有结果。GraphContextProjector作为通用可选组件保留，但从BrowseComp在线控制器撤下，直到Agent能显式声明artifact consumption。

## Generic Graph Core v0.5：Effect-Novelty Replan（执行中）

- 问题：v0.1与v0.4均存在大量精确重复查询。现有stalled-retrieval提示依赖search/query语义，不能迁移到其他工具域；同时图已经保存了action、artifact与state effect，却没有用这些结构识别执行循环。
- 技术点：ToolGraphRuntime对内容相同的产物增加equivalent_to边；通用GraphProgressTracker按目标检查每个block是否产生非等价artifact或state mutation，并累计无进展链。连续两次仅复现既有产物时，控制层提供REPLAN机会，要求Agent改变计划或依赖路径；Runtime仍不替Agent选择工具或参数。该机制只依赖tool effect与artifact等价性，不依赖search/fetch名称。
- 验证目标：固定20题超过8/20；同时检查REPLAN是否真实出现、是否降低重复调用并改善候选覆盖。若REPLAN不被采用或只改变标签，则下一步必须把plan revision本身建成可执行图对象，而不是加强提示文字。

- 结果：8/20（40%），追平当前固定20题最佳；candidate retrieval recall 49.87%，fetched-evidence recall 45.10%，均高于v0.1。REPLAN真实出现13次；realized/missed delta从v0.1的235/206改善为278/155。正确题为896、772、1234、653、380、991、181、266。
- 边界：search 1553、重复exact query 641，未低于v0.1的1463/599；REPLAN虽被模型采用，但当前只是一项metadata动作，没有保存“旧计划被什么新计划替代”，因此无法在后续block检查Agent是否持续执行新的依赖路径。12个错误仍含9个candidate retrieval miss和3个已fetch evidence后的答案错误。

## Generic Graph Core v0.6：Explicit Plan Revision（执行中）

- 技术点：新增通用PlanRevisionLedger。REPLAN必须声明plan_revision.approach；Runtime把它保存为PLAN_REVISION节点，连接target与声明该计划的action，并用supersedes串联同一target的历次计划。最新修订会进入下一轮图signals，执行结果仍由artifact/state novelty校验。
- 目的：让“改变计划”成为可追踪、可延续、可验证的图状态，而不只是动作标签；不规定查询词、工具或benchmark语义。
- 验证目标：固定20题超过8/20。若只增加计划节点而不改善结果，则不再增强prompt或schema，转向图对候选假设的并行维护与证据覆盖验证。

- 结果：6/20（30%），candidate retrieval recall 39.88%，fetched-evidence recall 23.89%。REPLAN 14次，search/重复exact query降至1224/388，但fetch仅152，证据覆盖和正确率同时下降。
- 结论：不接入在线策略。增加plan_revision schema使Agent花费结构化输出表达计划，却没有获得新的环境信息；显式计划节点本身不等于更好的行动选择。保留PlanRevisionLedger为可选图原语，但恢复v0.5的模型可见schema与REPLAN行为。

## Generic Graph Core v0.7：Contract-Defined Semantic Novelty（执行中）

- 问题：v0.5只用完整返回值判断artifact等价。相同候选集合会因分数、顺序或附加字段变化被误判为新信息，导致图低估循环；把docid逻辑写进通用core又会退化为检索专用实现。
- 技术点：ToolEffectContract新增可选novelty_key。每个domain adapter声明“什么构成新的工具产物”，ToolGraphRuntime统一建立equivalent_to边。检索adapter以返回docid集合定义搜索产物身份；core本身不知道document、search或benchmark。其他领域可分别用row主键、页面版本、文件hash或状态版本定义等价性。
- 验证目标：在保持v0.5 schema的条件下，让等价产物循环更早触发REPLAN并超过8/20；若候选覆盖提高但正确率仍未超过，则进入通用候选假设/证据验证阶段。

- 结果：7/20（35%），candidate retrieval recall 43.19%，fetched-evidence recall 40.22%。REPLAN从v0.5的13增至17，search从1553降至1203，但候选/证据覆盖和正确率均低于v0.5。
- 结论：novelty_key保留为通用ToolEffectContract能力，但检索adapter恢复完整返回值等价语义；过早把相同docid集合判为无进展会忽略排序/分数变化对Agent的价值。在线最佳仍为v0.5。

## Generic Graph Core v0.8：Conservative Graph Answer Review（执行中）

- 问题：v0.5的12个错误中有3个已经fetch到evidence但最终答案错误；例如qid 278的verified quote明确写出“Medialab Group”，最终却输出较不精确的“Medialab”。图已保存可验证支持关系，但最终输出绕过了图一致性检查。
- 技术点：增加可选graph_answer_review。仅当图中存在source-verified evidence时，独立review读取任务约束、候选与证据子图；默认KEEP，只有证据直接证明答案错误、不完整或实体名不精确时才REVISE。review无工具、不得使用外部知识，额外调用计入模型请求与token统计。
- 泛化边界：这是goal/candidate/evidence图上的通用最终一致性检查，不依赖retrieval API或题目关键词；没有verified evidence时完全跳过，避免把缺失信息伪装成推理修复。
- 验证目标：在v0.5主机制上超过8/20，并审计每个REVISE是否由图内证据直接支持。

- 结果：6/20（30%），candidate retrieval recall 33.86%，fetched-evidence recall 28.94%。存在verified evidence的episode中review全部KEEP，其余均因无verified evidence跳过，REVISE为0。
- 结论：不进入最佳在线版本。保留为默认关闭的可选组件；当前主要瓶颈仍是上游候选发现和语义入图稀疏，final review不能从缺失的证据中创造收益。

## 固定20题选择与最终pilot100

- 固定20题最佳：Generic Graph Core v0.5，8/20（40%），追平此前v12.1的8/20；candidate retrieval recall 49.87%，fetched-evidence recall 45.10%，并真实执行13次REPLAN。
- 选择理由：v0.5首次让domain-neutral artifact equivalence与target-local progress chain改变在线动作；相比仅展示图、自动target排序、上下文裁剪、显式计划schema和final review，它在不接管工具参数的前提下取得最高正确率与最高候选/证据覆盖。
- 冻结行为：fewshot-ptc-v1、REPLAN动作、完整artifact值等价、连续无新artifact/state后暴露重规划机会；graph_answer_review关闭，retrieval novelty_key使用默认完整值，不启用context projector或显式plan_revision schema。
- 下一步：仅运行一次固定pilot100，不再根据pilot20继续调参。

## Generic Graph Core v0.5：最终pilot100结果

- 有效性：100/100生成成功、100/100评分有效，无空响应、无grader错误。结果为38/100（38%）；candidate retrieval recall 43.30%，fetched-evidence recall 33.95%。固定20题的40%不能外推到全集，pilot100确认实际收益更有限。
- 执行指标：2283个PTC blocks，6579次search、1043次fetch、2718次精确重复query、0次重复fetch；最大上下文307064字符、最大请求输入80928 token。
- 图机制：2107次CONTINUE、143次REPLAN、7次INSPECT、4次PATCH；expected delta实现/未实现为1295/988，source-verified evidence 155条，artifact 12187个。REPLAN不是只在20题偶然出现，而是在100题上持续进入真实轨迹。
- 失败归类：62个错误中，按同一qrels交集口径有46个candidate retrieval miss、5个candidate命中但未fetch evidence、11个已fetch evidence后答案错误。主要瓶颈仍是候选发现，其次是证据后的语义判断。
- 同一pilot100数据快照的当前报告：v12.1为35%，v0.5为38%，v11原始在线图版本的现存重评分报告为42%。因此v0.5相对v12.1提升3个百分点，但没有超过v11，不能宣称pilot100端到端SOTA。历史日志中v11的39%与当前report.json的42%不一致，本结论以当前100/100有效、同dataset hash的报告为准。
- 最终结论：本轮完成了从retrieval-specific sidecar到通用EpisodeGraph、ToolEffectContract、共享execution projection和effect-novelty REPLAN的架构迁移；非检索工具工作流已验证可复用同一core。有效收益来自“图识别重复effect并要求Agent重规划”，而不是更多图展示。失败的自动target轮换、context裁剪、显式plan schema和final review均保持非默认，不能作为性能贡献。

## Generic Graph Core v2：Minimal Frontier

- 目的：把通用架构收敛为 `Goal / Action / Artifact / Effect` 闭环。BrowseComp-Plus 仅作为 search/fetch 工具 adapter；核心不再要求 constraint/query/document/evidence/candidate 字段，也不再把完整 execution nodes 暴露给模型。模型仍使用冻结的 `fewshot-ptc-v1`，仅增加 action、target、expected change 和可选 input artifacts。
- 实现：复用 `EpisodeGraph`、`ToolEffectContract/ToolGraphRuntime`、`PTCExecutionProjection`、`GraphProgressTracker` 和 `GraphAgentHooks`。模型只看到 ready/blocked goals、未消费 artifact 引用和本轮新/等价 effect；目标声明为可选，Runtime 不选择工具、参数或检索目标。
- 固定20题：20/20生成和评分有效，7/20（35%）；candidate retrieval recall 40.58%，fetched-evidence recall 24.42%。493个blocks、1446次search、205次fetch、659次精确重复query；52次REPLAN，316/177次effect实现/未实现。正确题为896、772、1234、653、380、991、266，相比v0.5仅丢失qid 181。
- 结论：极简核心在显著减少图规模和模型上下文的同时基本保持结果（最大输入由v0.5约81k降到61k token），证明retrieval-specific research graph不是在线控制的必要前提；但52次REPLAN仍未改善候选发现，13个错误按qrels交集口径均为retrieval miss。保留v2作为新通用主线，不以7/20宣称性能提升。

## Generic Graph Core v2.1：Branch Frontier（执行中）

- 假设：REPLAN缺少“哪些依赖路径曾产生新effect、哪些只复现旧effect”的最小记忆。复用已有expected_change，不新增plan schema或BRANCH节点；从Action→Effect历史派生productive/exhausted paths和共享artifact，交由Agent选择替代路径。
- 边界：不做语义相似度、自动分支排序、查询改写或目标切换；branch frontier只是EpisodeGraph的紧凑投影。
- 固定20题：20/20有效，7/20（35%）；candidate retrieval recall 45.92%，fetched-evidence recall 30.88%。437个blocks、1561次search、262次fetch、583次精确重复query；25次REPLAN，314/123次effect实现/未实现。正确题为896、772、1234、653、380、160、181。
- 结论：相对v2，REPLAN和重复query减少，候选/证据覆盖提高，并新增qid 160、181，但丢失qid 266、991，最终正确率未超过7/20。保留branch frontier，因为它复用现有expected_change且改善通用路径反馈；不继续增加branch字段或语义评分。

## Generic Graph Core v2.2：Dependency Reuse（执行中）

- 假设：冻结环境中的search和fetch都是确定性只读工具。直接使用已有 `ToolEffectContract` cache语义复用相同工具+参数的artifact，可让精确重复调用不再伪装成新执行，并让effect/branch闭环更快要求新依赖路径。
- 边界：这是由tool adapter声明的通用执行属性；core不知道search、query或docid，不增加重复次数阈值和检索规则。
- 固定20题：20/20有效，7/20（35%）；candidate retrieval recall 45.99%，fetched-evidence recall 29.19%。488个blocks，但实际search从v2.1的1561降至935，精确重复query从583降至0；172次fetch，1267次artifact reuse。正确题为772、1234、653、380、991、181、1204。
- 结论：准确率仍为7/20，但确定性依赖复用在不改变工具结果的情况下消除了全部外部精确重复search，并新增qid 1204；候选覆盖基本保持。保留为通用主线能力。当前branch frontier只总结model-authored expected_change，下一轮改为直接投影EpisodeGraph中实际Tool Action→Effect路径，不引入新节点或语义规则。

## Generic Graph Core v2.3：Actual Tool Paths（执行中）

- 假设：实际工具名、参数及其novel/equivalent/reused effect比自由文本expected_change更能描述已尝试的依赖路径。由现有TOOL_ACTION、produces、equivalent_to和reuses边派生productive/exhausted paths，仍由Agent决定下一路径。
- 固定20题：20/20有效，6/20（30%）；candidate retrieval recall 45.96%，fetched-evidence recall 31.33%。
- 结论：不保留。实际tool arguments增加了模型可见细节但没有提高候选覆盖，正确率反而下降。撤回该投影，停止继续扩展branch capsule；主线恢复v2.2的expected-change branch和确定性依赖复用。

## Generic Graph Core v2.4：Causal Working Memory（执行中）

- 假设：工作记忆必须建立在真实artifact consumption上，而不是按时间或检索计数裁剪。ToolGraphRuntime从任意工具结果的嵌套值推断后续参数依赖；PTCExecutionProjection把block内tool artifact与持久Python state连接。只有一个旧block的tool artifacts都已被后续action/state消费时，才把其stdout替换为可恢复GRAPH_MEMORY_REF。
- 边界：完整artifact和图不删除；最近8个block、未消费artifact相关block及没有闭合依赖的block始终保留。该机制不识别query/document/evidence，也不根据token阈值或benchmark规则裁剪。
- 固定20题：20/20有效，7/20（35%）；candidate retrieval recall 38.65%，fetched-evidence recall 28.97%。448个blocks、695次实际search、246次fetch；最大上下文229853字符、最大输入65890 token，均未低于v2.2的218811字符和62229 token。
- 结论：嵌套artifact依赖和artifact→Python state派生边保留为通用provenance能力；自动工作记忆投影不保留在默认Controller中。它没有减少本轮最大上下文，也降低了候选覆盖，继续调裁剪阈值会偏离简洁主线。

## Generic Graph Core v2.5：Unified Failure Recovery（执行中）

- 假设：failure不需要独立repair controller。执行投影已生成FAILURE节点，通用Controller只需把最近失败作为active frontier的PATCH机会；Agent在下一PTC block修正代码或依赖假设并重执行，Runtime验证PATCH是否成功。
- 边界：不自动改代码、不触发额外模型、不针对异常类型写规则，也不恢复旧的retrieval-specific repair/replay管线。failure、action、effect仍在同一EpisodeGraph和同一控制闭环中。
- 固定20题：20/20有效，6/20（30%）；candidate retrieval recall 39.94%，fetched-evidence recall 32.75%。483个blocks中有23个执行失败，但action分布为468 CONTINUE、15 REPLAN、0 PATCH；正确题为1234、653、380、315、181、266。
- 结论：保留FAILURE→PATCH作为通用Controller的最小恢复能力，但本轮模型没有采用PATCH，不能归因出在线修复收益，也不围绕异常类型增加规则。该版本不作为候选最优版。

## Generic Graph Core v2.6：Implicit Graph Core（执行中）

- 假设：BrowseComp任务中模型从未调用手工goal/artifact API，target也始终为task。让Runtime自动维护依赖图，模型只声明action和expected_change，可以减少无效图操作面，同时保留effect验证、branch frontier、确定性复用和failure recovery。
- 边界：完整GoalGraphAdaptation API仍可由需要显式子目标的其他tool domain启用；BrowseComp adapter仅暴露search/fetch。核心不新增节点或字段，反而从该adapter的prompt、manifest和PTC schema移除未使用接口。
- 固定20题：20/20有效，7/20（35%）；candidate retrieval recall 41.46%，fetched-evidence recall 34.02%。483个blocks、870次实际search、159次fetch；最大上下文212569字符、最大输入57778 token。正确题为772、1234、653、380、991、181、266。
- 机制：10个执行失败后模型选择9次PATCH，9次均由下一block成功执行验证；这是统一FAILURE→PATCH首次在通用Controller中形成真实在线闭环。精简接口同时把最大输入降到所有v2.x版本最低。
- 结论：保留Implicit Graph Core为默认主线。它恢复到7/20但仍未追平8/20；13个错误中多数仍没有命中gold candidate，且结构性artifact novelty会把“新但无关”的结果误判为progress。下一轮不增加检索规则，而是让Agent的REPLAN选择成为对上一effect的通用语义反馈。

## Generic Graph Core v2.7：Agent-Semantic Effect Feedback（执行中）

- 假设：Runtime能判定artifact是否新颖，却不能判定其是否回答了语义目标；Agent恰好具备该判断。每轮都允许Agent在结果语义不足时选择REPLAN；该选择把上一expected-change path标为agent-rejected，branch frontier随后把它作为exhausted path保留。
- 边界：不新增PTC字段、模型调用、query规则或语义评分器；结构性effect统计保持独立，Agent只提供一个已有动作所携带的语义反馈，Runtime负责持久化和投影。
- 固定20题：20/20有效，5/20（25%）；candidate retrieval recall 36.99%，fetched-evidence recall 31.61%。REPLAN增至57次、实际search降至731次，但正确率和候选覆盖均下降。
- 结论：不保留。始终暴露REPLAN把“可选择的语义反馈”变成了过强的动作暗示，模型过早离开仍有价值的路径；恢复v2.6仅在结构性effect连续不新颖时暴露REPLAN。下一轮不再改变动作频率，而是验证结构化产物中是否存在可复用的未消费依赖。

## Generic Graph Core v2.8：Implicit Subgoal Branches（执行中）

- 假设：多跳任务需要把effect history按语义依赖分支隔离，但不需要完整task-graph schema或显式graph API。PTC只增加一个短target标签；Runtime在首次出现时自动建立GOAL节点，相同标签复用同一分支，branch frontier按target局部反馈。
- 边界：不要求预先列出全部目标，不增加constraint/evidence/candidate类型、依赖关系字段或额外模型调用；目标仍由Agent做语义命名，Runtime只保证稳定节点、局部effect历史和最多四个活跃分支的紧凑投影。
- 固定20题：20/20有效，3/20（15%）；candidate retrieval recall 40.70%，fetched-evidence recall 25.31%。67次REPLAN、12次INSPECT，最大输入71356 token；正确题仅1234、653、181。
- 结论：不保留。即使只增加一个自由target标签，模型仍会碎片化目标并在局部branch间频繁切换；它重复了旧显式task graph的复杂度问题。恢复v2.6的单root target，不再向模型schema增加语义图字段。

## Generic Graph Core v2.9：First Equivalent Effect Replan（执行中）

- 假设：确定性复用已经证明等价effect不会增加信息；等到连续两次等价effect才提示REPLAN会浪费一个动作。把通用阈值从2降为1，在第一次结构性停滞时暴露同一branch frontier，其余架构保持v2.6不变。
- 边界：仍不对新但无关的artifact做语义判断，不改变工具、query、目标或程序；这是单一通用控制参数，不增加schema和组件。
- 固定20题：20/20有效，4/20（20%）；candidate retrieval recall 37.21%，fetched-evidence recall 30.03%。REPLAN激增至111次，正确题为1234、653、380、266。
- 结论：不保留。第一次等价effect就提示会让模型围绕同一结构性停滞反复声明REPLAN，而不是产生更好的新路径；恢复v2.6的stagnant streak=2。阈值继续微调没有通用价值，停止此方向。

## 本轮最终选择：通用主线与结果最优版分开记录

- 通用架构主线选择v2.6 Implicit Graph Core：单EpisodeGraph、ToolEffectContract、PTCExecutionProjection和一个GoalGraphAdaptation Controller；BrowseComp adapter只暴露search/fetch，模型schema仅增加action和expected_change。保留嵌套artifact consumption、artifact→persistent state provenance、确定性复用、stagnant streak=2的branch frontier，以及同图FAILURE→PATCH。
- 固定20题结果：v2.6为7/20（35%），没有超过项目既有固定20题最好v0.5的8/20（40%）。因此不能宣称本轮通用化提高了端到端准确率；它的实证收益是9/10失败进入成功PATCH、实际重复search被缓存，以及较小的模型接口与上下文。
- 结果最优版仍为v0.5：固定20题8/20；其唯一一次最终pilot100已完成，结果38/100，且未超过现存v11重评分报告42%。本轮不重复运行pilot100，因为新通用版本没有越过固定20题最好结果，重复大样本只会增加成本而不改变版本选择。
- 停止理由：v2.3的实际tool path、v2.4的自动工作记忆、v2.7的持续语义REPLAN、v2.8的隐式子目标和v2.9的提前REPLAN均无增益或明显退化。继续添加member frontier、semantic evidence或检索特定字段会重新走向BrowseComp专用research graph，违背本轮“简洁、通用、逐步反馈”的原则。
