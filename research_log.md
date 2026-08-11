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