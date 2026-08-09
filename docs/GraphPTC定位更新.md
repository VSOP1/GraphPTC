Exit code: 0
Wall time: 1.4 seconds
Output:
**GraphPTC 的核心定位应该从“异常后的局部修复器”升级为“基于执行图的 PTC 自适应控制层”**。

你现在的结果已经说明两点：

1. **选择性回放机制本身有效**：5 次修复中复用 62/76 个工具调用，局部复用率约 82%。
2. **但异常修复不是 BrowseComp-Plus 的主要瓶颈**：真正造成失败和成本膨胀的是错误检索方向、重复搜索、证据利用不足、逻辑错误和停止判断失败。

因此，不应放弃 selective replay，而应把它从 GraphPTC 的“唯一主要功能”降为多个执行动作之一。

## 新的核心问题

原来的研究问题是：

> PTC 程序报错后，如何定位错误、最小修改并选择性重执行？

新的研究问题应扩展为：

> 如何把 PTC 的中间执行过程组织成动态依赖图，使 Agent 能够判断任务进展、识别运行时与语义层面的低效或错误，并基于已有计算结果选择继续、检查、修复、复用或终止？

也就是说，GraphPTC 不再只处理：

```text
Python exception
→ failure attribution
→ patch
→ selective replay
```

而是处理更一般的：

```text
执行结果
→ 任务进展评估
→ 图引导诊断
→ 自适应动作选择
```

## 新的完整闭环

建议将主流程改成：

[
\boxed{
Generate
\rightarrow
Execute
\rightarrow
Trace
\rightarrow
Assess
\rightarrow
Diagnose
\rightarrow
Adapt
}
]

其中 `Adapt` 包含五类动作：

```text
1. ANSWER
   当前证据已经充分，生成最终答案

2. CONTINUE
   生成下一个 PTC block，继续未完成的研究

3. INSPECT
   查询图中的节点、依赖路径、中间结果或证据来源

4. PATCH
   修改存在运行错误或逻辑错误的程序片段

5. REUSE / REPLAY
   复用仍有效的历史计算，只执行新增或失效部分
```

这样 selective replay 仍然保留，但它服务于更大的自适应执行过程。

## 需要覆盖三类失败

### 1. Execution Failure

显式异常：

* Python exception；
* 参数错误；
* 工具失败；
* timeout；
* 类型不匹配。

这仍然走现有的：

```text
Failure node
→ backward slice
→ patch
→ invalidation
→ selective replay
```

### 2. Semantic Failure

程序成功结束，但任务结果不正确：

* 检索到错误对象；
* 证据支持了错误候选；
* 筛选条件错误；
* 聚合逻辑错误；
* 使用了不相关文档；
* 输出与问题约束不匹配；
* 候选之间存在冲突却直接回答。

这类失败没有异常节点，因此需要由 Agent 根据输出和图摘要主动触发诊断：

```text
Agent 判断结果可疑
→ 查询候选答案的来源节点
→ 检查支持路径
→ 定位错误检索或错误 transform
→ patch 或生成替代研究分支
```

### 3. Progress Failure

程序未必错误，但研究过程陷入低效：

* 重复 query；
* 重复 fetch；
* 连续多个 block 没有新增 docid；
* 搜索结果高度重叠；
* 候选和证据长期没有变化；
* 工具调用很多，但 evidence recall 不再增加；
* 剩余预算不足，却仍重复已有路径。

这是当前 BrowseComp-Plus 最主要的问题。

GraphPTC 应能够将这些现象组织成可供 Agent 判断的进展信息，例如：

```text
最近 4 个 blocks：
- 28 次搜索
- 21 次返回已见文档
- 7 次重复 fetch
- 0 个新候选
- 1 条新证据
```

然后 Agent可以选择：

```text
改变查询方向
检查已有候选
复用已有文档重新聚合
停止当前分支
直接回答
```

## Graph 的角色也要改变

原设计中的图主要是程序依赖图：

```text
Tool Call
→ Transform
→ Tool Call
→ Output
```

新定位下，它还需要表达**任务级研究状态**。不一定现在就实现全部具体节点，但概念上至少包含两层。

### Execution Layer

记录程序怎样执行：

* PTC block；
* tool call；
* transform；
* control/data dependency；
* artifact；
* program version；
* reuse/replay。

这层负责精确归因和重执行。

### Research Layer

记录任务怎样推进：

* query；
* retrieved document；
* fetched evidence；
* candidate answer；
* supporting/refuting relation；
* unresolved requirement；
* final claim。

例如：

```text
Query Q3
   ↓ retrieves
Document D8
   ↓ contains
Evidence E4
   ↓ supports
Candidate C2
   ↓ contributes to
Final Answer
```

如果最终答案错误，Agent 可以追问：

```text
这个 candidate 来自哪些 evidence？
这些 evidence 来自哪些 query 和 document？
是否存在相反证据？
哪些问题约束尚未覆盖？
```

这比只看到变量级 Def-Use 更适合语义失败诊断。

## 选择性复用也不应只发生在 Patch 后

当前逻辑是：

```text
代码报错
→ patch
→ selective replay
```

应扩展为三种复用。

### Patch Reuse

代码修改后，复用未失效工具结果。
这是你当前已经验证有效的部分。

### Continuation Reuse

Agent 生成下一个 PTC block 时，直接使用历史 graph artifacts，而不是重新搜索或重新 fetch：

```python
docs = graph.load_artifacts(doc_ids)
evidence = graph.load_evidence(candidate_id)
```

这样可以直接缓解重复检索。

### Alternative-Branch Reuse

Agent 判断当前候选方向错误，转向另一个候选时：

* 复用公共的初始查询；
* 复用已经获取的文档；
* 仅执行新区分性查询；
* 从已有证据图建立替代推理路径。

这比“从头生成下一个 PTC block”更接近 GraphPTC 的独特价值。


## 研究贡献可以重新组织为三项

### 1. Persistent Dynamic Execution Graph

跨 PTC blocks 统一记录：

* 程序执行；
* 工具调用；
* artifact；
* 数据和控制依赖；
* 研究证据和候选来源；
* 程序版本与复用关系。

### 2. Graph-Guided Agent Adaptation

Agent 不只在 exception 后收到 failure capsule，而是在每个 block 后获得紧凑的 graph observation，并自主选择：

```text
answer / continue / inspect / patch / replay
```

显式异常、语义错误和进展停滞都可以触发图查询。

### 3. Dependency-Aware Reuse and Execution

图用于：

* Patch 后选择性重执行；
* 后续 PTC block 复用已有 artifact；
* 避免重复 search/fetch；
* 对替代研究路径共享已有证据；
* 只执行新增或真正失效的子图。

## 下一阶段不应先实现复杂语义评分器

要避免把 GraphPTC 变成一堆针对 BrowseComp-Plus 的硬编码规则，例如：

```text
连续三次没新 docid就强制停止
搜索超过十次就禁止继续
candidate 数量达到两个就回答
```

更合理的是：

1. Runtime 确定性地计算通用图统计；
2. 将这些统计和局部子图提供给 Agent；
3. Agent进行语义判断；
4. Agent选择下一动作。

例如 Runtime 可以提供：

```text
new_docids
repeated_docids
new_evidence
repeated_fetches
candidate_changes
unresolved_constraints
dependency summary
remaining budget
```

但不直接决定“答案已经正确”。

这能保持：

> 图负责可观测性和可执行性，Agent 负责语义决策。


## 最关键的定位变化

原来是：

> **GraphPTC 是一个 PTC failure recovery system。**

现在应改成：

> **GraphPTC 是一个以动态执行图为核心的 PTC runtime and control layer；故障恢复只是其中一个应用。**

这样 BrowseComp-Plus 上没有大量 Python exception 也不会削弱论文动机。相反，它能验证更重要的问题：

> PTC 即使执行成功，也可能逻辑错误、证据不足或持续进行低收益搜索；GraphPTC 能否利用执行图帮助 Agent理解已经做过什么、结果从哪里来、哪些计算仍可复用，以及下一步真正需要执行什么。

## 当前实验边界

当前结果支持的结论是：selective replay 在局部执行层面有效，但显式异常不是 BrowseComp-Plus 的主要瓶颈。Active pilot20 未通过 outcome Gate，因此 GraphPTC 仍保持 opt-in，不能宣称端到端准确率提升。

研究定位应改为：

> 以动态执行图为核心的 PTC runtime and control layer，面向执行 provenance、研究进展判断、依赖感知复用和局部恢复。

故障恢复仍然保留，但不再是唯一主线。当前框架需要同时覆盖 execution failure、semantic failure 和 progress failure。

## 目标框架

```text
Generate
  -> Execute
  -> Trace
  -> Assess
  -> Diagnose
  -> Adapt
```

其中：

- Execution Layer：记录 block、tool、transform、state、artifact、版本和回放依赖。
- Research Layer：逐步增加 query、document、evidence、candidate 与 unresolved frontier 的来源关系。
- Control Layer：基于图摘要选择 continue、inspect、reuse、patch 或终止。
- Evaluation Layer：分别评估结构正确性、因果修复收益、检索效率和最终答案质量。

Graph 负责可观测性、来源追踪和安全执行；Agent 负责语义判断。两者边界不能混淆。

## 下一步路线

1. **冻结轨迹消融**：用相同失败前缀比较 no-repair 与 repair，先确认 block-level 因果收益。
2. **研究层投影**：在离线模式增加检索、文档、证据和消费关系，验证图是否提供超出简单调用计数的诊断能力。
3. **Progress shadow**：识别重复检索、低新增证据和预算风险，但不改变 prompt、tool schema 或 runtime。
4. **最小在线 challenger**：先验证语义保持的 artifact reuse，再分别测试 evidence compaction、alternative branch 和 progress control。
5. **重复配对 Gate**：在新鲜 control/active 配对上重复验证；未通过前不扩大到 pilot100，也不将 Active 设为默认。

每一步都必须保留 frozen `original-ptc-v1`、`fewshot-ptc-v1` prompt 约束、运行签名、原始事件和开发结果标签。

## 创新方向

优先考虑三类、但不要同时实现：

- **Provenance checkpoint**：把跨 block 的公共证据子图持久化，支持替代研究分支共享已有结果。
- **Counterfactual graph replay**：从同一执行前缀比较不同动作，区分 repair、reuse 和继续搜索的因果效果。
- **Evidence frontier**：维护已支持、冲突和未解决的任务约束，让下一次检索针对缺口，而不是重复已有路径。

其中前两项适合近期验证，Evidence frontier 需要更强的语义判断，应作为后续创新，不应成为当前阶段的前置依赖。

## 方案原则

- 不重写现有 Stage 1-5；先以旁路组件扩展能力。
- 不把 BrowseComp-Plus 规则硬编码成通用控制器。
- 不用一次配对结果证明因果收益。
- 新增会改变模型可见行为的控制器必须作为独立 challenger。
- 先证明 graph-specific diagnostic lift，再增加复杂语义组件。

