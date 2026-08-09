# GraphPTC：基于动态依赖图的可归因、增量修复和重执行 PTC

> 本版本只描述 GraphPTC 的整体实现框架、核心部件及其协作关系，不预先限定具体的数据结构、存储后端、插桩方式、接口格式或重执行策略。各部件将在原始 PTC 能力之上逐步接入和迭代。

# 一、Motivation

## 背景

Programmatic Tool Calling（PTC）的核心思想是让 LLM 一次生成一段可执行程序，在代码执行环境内部连续调用多个工具，而不是每调用一次工具都重新经过一次 LLM。

PTC 对正常成功执行具有明显的效率优势：中间工具结果可以直接返回正在运行的程序，由程序继续处理，而不必逐个进入 LLM 上下文。

## 核心问题

本工作关注的不是 PTC 能否获得中间结果，而是：

> **如何将 PTC 的中间执行状态组织成可供后续 Agent 进行依赖分析、故障归因、增量修复和选择性重执行的显式结构。**

现有 PTC 主要缺少两类能力：

1. 中间执行过程缺乏结构化可观测性，失败后难以定位具体错误来源；
2. 修复失败程序时，缺少对已成功计算结果的系统性复用机制。

因此，GraphPTC 不改变 PTC“由程序连续调用工具”的基本执行范式，而是在其外部增加一层面向执行过程的结构化管理机制。

# 二、Design Goal

GraphPTC 的目标是构造一个能够覆盖完整 PTC episode 的执行管理框架，使系统具备以下能力：

- 将一次或多次 PTC block 的执行过程表示为统一的动态依赖图；
- 记录关键中间计算、工具调用和执行状态之间的依赖关系；
- 在失败后从错误位置追溯相关因果路径；
- 仅向 LLM 提供与当前故障相关的局部信息；
- 将 LLM 的修改限制在与故障相关的最小程序区域；
- 判断修改会影响哪些已有执行结果；
- 复用未受影响的中间结果，并仅重新执行必要部分；
- 在涉及有状态工具和副作用时保证重执行安全性。

整体闭环为：

\[
\boxed{
Execute
\rightarrow
Trace
\rightarrow
Attribute
\rightarrow
Patch
\rightarrow
Invalidate
\rightarrow
Selective\ Replay
}
\]

该思想与 Self-Adjusting Computation 的基本思想一致：记录计算之间的动态依赖，并在输入或程序发生变化后，仅更新受影响的部分。

# 三、Overall Architecture

```text
User Task
   │
   ▼
LLM generates PTC
   │
   ▼
PTC Adaptation Layer
   │
   ├── Program Analysis
   ├── Execution Instrumentation
   └── Runtime Coordination
   │
   ▼
Original PTC Runtime ───────► Tools
   │
   ▼
Execution Events and Intermediate Artifacts
   │
   ▼
Dynamic Dependency Graph
   │
   ├── Success ─────────────► Final Output
   │
   └── Failure
          │
          ▼
   Failure Attribution
          │
          ▼
   Local Repair Context
          │
          ▼
   LLM Minimal Patch
          │
          ▼
   Change and Invalidation Analysis
          │
          ▼
   Reuse Unaffected Results + Re-execute Affected Parts
          │
          ▼
   Updated Execution Graph
```

GraphPTC 由以下核心部件构成。

## 1. Original PTC Adapter

负责将 GraphPTC 接入现有 PTC 执行流程，而不要求重写原始 PTC runtime。

主要职责：

- 接收 LLM 生成的原始 PTC 程序；
- 将程序交给后续分析和执行管理部件；
- 对接原始工具注册、代码执行和结果返回机制；
- 保持 GraphPTC 与具体模型、工具平台和 PTC runtime 解耦。

该部件的核心目标是先建立一个稳定的适配边界，使后续图构建、归因和重执行能力能够逐步增加。

## 2. Program Analysis Component

负责在程序执行前建立代码结构与潜在依赖关系的初始表示。

主要职责：

- 识别程序中的关键执行单元；
- 建立代码位置、工具调用和中间计算之间的映射；
- 提取后续动态追踪所需的静态结构信息；
- 为运行时事件与源代码之间的对齐提供依据。

该部件只负责提供候选结构，不单独决定最终执行图。最终依赖关系需要结合实际运行轨迹确定。

## 3. Execution Instrumentation Component

负责在不改变原始程序语义的前提下，使关键执行行为能够被 GraphPTC runtime 观察。

主要职责：

- 捕捉关键工具调用和中间计算事件；
- 记录事件的输入、输出、状态及其代码位置；
- 将运行时实例与程序分析阶段识别出的静态位置关联起来；
- 将执行事件交给图构建和 artifact 管理部件。

该部件只规定需要捕捉什么，不在当前阶段限定采用何种插桩技术。

## 4. GraphPTC Runtime

GraphPTC Runtime 是整个框架的执行协调中心。

主要职责：

- 接收程序执行过程中产生的事件；
- 维护当前 episode、PTC block 和执行节点之间的关系；
- 协调工具执行、artifact 保存、图更新和故障处理；
- 在修复后控制结果复用和必要的重新执行；
- 保证多次 PTC block 共享同一条 provenance 链。

GraphPTC Runtime 不取代原始 PTC runtime，而是在其上层管理执行状态和依赖关系。

## 5. Dynamic Dependency Graph Manager

负责构造和维护 GraphPTC 的核心执行图。

主要职责：

- 将静态程序信息和动态执行事件融合为统一图结构；
- 表示关键执行节点及其依赖关系；
- 支持跨 PTC block 的依赖连接；
- 支持局部图查询、反向追踪、影响范围分析和版本对比；
- 在程序修复和重新执行后更新图状态。

图管理部件应与具体图数据库或内存结构解耦，先确定统一的图语义，再逐步选择实现方式。

## 6. Artifact and State Manager

负责管理执行过程中产生的中间结果和外部状态信息。

主要职责：

- 保存可供后续复用或检查的中间 artifact；
- 将 artifact 与其来源节点、版本和执行状态关联；
- 为跨 block 使用已有结果提供稳定引用；
- 区分程序内部值、工具返回结果和外部环境状态；
- 为结果有效性判断和选择性重执行提供依据。

该部件只规定 artifact 的生命周期和引用关系，不预先限定具体存储后端。

## 7. Failure Attribution Engine

负责将执行失败转换为可供 Agent 理解和修复的局部因果上下文。

主要职责：

- 确定失败对应的执行节点或程序区域；
- 从失败位置沿依赖图反向追踪相关前驱；
- 提取与错误直接相关的代码、依赖节点和 artifact 摘要；
- 区分程序运行异常、工具执行错误以及由上游错误数据传播导致的失败；
- 控制暴露给 LLM 的信息范围，避免默认提供完整执行历史。

该部件的输出是结构化的局部修复上下文，而不是固定格式的错误报告。

## 8. Patch Controller

负责约束和管理 LLM 对失败程序的修改。

主要职责：

- 向 LLM 提供原始任务、失败位置、相关依赖路径和局部代码；
- 要求 LLM 优先生成局部、最小的程序修改；
- 将修改映射回原始程序；
- 识别修改涉及的程序区域和图结构变化；
- 保存不同程序版本及其与执行图之间的对应关系。

Patch Controller 不规定补丁必须采用何种文本格式，只要求修改范围可定位、可比较和可回滚。

## 9. Invalidation Analyzer

负责判断程序修改会使哪些已有结果失效。

主要职责：

- 分析修改影响的代码区域和执行节点；
- 根据依赖关系向下游传播失效状态；
- 区分仍然有效、需要重新验证和必须重新执行的结果；
- 将程序变化、输入变化和外部状态变化纳入统一影响分析；
- 为选择性重执行生成最小必要执行范围。

该部件只描述失效分析的逻辑目标，不在当前阶段固定具体传播算法。

## 10. Selective Replay Controller

负责在修复后协调缓存复用和重新执行。

主要职责：

- 复用未受修改影响且仍然有效的中间结果；
- 重新执行失效节点及其必要下游计算；
- 保持重新执行结果与旧图之间的版本关系；
- 避免重复触发不应重复发生的外部副作用；
- 在无法安全局部恢复时回退到更大范围的重执行或环境重置。

该部件需要将“计算依赖是否失效”和“工具调用是否适合重放”作为两个独立问题处理。

## 11. Episode Coordinator

负责管理完整任务生命周期。

主要职责：

- 为一个用户任务建立统一的 episode；
- 允许 Agent 自然生成一个或多个 PTC block；
- 保证所有 block 的程序版本、执行图、artifact 和修复记录属于同一任务；
- 管理 episode 的开始、继续、修复、重执行和结束；
- 对外提供最终结果及必要的执行摘要。

# 四、Dynamic Dependency Graph Abstraction

## 1. 分层表示

GraphPTC 采用 episode、PTC block 和 execution node 的分层表示：

```text
PTC Episode
│
├── Block 1
│   ├── Execution Node
│   ├── Execution Node
│   └── ...
│
├── Block 2
│   ├── Execution Node
│   └── ...
│
└── Cross-block Dependencies
```

图的生命周期覆盖完整 episode，而不是局限于单次代码执行请求。

## 2. 节点抽象

GraphPTC 不需要把每一个底层 Python operation 都转化为图节点，而应只保留对故障归因、结果复用和重执行有意义的关键计算。

核心节点类型包括：

- **TOOL**：外部工具调用；
- **TRANSFORM**：对工具输入、控制条件、迭代过程或最终结果有实质影响的中间计算；
- **OUTPUT**：最终暴露给 LLM 或用户的结果；
- **STATE**：在有状态环境中需要显式追踪的外部状态变化。

不同节点共享统一的身份、来源、状态、版本和 artifact 引用语义，但当前阶段不固定具体字段设计。

## 3. 边抽象

图中主要表示以下关系：

- **数据依赖**：一个节点的输出被另一个节点使用；
- **控制依赖**：一个节点的执行由条件、分支或迭代状态决定；
- **状态依赖**：一个节点依赖外部环境或前序工具调用产生的状态；
- **版本关系**：修复前后的程序节点或执行节点之间存在对应关系。

边的目标是支持因果追踪和影响传播，而不是完整复刻 Python 的所有执行语义。

## 4. 构图思路

GraphPTC 使用“执行前程序分析 + 运行时事件追踪”的组合方式构图：

1. 程序分析部件先建立关键代码区域和潜在依赖的静态骨架；
2. 执行插桩部件记录实际发生的工具调用和中间计算；
3. Graph Manager 将静态位置与动态实例对齐；
4. 仅将实际发生并对后续计算有影响的关系写入执行图；
5. 多次执行和循环产生的动态实例共享静态来源，但保留独立执行身份。

图提取过程应尽量确定化，不依赖 LLM 自行解释程序依赖。

## 5. 图与 Artifact 生命周期

执行图、程序版本和 artifact 的生命周期与 episode 对齐。

框架需要同时支持：

- 持久保存图结构和执行元信息；
- 保存较大的原始中间结果；
- 在运行期间快速进行局部图查询；
- 在不同 PTC block 之间恢复已有图和 artifact；
- 在 episode 结束后保留完整 provenance 以供评测和分析。

具体持久化技术在后续实现阶段决定。

# 五、Graph-Guided Failure Attribution

GraphPTC 从失败节点出发进行反向依赖追踪，只向 Agent 提供与当前故障相关的局部因果子图，而不是完整 PTC 程序和全部执行历史。

归因过程包括：

1. **Failure Anchoring**：将异常、工具错误或无效结果定位到对应执行节点和源代码区域；
2. **Backward Dependency Tracing**：追踪产生失败输入或控制条件的关键上游节点；
3. **Context Reduction**：移除与当前故障无关的执行分支；
4. **Artifact Summarization**：提供必要的中间结果摘要，并允许在需要时进一步展开；
5. **Repair Context Construction**：形成可直接交给 LLM 的局部诊断上下文。

默认提供最小必要信息；当局部信息不足时，再允许 Agent 按需扩展相关节点、artifact 或代码范围。

# 六、Dependency-Aware Patch

GraphPTC 的修复目标不是让 LLM 重新生成整段 PTC，而是优先修改与故障相关的最小代码区域。

Patch Controller 需要完成以下工作：

1. 从 Failure Attribution Engine 接收局部因果上下文；
2. 组织原始任务、失败信息、相关代码和可复用结果；
3. 请求 LLM 生成局部修改；
4. 验证修改能够被定位到原程序；
5. 建立修复前后程序版本之间的对应关系；
6. 将程序变化交给 Invalidation Analyzer。

当局部修改不足以解决问题时，框架可以逐步扩大修复范围，但不应默认退化为整段程序重写。

# 七、Dependency-Aware Invalidation and Selective Replay

## 1. 失效分析

修复后，GraphPTC 需要判断哪些已有执行结果仍然有效。

基本思路为：

- 以被修改的程序区域及其对应执行节点为起点；
- 沿数据依赖、控制依赖和状态依赖向下游传播影响；
- 将节点划分为可直接复用、需要重新验证和需要重新执行的集合；
- 生成本轮修复对应的最小重执行范围。

## 2. 选择性重执行

Selective Replay Controller 在重新运行修复后的程序时：

- 对仍然有效的历史计算返回已有结果；
- 对已失效的计算重新执行；
- 将新结果写入新的执行版本；
- 更新受影响的图节点和依赖关系；
- 保持未受影响的历史 provenance 不变。

框架可以采用“程序从统一入口重新进入、runtime 在关键边界决定复用或执行”的总体思路，也可以在后期探索更细粒度的子图调度方式。

## 3. 有状态工具与副作用

对有状态工具，GraphPTC 需要额外构造副作用安全层，主要解决：

- 某个成功工具调用是否可以直接复用其结果；
- 某个调用是否可以安全重复执行；
- 程序修改是否使此前已经发生的副作用变得无效；
- 当前环境是否仍与缓存结果一致；
- 无法局部恢复时是否需要重置环境并扩大重执行范围。

当前阶段只定义这些判断维度，具体工具分类和策略可在 ToolSandbox 等环境中逐步实现。

# 八、Cross-Block Execution

GraphPTC 允许 Agent 根据任务复杂度自然决定 PTC block 数量，但所有 block 共享同一个 episode-level execution graph。

跨 block 机制需要满足：

- 后续 block 可以引用前序 block 产生的 artifact；
- 新的计算可以与前序节点建立显式依赖；
- 前序 block 失败后产生的修复不会丢失已有 provenance；
- 容器生命周期与图生命周期相互独立；
- 即使原始执行环境发生变化，GraphPTC 仍能恢复任务级执行状态。

因此，跨 block 复用不应只依赖 Python 进程或 container 中仍然存在的变量，而应由 GraphPTC 的 episode 状态统一管理。

# 九、End-to-End Workflow
完整执行流程如下：

1. 用户提交任务； 
2. Agent 根据当前任务状态生成一个 PTC block； 
3. GraphPTC Adapter 将 PTC 程序交给分析与插桩部件，建立静态代码位置与运行时事件之间的映射； 
4. 原始 PTC runtime 执行插桩后的程序，并在程序内部调用外部工具； 
5. GraphPTC Runtime 收集关键执行事件、中间 artifact、工具调用结果和程序运行状态； 
6. Graph Manager 根据运行时信息增量构造或更新动态依赖图； 
7. PTC block 执行结束后，Runtime 向 Agent 返回执行观察；
8. Agent 对当前执行结果进行判断。即使程序没有报错，也需要进一步判断： 
  - 当前输出是否真正回答了用户任务； 
  - 信息是否完整、可信且相互一致； 
  - PTC 程序是否存在逻辑错误或错误的数据处理； 
  - 是否还需要调用其他工具获取补充证据； 
  - 是否已经满足最终终止条件。 
9. Agent 根据当前观察自主选择以下动作之一：
- 输出最终答案； 
- 生成下一个 PTC block； 
- 查询 Graph； 
- 继续生成 Patch。
10. 对于显式执行失败，Failure Attribution Engine 从失败节点出发构造局部因果上下文，并将相关代码、依赖关系和 artifact 摘要提供给 Agent； 
11. 对于没有运行时异常但结果不合理的情况，Agent 可以根据输出异常、信息冲突或任务未完成状态，主动查询相关图节点，并定位可能的逻辑错误区域； 
12. 当 Agent 选择生成 Patch 时，Patch Controller 基于原程序和补丁建立新的程序版本； 
13. Invalidation Analyzer 根据程序修改和动态依赖关系，计算可能受到影响的节点与结果范围； 
14. Selective Replay Controller 复用仍然有效的历史结果，并重新执行受影响的程序部分； 
15. Graph Manager 将新的执行事件、复用关系和程序版本信息更新到当前执行图中； 
16. 重执行完成后，Agent 再次观察新的执行结果并选择下一步行动； 
17. 若仍失败，则继续新的归因—修复—重执行循环；
18. 整个过程持续到以下任一终止条件满足： 
- Agent 判断任务已经正确完成； 
- 达到最大 PTC block 数量； 
- 达到最大修复次数； 
- 达到工具调用、时间或 token 预算； 
- 当前错误无法通过局部修复安全恢复。

# 十、Incremental Implementation Roadmap

GraphPTC 应建立在可运行的原始 PTC baseline 上，并分阶段增加能力，避免一开始同时实现所有部件。

## Stage 0：Original PTC Baseline

目标：得到一个可稳定运行和评测的原始 PTC 框架。

需要具备：

- LLM 生成 PTC；
- 程序能够连续调用工具；
- 记录最终成功、失败、token 和执行开销；
- 支持 baseline 与 GraphPTC 在同一固定语料、检索器和 grader 下端到端运行。

该阶段不引入图、归因或选择性重执行。

## Stage 1：Execution Observability Skeleton

目标：在不改变原始 PTC 行为的情况下，捕捉关键执行事件。

需要构造：

- 原始 PTC Adapter；
- Program Analysis Component 的最小版本；
- Execution Instrumentation Component；
- GraphPTC Runtime 的事件接收框架；
- episode、block 和执行事件的基础生命周期。

该阶段只保证“能观察”，不要求执行结果复用。

当前实现状态：待基于冻结的 Original PTC baseline 从头实现。仓库不保留早期 Stage 1
原型，避免旧 Agent、prompt、retriever 和 runtime 语义污染新的 GraphPTC 实现。

## Stage 2：Dynamic Dependency Graph

目标：将已捕捉的执行事件组织成可查询的依赖图。

需要构造：

- 节点和边的统一抽象；
- 静态结构与动态实例的映射；
- Graph Manager；
- Artifact and State Manager；
- 跨 block 图与 artifact 持久化。

该阶段重点验证构图是否正确，不接入自动修复。

## Stage 3：Failure Attribution

目标：利用依赖图为失败生成局部因果上下文。

需要构造：

- failure anchoring；
- backward dependency tracing；
- 局部代码和 artifact 提取；
- 默认最小暴露与按需扩展机制。

该阶段可以先由人工检查归因结果，再接入 LLM 修复。

## Stage 4：Graph-Guided Minimal Patch

目标：让 LLM 基于局部依赖上下文修改程序，而不是重新生成完整 PTC。

需要构造：

- Repair Context Builder；
- Patch Controller；
- 程序版本管理；
- 修改区域与旧图节点之间的映射。

该阶段仍可采用完整重新执行，以单独验证局部修复是否有效。

## Stage 5：Invalidation and Selective Replay

目标：在局部修复成功的基础上加入结果复用。

需要构造：

- Invalidation Analyzer；
- 结果有效性判断；
- Selective Replay Controller；
- 新旧执行版本合并和 provenance 更新。

首先在只读、无副作用的 search 类工具上实现，再扩展到复杂工具。

## Stage 6：Stateful Tool Support

目标：支持 ToolSandbox 等存在外部状态和副作用的环境。

需要构造：

- STATE dependency；
- side-effect safety layer；
- 环境一致性检查；
- 安全回放与环境重置 fallback。

该阶段验证 GraphPTC 能否从 search 类任务泛化到 stateful tool environment。

# 十一、Benchmark

## BrowseComp-Plus（前期主评测）

前期 Agent 架构迭代使用 BrowseComp-Plus。该 benchmark 将网页资料冻结为约 10 万篇固定
文档和 830 个问题，使 Original PTC baseline 与 GraphPTC 可以共享完全相同的本地检索
后端，不依赖动态网页状态或额外搜索 API。

开发期使用本地 SQLite FTS5/BM25、固定 top-5 和确定性片段截断，重点比较：

- 答案准确率与 evidence retrieval recall；
- PTC block、模型请求和本地搜索调用数量；
- GraphPTC 的归因、修复和结果复用是否改善成功率与执行开销。

该开发配置用于受控消融，不作为 BrowseComp-Plus 官方 leaderboard 配置。完成 baseline 与
GraphPTC 全量对比后，正式对齐实验切换到官方 Pyserini/Lucene BM25、Qwen tokenizer 的
512-token snippet 和 Qwen3-32B judge。

## DeepSearchQA

用于验证 GraphPTC 在长链、多步骤信息查找任务中的能力，重点观察：

- 长程依赖是否能够被正确记录；
- 中间失败是否能够被局部归因；
- 修复后是否能够复用前序搜索结果；
- GraphPTC 是否能够在保持任务性能的同时减少重复工具调用和上下文开销。

该 benchmark 在后期按 Anthropic 公开复现配置运行，使用联网搜索和官方 evaluation
pipeline，避免在前期 Agent 架构迭代中引入动态搜索后端噪声。

## BrowseComp

用于验证 GraphPTC 能否从相对明确的多步骤检索流程泛化到更开放的 deep browsing 场景。

重点观察开放式搜索过程中动态分支、长依赖链和多次修复下的图管理能力。

该 benchmark 在后期按 Anthropic 公开复现配置运行，使用联网搜索和官方 evaluation
code。

# 十二、Baseline

## Original PTC

原始 PTC 是最核心 baseline，用于衡量 GraphPTC 在任务性能、故障恢复、工具调用开销和执行效率方面带来的变化。

为了支持不同基础模型，实验框架需要将 PTC 执行能力与特定厂商 API 解耦，并提供统一的模型和工具适配层。

## CodeAct

CodeAct 将可执行代码作为 Agent action，并允许模型根据执行 observation 继续修改或生成新的 action。

其作用是作为“代码驱动 Agent + 迭代修复”范式的对照，比较 GraphPTC 显式依赖图、局部归因和选择性重执行是否带来额外收益。

# 十三、Scope Boundary

当前完整框架只要求明确各部件的职责、输入输出关系和接入顺序，不要求一次性完成以下内容：

- 最终节点字段和图存储 schema；
- 特定数据库、文件系统或缓存后端；
- 固定的源代码插桩方式；
- 固定的 Graph API 名称和参数格式；
- 固定的 failure capsule 文本模板；
- 固定的 patch 表示形式；
- 完整的副作用类型系统；
- 最终的失效传播和重放调度算法。

这些内容应在每个 Stage 中以原始 PTC 为 baseline 逐步实现、验证和替换。GraphPTC 当前首先需要保证的是：

> **所有中间部件都可以独立接入、独立评测，并最终组合成“执行—追踪—归因—修复—失效分析—选择性重执行”的完整闭环。**
