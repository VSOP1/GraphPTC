# ToolSandbox evaluation

## Scope and official environment

This integration evaluates only GraphPTC and the matched Fewshot PTC baseline. It uses the
official [apple/ToolSandbox](https://github.com/apple/ToolSandbox) repository at commit
`165848b9a78cead7ca7fe7c89c688b58e6501219`, package version `0.0.1`, and all 1,032 official
scenarios with the default tool backend. The official repository is installed in the isolated WSL
environment `/home/agent/graphptc-toolsandbox/.venv`; the GraphPTC `.venv` is not modified.

The official package declares Python 3.9 or newer. This run used Ubuntu 22.04 under WSL with Python
3.10.12. ToolSandbox's dependencies are old enough that its pinned OpenAI 1.17 client is
incompatible with httpx 0.28, so the isolated environment fixes `httpx==0.27.2`. The only extra
packages needed by the adapter and sanity checks are `codecell==0.2.1`,
`toolregistry[ptc]==0.14.0`, `tomli==2.2.1`, and `pytest==8.3.5`; `pip check` reports no broken
requirements. The official Apple sample-code license in the repository governs ToolSandbox. The
integration is verified on WSL, not native Windows Python.

`MIMO_API_KEY` supplies both the agent and the on-policy user simulator. `RAPID_API_KEY` is passed
into WSL for official external search functions. Neither key is written to results or prompts.

## Interface boundary

The official `Scenario`, starting `ExecutionContext`, state databases, tool definitions, user
simulator prompt/few-shot, `end_conversation` convention, persistent `InteractiveConsole`,
milestone/minefield evaluator, and trajectory serialization remain authoritative.

The model sees exactly one directly callable tool, `programmatic_tool_call`. Its `code` is sent as
one source block directly to the scenario's persistent ToolSandbox shell. The official scenario
functions are installed as Python globals, including official agent-facing aliases for scrambled
tool-name cases. There is no outer Python runtime calling an inner code string. Variables persist
between blocks; every scenario receives a fresh official context and role set.

ToolSandbox's stock execution role assumes one API trace per model tool call. A PTC block can call
several official APIs, so `PTCExecutionEnvironment` keeps the same official compiler, console,
state snapshots, stdout/stderr behavior, and tool tracing, but attaches all traces produced by the
single coherent program to its single response. This is the only execution seam.

The official agent system instruction (do not assume argument values; ask for clarification when
ambiguous) is retained verbatim and followed by the shared PTC prompt. Both arms receive the same
authoritative per-scenario function schemas and one task-independent PTC organization example.
The example demonstrates Python-side filtering/aggregation only and contains no benchmark answer
or app/API-specific tactic. The shared semantics require direct persistent-shell execution,
coherent program phases, compact stdout, state preservation, and forward repair.

The two full-run configs differ only in graph control and output paths:

- GraphPTC: `graph_adaptation_mode="generic"`; the outer call also declares `action`, `target`, and
  `expected_change`, and every execution returns a model-visible `GRAPH_DELTA`.
- Fewshot PTC: `graph_adaptation_mode="off"`; graph fields and graph output are removed from the
  same demonstration and tool schema.

Graph inspection is disabled in both arms. There is no placebo, third arm, gate, fallback branch,
task-specific rule, or retry of final-run failures.

## Necessary sanity checks

- Official resolution returned 1,032 scenarios and the documented test scenario names.
- Direct execution in the official shell preserved variables and attached two API traces from one
  PTC program containing two calls.
- Agent tool exposure follows the official `visible_to` contract; the user-only
  `end_conversation` tool is not exposed to the agent.
- Stdout sent back to the model is capped at 8,000 characters; the official execution context and
  state snapshots remain intact.
- Tool/runtime exceptions become failed graph blocks and remain in the official trajectory.
- Official scenario termination is controlled by `conversation_active`; contexts and role clients
  are isolated per scenario and torn down afterward.
- The selected official execution/evaluation/context tests produced 21 passes. Two upstream
  assertions did not pass under the current Python/runtime combination: exact SyntaxError caret
  formatting and a parallel-call permutation order. GraphPTC submits one coherent program message
  and does not use ToolSandbox's parallel-call permutation path.
- A one-scenario GraphPTC smoke completed official serialization and evaluation, including a
  model-visible graph failure/delta and a later repaired action.

An initial engineering batch exposed the stock execution role's one-trace-per-message assertion.
That batch was discarded before the final runs. The deterministic multi-call seam was fixed and
checked, then GraphPTC was restarted from an empty result file. The final trajectories below were
not individually retried.

## Full matched evaluation

Both final runs used MiMo `mimo-v2.5`, temperature 0, a 4,096-token per-response cap, 30 official
messages, an 8,000-character model-visible stdout cap, the default backend, and 16 scenario
workers. Runner failures are scored as zero and remain in the 1,032-scenario denominator, matching
the official CLI failure convention. `minefield_similarity` is an undesirable violation score;
the official combined similarity is milestone similarity when it is zero and zero otherwise.

| Arm | Scenarios | Official similarity | Mean milestone | Mean minefield | Runner failures | Execution-failure scenarios / blocks | Max-message incomplete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphPTC | 1,032 | **74.16** | 83.46 | 9.30 | 2 | 390 / 685 | 84 |
| Fewshot PTC | 1,032 | 43.67 | 48.04 | 4.36 | 39 | 357 / 808 | 12 |

GraphPTC improves official similarity by **30.48 percentage points**. In the paired per-scenario
comparison (runner failures equal zero), GraphPTC wins 705, ties 226, and loses 101 scenarios.
There were no evaluator exceptions after a completed rollout and no model/network transport
failures. The two GraphPTC runner failures and seven of the Fewshot PTC runner failures came from
the MiMo user simulator selecting a non-user tool. The other 32 Fewshot PTC runner failures came
from an agent response that violated the single-outer-PTC-call contract. They are retained as zero
scores without retry.

| Official category | N | GraphPTC | Fewshot PTC | Delta (pp) |
| --- | ---: | ---: | ---: | ---: |
| All categories | 1,032 | **74.16** | 43.67 | **+30.48** |
| Multiple tool call | 656 | **78.97** | 30.93 | **+48.04** |
| State dependency | 192 | **83.35** | 8.40 | **+74.95** |
| Multiple user turn | 224 | **72.84** | 31.10 | **+41.73** |
| Canonicalization | 472 | **74.04** | 34.01 | **+40.03** |
| Insufficient information | 224 | 54.42 | **72.90** | **-18.48** |
| Tool name scrambled | 129 | **80.00** | 48.11 | **+31.89** |
| All tools available | 129 | **68.94** | 38.44 | **+30.50** |

The negative insufficient-information result is important: GraphPTC's strong stateful execution
and repair gains do not translate to conservative refusal/clarification behavior. The aggregate
claim should therefore be framed around dependency-rich execution, not universal dominance.

## What the graph recorded and used

Across 1,030 evaluated GraphPTC trajectories (the two runner failures have no completed graph), the
graph contains 1,030 task nodes, 5,068 action intents, 4,043 PTC blocks, 4,034 official API actions,
15,024 artifacts, 4,881 Python state versions, 1,130 official-state effects, and 685 failure nodes.
Effects are inferred generically from official database deltas, never from API-name hardcoding.

All 4,043 completed PTC blocks produced model-visible graph feedback. The model declared 3,723
`CONTINUE`, 168 `PATCH`, and 37 `REPLAN` block actions. Of 972 graph deltas that did not realize
their declared change, the next declared block was `PATCH` 120 times and `REPLAN` 33 times. This
shows that graph feedback was in the decision context and was followed by explicit repair/replan
behavior. It does not by itself isolate graph feedback from the raw execution error: the matched
two-arm outcome establishes the method-level difference, while a separate feedback ablation would
be needed for a narrower causal claim.

## Reproduction and artifacts

```powershell
.\.venv\Scripts\graphptc.exe inspect-toolsandbox `
  --config configs\toolsandbox\graphptc-smoke.toml

.\.venv\Scripts\graphptc.exe run-toolsandbox `
  --config configs\toolsandbox\graphptc.toml --restart
.\.venv\Scripts\graphptc.exe run-toolsandbox `
  --config configs\toolsandbox\fewshot-ptc.toml --restart

.\.venv\Scripts\graphptc.exe evaluate-toolsandbox `
  --config configs\toolsandbox\graphptc.toml
.\.venv\Scripts\graphptc.exe evaluate-toolsandbox `
  --config configs\toolsandbox\fewshot-ptc.toml
```

Per-scenario terminal records and aggregate reports are under `runs/toolsandbox/<arm>/`. Official
`pretty_print.txt`, `execution_context.json`, and `conversation.json` files are under each arm's
`artifacts/trajectories/`; graph JSON exists only for completed GraphPTC worker runs. The `runs/`
tree is ignored by git because it contains simulated conversations, tool arguments/results, state
snapshots, and external API data. Tracked configs and this aggregate report contain no task text,
credentials, or raw trajectory content.
