# GraphPTC AppWorld evaluation

This document is the source of truth for the first GraphPTC AppWorld dev evaluation. Raw
AppWorld task outputs remain untracked because they contain protected data; reproducibility is
carried by the frozen manifest, environment snapshots, run signatures, and artifact hashes.

## Compatibility decision

GraphPTC and AppWorld run in separate Python environments. The verified GraphPTC environment is
Windows Python 3.11.1 with Pydantic 2.13.4. The verified AppWorld environment is WSL2 Ubuntu 22.04,
Python 3.11.15, AppWorld 0.1.3.post1, Pydantic 1.10.26, and data 0.1.0. Both environments pass
`pip check`. This avoids contaminating GraphPTC's existing `.venv` with AppWorld's incompatible
dependency set.

The current PyPI release used here is 0.1.3.post1. The AppWorld repository main branch reports
0.2.0.dev0, so this evaluation pins the stable release rather than silently following main.

The official setup and verification sequence is:

```bash
python3.11 -m venv /home/agent/graphptc-appworld/.venv
/home/agent/graphptc-appworld/.venv/bin/python -m pip install appworld==0.1.3.post1
/home/agent/graphptc-appworld/.venv/bin/python -m appworld.cli install
/home/agent/graphptc-appworld/.venv/bin/python -m appworld.cli download data \
  --root /home/agent/graphptc-appworld
/home/agent/graphptc-appworld/.venv/bin/python -m appworld.cli verify tests \
  --root /home/agent/graphptc-appworld
/home/agent/graphptc-appworld/.venv/bin/python -m appworld.cli verify tasks \
  --root /home/agent/graphptc-appworld
```

The latest verification completed with 1,553 app tests plus 138 other tests, and all 147
train/dev tasks verified. Native Windows verification failed in this environment at AppWorld's
POSIX `SIGALRM` use; WSL2 is therefore the verified GraphPTC execution path. This is an
environment result, not a claim that every AppWorld revision is categorically unsupported on
Windows.

Official references:

- <https://github.com/StonyBrookNLP/appworld/blob/main/README.md>
- <https://github.com/StonyBrookNLP/appworld/blob/main/experiments/prompts/react_code_agent/instructions.txt>
- <https://github.com/StonyBrookNLP/appworld/blob/main/experiments/prompts/full_code_agent/full_code_instructions.txt>
- <https://pypi.org/project/appworld/>

## Runtime and evaluator boundary

The execution path is deliberately one layer deep:

```text
model programmatic_tool_call.code
  -> OriginalPTCAgent shared PTC loop
  -> AppWorldProgramRuntime JSONL request
  -> one task-scoped worker
  -> world.execute(the exact code)
  -> persistent AppWorld Python shell
```

The worker owns exactly one `AppWorld` instance. The model's code sees AppWorld's `apis` and
`requester` globals and executes directly in its persistent shell; GraphPTC does not wrap that code
inside a second Python program or expose `world`, `save_state`, or `load_state`. A fresh worker is
created per task, official individual evaluation runs against that task's saved output, and the
world is then closed in `finally`. Limited concurrency uses independent workers and does not share
an AppWorld API server.

`apis.supervisor.complete_task()` remains AppWorld's authoritative completion operation, and
`world.task_completed()` controls GraphPTC stopping. A fatal timeout or worker loss ends the task
instead of silently starting a clean world. Close acknowledgement is recorded as
`termination_confirmed`.

GraphPTC truncates only the model-visible stdout observation at 8,000 characters. Official task
logs, API-call logs, database outputs/deltas, AppWorld versions, and evaluator artifacts are kept in
AppWorld's output tree. Derived graph/report traces recursively redact credential-like values;
official logs are not rewritten and must remain protected.

`OriginalPTCAgent` is intentional. `CodeActPTCAgent` subclasses the same core PTC loop but owns the
normal persistent-code runtime lifecycle. AppWorld needs an injected runtime and the order
`agent.run -> official evaluate -> close`; switching agent classes would add lifecycle coupling
without changing the model loop or graph hooks.

## Prompt policy

The adapter has three named prompt variants so prompt effects are explicit:

| Variant | Purpose |
| --- | --- |
| `appworld-general` | Minimal control prompt. |
| `appworld-ptc-semantics` | Official general operating semantics plus GraphPTC PTC and graph-control semantics. |
| `appworld-ptc-fewshot` | The semantics variant plus one synthetic AppWorld PTC demonstration using only fake values and no benchmark-task answer. |

The aligned semantics cover ApiDocs-before-use, authentication, pagination, Supervisor
credentials, contacts, time/timezone, filesystem boundaries, concise task completion, and
`complete_task`. They do not encode task answers, app-specific error branches, or API names as
graph schema.

The few-shot is not a FullCode imitation. It deliberately divides one simulated task into two
coherent direct-execution PTC blocks: inspect the relevant contracts and then
authenticate/paginate/complete. Each block retains the same `programmatic_tool_call` contract and
generic AppWorld graph-intent fields: `action`, `target`, and `expected_change`. Input/output
artifacts are derived from execution rather than declared in this adapter's tool schema. The agent
still chooses APIs, parameters, program structure, and repairs.

## Graph boundary and INSPECT

Every executed block is projected into the domain-neutral `EpisodeGraph`: task/action intent,
block, API action, input/output artifact, candidate state effect, failure, and dependency edges.
API effects inferred from method names are explicitly marked as candidate/inferred; the official
database delta remains the authoritative state record. `GoalGraphAdaptation` returns bounded
`GRAPH_DELTA` feedback after each block, including observed progress, failure, and generic next
action opportunities.

The original concern about `INSPECT` was correct: an enum value alone is metadata, not a graph
query. The current behavior is therefore:

- `graph_inspection_enabled = false` by default for both benchmarks; `INSPECT` is absent from the
  tool schema.
- When explicitly enabled, `INSPECT` requires a structured `inspection` request for `frontier` or
  `trace`. The host executes the bounded, read-only query after projecting the current block and
  returns the result in the next `GRAPH_DELTA`.
- Telemetry separates declaration, well-formed request, attempted query, success, failure, emitted
  response, and returned result.

A deterministic synthetic test proves that a returned artifact ID can change the following code
relative to a metadata-only placebo. In the real AppWorld smoke, pilot, and full evaluation, the
model made zero `INSPECT` declarations or queries. The graph was populated and graph deltas were
model-visible, but these runs do not establish that inspection causally changed an AppWorld action.

## Reproducible commands and artifacts

Inspect the isolated installation and dataset without running a task:

```powershell
.\.venv\Scripts\graphptc.exe inspect-appworld `
  --config configs\appworld.graphptc-dev-runtime-smoke.toml
```

The current-source smoke was run as:

```powershell
.\.venv\Scripts\graphptc.exe run-appworld `
  --config configs\appworld.graphptc-dev-runtime-smoke.toml `
  --task-id fac291d_3
```

The frozen pilot is defined by
`data/appworld/dev-frozen-pilot.manifest.json` and
`configs/appworld.graphptc-dev-frozen-pilot.toml`. Its three selected trajectories must not be
restarted, retried, replaced, or used for task-specific tuning. The manifest records the official
dev task-list hash, task IDs, code/data versions, dirty-tree source hash, config and prompt hashes,
environment freeze hashes, run signature, result/graph hashes, and official output-tree hashes.

The runs were produced from a dirty working tree at baseline commit
`c8da46adf9e10f2ffd9c02fa6d42c4a3f864804a`; consequently the recorded source hash, not that
commit alone, identifies the evaluated implementation. Create a clean commit before any larger
run. This paragraph describes the earlier dev pilot; the later frozen full evaluation is reported
below.

## Dev evidence

| Run | Tasks | Completed | Official successes | TGC | SGC | Execution-failure tasks / blocks | Incomplete | Evaluator failures | Runner failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current-source smoke | 1 | 1 | 1 | 100.0 | 100.0 | 1 / 2 | 0 | 0 | 0 |
| Frozen dev pilot | 3 | 3 | 2 | 66.7 | 0.0 | 3 / 3 | 0 | 0 | 0 |

The smoke's 100.0 SGC is only a one-task/one-variation plumbing diagnostic; it is not a comparable
scenario score. In the three-variation pilot, all tasks called `complete_task`, all workers closed
cleanly, and the single official failure was a semantic collection/data-model error rather than an
execution, completion, evaluator, or isolation failure. It is retained without retry.

The pilot recorded 22 PTC blocks, 130 API calls, three recovered execution-failure blocks, no
stdout truncation, and no inspection query. Its graphs contain 3 task nodes, 25 action intents
(including answers), 22 blocks, 130 API actions, 293 artifacts, 3 failures, and 11 candidate state
effects. All three graph tasks are complete because all three AppWorld worlds observed
`complete_task`; graph completion does not imply official correctness.

## Frozen full evaluation

The matched full evaluation was frozen at commit `1676f7e`. Both arms used the same MiMo model,
AppWorld prompt semantics and few-shot program code, `OriginalPTCAgent`, AppWorld runtime, budgets,
8,000-character stdout limit, official evaluator, and four task workers. The only method difference
was `graph_adaptation_mode`: `generic` for GraphPTC and `off` for Fewshot PTC. Graph inspection was
disabled in both arms. No test task, individual evaluation, or failure report was inspected, and no
trajectory was retried or used to change the method.

| Split | Arm | Official successes | TGC | SGC | Completed | Execution-failure tasks / blocks | Incomplete | Evaluator / runner failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `test_normal` | GraphPTC | 131 / 168 | 78.0 | 67.9 | 162 | 118 / 251 | 6 | 0 / 0 |
| `test_normal` | Fewshot PTC | 113 / 168 | 67.3 | 41.1 | 143 | 88 / 249 | 25 | 0 / 0 |
| `test_challenge` | GraphPTC | 291 / 417 | 69.8 | 54.0 | 395 | 346 / 844 | 22 | 0 / 0 |
| `test_challenge` | Fewshot PTC | 219 / 417 | 52.5 | 30.2 | 336 | 288 / 697 | 81 | 0 / 0 |

GraphPTC improved over the matched baseline by 10.7 TGC points and 26.8 SGC points on
`test_normal`, and by 17.3 TGC points and 23.8 SGC points on `test_challenge`. All 1,170 task
records were unique and terminal, every worker termination was confirmed, and the code/data
versions remained AppWorld 0.1.3.post1 and data 0.1.0.

## Data handling and next gate

The downloaded AppWorld data license is Apache-2.0 with an additional requirement that public
redistribution of protected data or derivatives be encrypted. Do not commit or publicly expose
raw `runs/appworld`, AppWorld output trees, evaluator traces, task text, ground truth, or database
content. Only non-sensitive manifests, dependency freezes, aggregate metrics, and hashes are
tracked here.

The main method and matched baseline are now frozen. Do not use the test results for task-level
error analysis, prompt changes, or another tuned rerun. Any future method change is a new study and
must not replace these trajectories.
