# GraphPTC ALFWorld adapter

This adapter connects the frozen GraphPTC method and its matched Fewshot PTC baseline to the
official ALFWorld text environment. It is an implementation and local protocol result only; no
official ALFWorld task has been run yet.

## Frozen comparison

Both arms use the same MiMo model, ALFWorld prompt and synthetic PTC demonstration,
`OriginalPTCAgent`, worker/runtime, budgets, official environment configuration, task order, and
metrics. The method variable is:

- GraphPTC: `runtime.graph_adaptation_mode = "generic"`.
- Fewshot PTC: `runtime.graph_adaptation_mode = "off"`.

The paired-config validator rejects any other behavioral difference. Graph inspection remains
disabled in both arms.

## Official alignment

The isolated worker pins `alfworld==0.4.2` and loads the semantic copy of that release's
`configs/eval_config.yaml` at `configs/alfworld/official-eval-0.4.2.yaml`. Inspection fails closed
unless the relevant official values are unchanged:

- `AlfredTWEnv` with no domain randomization;
- all six supported task types;
- `valid_seen` / `valid_unseen` through the official in- and out-of-distribution split names;
- seed 42, full evaluation sets, generation action space, and 50 environment steps;
- official evaluation concurrency 3 and ALFWorld version 0.4.2.

The official config's parallel batch size is reproduced as three independent task workers. Each
worker initializes the official environment with `batch_size=1`, because a model conversation and
persistent PTC namespace belong to exactly one episode. The inspection and run signature record
this adapter boundary explicitly. When both matched arms are launched simultaneously, each arm
retains three workers and the shared model service sees at most six concurrent requests; reports
must label this as dual-arm concurrency rather than an official per-arm batch-size change.

The worker exposes only `act(command)` and mutable `state` to model-generated Python. It does not
expose the official admissible-command list because the official DAgger evaluation configuration
uses the generation action space. Each `act` call delegates to the official `env.step([command])`;
the adapter neither rewrites actions nor supplies an expert plan. A PTC block may make several
sequential actions with Python control flow. State and Python variables persist across blocks and
are reset between tasks. The official `AlfredTWEnv` placement form is
`move OBJECT to RECEPTACLE`; `put OBJECT in/on RECEPTACLE` is the THOR form and is not rewritten.

Official `infos["won"]`, `infos["goal_condition_success_rate"]`, episode steps, and `done` are the
authoritative outcome signals. Reports keep runner/evaluator failures in the selected-task
denominator and report execution failures separately. Graph completion is based on official
success, not merely reaching an episode time limit.

## Environment setup and commands

Use a separate Linux environment; do not add ALFWorld's dependencies to GraphPTC's `.venv`:

```bash
python3.10 -m venv /home/agent/graphptc-alfworld/.venv
PATH=/home/agent/graphptc-alfworld/.venv/bin:$PATH \
  /home/agent/graphptc-alfworld/.venv/bin/python -m pip install \
  alfworld==0.4.2 PyYAML==6.0.3
/home/agent/graphptc-alfworld/.venv/bin/alfworld-download
```

`PyYAML` is an undeclared runtime dependency of the 0.4.2 wheel's YAML configuration path. The
venv `bin` directory must be on `PATH` while installing because `fast-downward-textworld` invokes
the unqualified `python` command during its build.

The checked-in configs expect the downloaded data at `/home/agent/.cache/alfworld`. Inspecting is
read-only and verifies the version, official defaults, task counts, config hash, game-file hashes,
and split hash:

```powershell
.\.venv\Scripts\graphptc.exe inspect-alfworld `
  --config configs\alfworld\graphptc-smoke.toml
```

The frozen smoke uses the first three `valid_seen` task IDs recorded in
`configs/alfworld/frozen-manifest-0.4.2.json`. Pass the same three IDs to both arms; do not use
`--restart` after either trajectory has been accepted into the comparison. The runner uses three
workers and prints only the final run summary:

```powershell
.\.venv\Scripts\graphptc.exe run-alfworld `
  --config configs\alfworld\graphptc-smoke.toml `
  --task-id <task-1> --task-id <task-2> --task-id <task-3>
.\.venv\Scripts\graphptc.exe run-alfworld `
  --config configs\alfworld\fewshot-ptc-smoke.toml `
  --task-id <task-1> --task-id <task-2> --task-id <task-3>
```

Smoke outputs are isolated from the full-run signatures. Full paired configs are provided for all
140 `valid_seen` tasks and all 134 `valid_unseen` tasks.
`evaluate-alfworld` validates the saved run signature and aggregates the already-recorded official
environment outcomes; it does not invoke an LLM grader.

## Current verification boundary

Ubuntu 22.04.5, ALFWorld 0.4.2, and the official data are installed in the isolated WSL venv. Live
inspection passes for all 140 `valid_seen` and 134 `valid_unseen` tasks; the environment, config,
asset, and split hashes are frozen in `configs/alfworld/frozen-manifest-0.4.2.json`. The
fake-worker protocol, persistent PTC state, action projection, matched configs, official-value
validation, task parsing, denominator behavior, and existing AppWorld compatibility pass local
tests.

The first live smoke under `runs/alfworld/smoke` is retained but excluded: its adapter prompt used
the THOR `put ... in/on ...` placement grammar, while the official text environment and its
`HandCodedTWAgent` require `move ... to ...`. The corrected matched smoke writes to
`runs/alfworld/smoke-move-grammar` and does not overwrite those diagnostic trajectories. In the
matched smoke, each arm processed all three frozen tasks with two official
successes and zero runner, evaluator, or PTC execution failures. This is a development smoke, not
an official full-split comparison.

## Local full-split result

The 2026-08-29 development run used three workers per arm (six concurrent model jobs total) and
matched task sets within each split. All 140 `valid_seen` and 134 `valid_unseen` tasks finished in
both arms, with zero runner or evaluator failures.

- `valid_seen`: GraphPTC 106/140 (75.71%, 19.61 mean steps); Fewshot PTC 108/140 (77.14%, 15.08).
- `valid_unseen`: GraphPTC 95/134 (70.90%, 21.39 mean steps); Fewshot PTC 94/134 (70.15%, 17.37).
- Combined: GraphPTC 201/274 (73.36%, 20.48 mean steps); Fewshot PTC 202/274 (73.72%, 16.20).

These are local runs against the official ALFWorld environment outcomes, not official leaderboard
results. GraphPTC recorded 23 PTC execution-failure tasks across both splits versus 2 for the
baseline; those are distinct from runner or evaluator failures. Offline trace analysis found that
21 of the 23 GraphPTC records were calls made after the episode had already terminated and only 2
were `NoStdout` protocol errors. The corresponding baseline split was one post-termination call and
one `NoStdout` error.

The current GraphPTC arm is a model-visible generic execution-graph intervention, not a complete
semantic task-dependency graph: all 274 records report `task_graph_initialized=false`, with no
artifact loads, reuse, or graph inspection. In addition, all 579 blocks whose environment actions
only returned `Nothing happens.` were still marked as realized graph deltas. Interpret the current
`PATCH`/`REPLAN` and graph-delta telemetry as declared execution feedback, not demonstrated repair
or semantic graph progress.
