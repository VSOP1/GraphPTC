# GraphPTC AppWorld evaluation

GraphPTC runs AppWorld in a separate WSL Python environment. This is required because AppWorld
0.1.3.post1 requires `pydantic<2`, while GraphPTC's `toolregistry==0.14.0` requires
`pydantic>=2.7.2`. Do not install AppWorld into GraphPTC's `.venv`.

## Verified environment

- WSL2 Ubuntu 22.04
- Python 3.11
- AppWorld code 0.1.3.post1
- AppWorld data 0.1.0
- AppWorld root `/home/agent/graphptc-appworld`

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

The native Windows package installs, but the official package/task verifier fails because AppWorld
uses POSIX `signal.SIGALRM`. WSL is therefore the supported GraphPTC execution path. The official
AppWorld README is at <https://github.com/StonyBrookNLP/appworld>.

## Interface boundary

`OriginalPTCAgent` accepts a `codecell.BaseRuntime`. `AppWorldProgramRuntime` sends the model's exact
`programmatic_tool_call.code` over JSONL IPC to one task-scoped worker. The worker owns one
`AppWorld` object and calls `world.execute(code)` directly, so AppWorld's persistent Python shell,
API logs, database outputs, safety checks, completion signal, and official evaluator remain
authoritative. GraphPTC never installs or imports AppWorld in its own process.

Each task gets a separate worker process and is closed in `finally`. The adapter does not expose
`world`, `save_state`, or `load_state` inside the agent shell. Parallel runs use separate local
AppWorld processes and no shared API server.

## Commands

```powershell
.\.venv\Scripts\graphptc.exe inspect-appworld `
  --config configs\appworld.graphptc-dev-smoke.toml

.\.venv\Scripts\graphptc.exe run-appworld `
  --config configs\appworld.graphptc-dev-smoke.toml `
  --task-id 50e1ac9_1

.\.venv\Scripts\graphptc.exe evaluate-appworld `
  --config configs\appworld.graphptc-dev-smoke.toml
```

The frozen three-task dev pilot is defined by `data/appworld/dev-pilot.manifest.json` and
`configs/appworld.graphptc-dev-pilot.toml`. Do not retry or replace its selected task trajectories.
No `test_normal` or `test_challenge` runner has been executed.
