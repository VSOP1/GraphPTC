# DeepPlanning evaluation

## Frozen official protocol

This adapter targets Qwen-Agent `benchmark/deepplanning` at commit
`31a4d36d123688581a9e9744427272b33ce940e0` and Qwen/DeepPlanning v1.1 at
revision `213876cce679f993a476d01042e13d111c0e3648`. Code and data are
Apache-2.0. Checksums, task counts, tool counts, and the resolved Python 3.10
environment are recorded in `data/deepplanning/`.

The official task inventory is Travel Chinese 120, Travel English 120, and
Shopping levels 1/2/3 with 50/50/20 tasks. Travel exposes 9 official tools;
each Shopping level exposes 15. The Windows worker sets `PYTHONUTF8=1` because
the official tool loader prints Unicode status symbols. The official minimum
requirements omit `soundfile` and `tqdm`, which are required by the Travel
imports and are pinned in the freeze.

The official documentation sets 400 maximum LLM calls per sample. Travel's
implementation has one 400-call loop. Shopping's current implementation has
two consecutive loops, each accepting the same `max_llm_calls`, so its
theoretical implementation limit is 800. The matched GraphPTC comparison uses
one `OriginalPTCAgent` loop with a total cap of 400 model requests for every
domain and both arms; this follows the documented cross-domain budget rather
than reproducing Shopping's code-level doubling.

Official DeepPlanning does not impose a per-task wall-clock deadline. An
initial matched run incorrectly inherited GraphPTC's generic 3600-second task
deadline and was aborted when it terminated otherwise active tasks. The frozen
DeepPlanning configs now set the task wall clock to infinity; the official 400
model-call cap remains the terminal execution budget.

Travel requires a final `<plan>...</plan>` response. Its official conversion
prompt maps that text to evaluator JSON, and the official evaluator reports
delivery rate, commonsense, personalized, composite, and case accuracy.
Shopping's official state is the isolated per-case database, especially
`cart.json`; completion is read from `messages.json`. Its evaluator reports
match rate, case score, incomplete rate, and validity. The official aggregate
averages Travel Chinese and English, then computes `avg_acc` as the mean of
Shopping weighted average case score and Travel case accuracy. DeepPlanning
v1.1 leaderboard values are averages over four runs.

The official Travel converter hard-codes `qwen-plus` and permits 30 JSON parse
retries. Per the experiment decision, this evaluation instead uses MiMo for
conversion while retaining the official conversion prompt, 30 conversion
retries, JSON extraction, output format, and evaluator. The external Qwen-Agent
checkout is not modified.

The official aggregate CLI currently accepts `--travel-output-dir` but does
not pass it to `load_travel_statistics`. The adapter invokes that same official
loader directly with the explicit result path and uses the official Shopping
statistics function; it does not patch the checkout.

## Matched arms

`configs/deepplanning/graphptc.toml` and
`configs/deepplanning/fewshot-ptc.toml` are identical except for
`runtime.graph_adaptation_mode` (`generic` versus `off`). Both use
`OriginalPTCAgent`, MiMo, temperature 0, the official benchmark-layer retry
semantics (30 total attempts with a fixed 1.5-second delay), the same token and
call budgets, the same generic PTC demonstration, and `max_stdout_chars=8000`. GraphPTC preserves
`GRAPH_ASSESSMENT` and `GRAPH_DELTA`.

Transport retries apply only within one logical model request. SDK-hidden
retries are disabled so every attempt remains observable in telemetry. A task
is never restarted, selected failures are never replayed, and sandbox state is
never rolled back.

One task owns one official-environment subprocess and persistent Python
namespace. PTC blocks directly call the official DeepPlanning tool instances.
Shopping databases are copied once per arm/run for isolation; no task state is
rolled back and selected failures are not retried. Travel databases are
read-only. Runtime graph projection contains task goals/constraints, declared
next actions, tool calls, result artifacts, cart state effects, and failures.

## Commands

Audit the installation and verify the matched configs:

```powershell
.\.venv\Scripts\graphptc.exe inspect-deepplanning --config configs/deepplanning/graphptc.toml
.\.venv\Scripts\graphptc.exe compare-deepplanning-configs
```

Leaderboard-equivalent execution requires all 360 tasks for both arms at run indices 0, 1, 2,
and 3. Do not pass `--task-key`, `--domain`, or `--limit` for a full run. For each index, run and
evaluate both arms, then create the paired report:

```powershell
foreach ($runIndex in 0..3) {
  .\.venv\Scripts\graphptc.exe run-deepplanning --config configs/deepplanning/graphptc.toml --run-index $runIndex
  .\.venv\Scripts\graphptc.exe run-deepplanning --config configs/deepplanning/fewshot-ptc.toml --run-index $runIndex
  .\.venv\Scripts\graphptc.exe evaluate-deepplanning --config configs/deepplanning/graphptc.toml --run-index $runIndex
  .\.venv\Scripts\graphptc.exe evaluate-deepplanning --config configs/deepplanning/fewshot-ptc.toml --run-index $runIndex
  .\.venv\Scripts\graphptc.exe compare-deepplanning --run-index $runIndex
}
```

A single matched run must be reported as development evidence and not leaderboard-equivalent.

For the selected matched single run, run index 0 was aborted before scoring
because the initial configuration disabled the official benchmark-layer
transport retries and provider errors dominated the partial results. Its
artifacts remain preserved under each arm's `run-0` directory with an
`ABORTED.json` marker. A raw, task-independent, zero-retry API staircase then
tested total concurrency 10, 20, and 40 with two waves per level. Levels 10 and
20 had zero failures; level 40 produced 60 rate-limit failures in 80 requests.
The old `full` namespace also contains preserved partial run indices from the
initial launch. `matched-single/run-0` is separately preserved and marked
aborted because it still had the non-official 3600-second wall clock.
`matched-single/run-1` used 10 workers per arm (20 total), but sustained
six-figure-token requests caused severe MiMo rate limiting; it was paused by
the user and remains preserved with its original config and `PAUSED.json`.
The replacement frozen run is `matched-single/run-2`, restarted from scratch
with 5 workers per arm (10 total). Only `matched-single/run-2` is eligible for
the matched single-run report; it is still not leaderboard-equivalent.

All task progress is written to `progress.jsonl`; block checkpoints,
trajectories, official reports/cart/messages, graphs, conversion outputs,
evaluator files, aggregation, configuration metadata, usage, duration, and
failure fields remain under the run directory.
