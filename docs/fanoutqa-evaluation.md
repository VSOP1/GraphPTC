# FanOutQA evaluation

This adapter evaluates the frozen GraphPTC method and matched Fewshot PTC baseline on the official
FanOutQA open-book setting. The benchmark package is the updated `fanoutqa` 1.3.0 source release and
Wikipedia is served from `wikipedia_en_all_nopic_2023-09.zim` through local Kiwix.

The current `fanoutqa` Kiwix search helper omits `format=xml` and expects Atom entries, while
Kiwix Tools 3.8.2 returns RSS items. The shared adapter requests Kiwix XML explicitly, converts RSS
items to the official `Evidence` fields, and preserves the official Markdown conversion/cache
semantics for page content. Both calls use a 90-second local HTTP timeout. This compatibility layer
is identical in both arms and does not add another retrieval source.

The model can directly call only `programmatic_tool_call`. Inside its persistent Python namespace:

- `wiki_search(query=..., results=10)` returns official page records.
- `wiki_content(doc=...)` accepts one returned record and returns official Markdown content.

Both arms use the same model, synthetic PTC transport demonstration, official open-book task prompt,
Wikipedia snapshot, runtime budget, and concurrency. The only method difference is
`runtime.graph_adaptation_mode = "generic"` for GraphPTC and `"off"` for Fewshot PTC. Dev answers,
decompositions, dependency annotations, and necessary-evidence records are not exposed during
generation. The synthetic demonstration teaches only the PTC transport and contains no FanOutQA
question or Wikipedia fact.

For concurrent evaluation from Windows, serve the snapshot from the Ubuntu 22.04 ext4 filesystem,
not `/mnt/d`: eight synthetic concurrent search/content probes completed in at most 6.26 seconds on
ext4, while the NTFS-mounted snapshot caused systematic read timeouts. The evaluated server command
was:

```bash
kiwix-serve --daemon --port 8888 --threads 16 \
  /var/lib/fanoutqa/wikipedia_en_all_nopic_2023-09.zim
```

The paired dev configurations are:

- `configs/fanoutqa/graphptc-dev.toml`
- `configs/fanoutqa/fewshot-ptc-dev.toml`

The adapter commands are:

```powershell
.\.venv\Scripts\graphptc.exe inspect-fanoutqa --config configs\fanoutqa\graphptc-dev.toml
.\.venv\Scripts\graphptc.exe probe-fanoutqa-wikipedia --config configs\fanoutqa\graphptc-dev.toml
.\.venv\Scripts\graphptc.exe run-fanoutqa --config configs\fanoutqa\graphptc-dev.toml
.\.venv\Scripts\graphptc.exe evaluate-fanoutqa --config configs\fanoutqa\graphptc-dev.toml
.\.venv\Scripts\graphptc.exe compare-fanoutqa --config configs\fanoutqa\graphptc-dev.toml
```

Dev scoring uses the official FanOutQA implementation for loose/strict accuracy and ROUGE. The model
judge uses the official A-F factuality prompt and scoring rule with MiMo at temperature zero; it is
reported explicitly as `mimo_judge`, not as the official GPT judge. Missing or failed generations remain
in the 310-question denominator. The frozen dev result is reported without prompt or parameter tuning.

## Dev 310 result (2026-08-30)

| arm | loose | strict | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | MiMo judge | search/content calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GraphPTC | 72.13% | 29.03% | 38.13% | 22.01% | 35.13% | 58.39% | 5,368 / 3,484 |
| Fewshot PTC | 68.09% | 25.81% | 44.32% | 25.37% | 39.86% | 50.00% | 9,413 / 8,222 |
| GraphPTC delta | +4.04 pp | +3.23 pp | -6.20 pp | -3.36 pp | -4.72 pp | +8.39 pp | -4,045 / -4,738 |

All result, grade, and submission files contain 310 unique matched IDs. Each arm has one terminal
generation failure on the same content-filtered question; both are retained in the denominator. The
valid run has zero Kiwix read timeouts, and every GraphPTC task plus 309/310 baseline tasks fetched page
content. MiMo produced 310 valid labels per arm; paired MiMo outcomes are 60 GraphPTC-only correct, 34
baseline-only correct, 121 both correct, and 95 both incorrect.

An earlier complete generation was archived locally as
`runs/fanoutqa/dev-invalid-kiwix-timeout-20260830`. It is excluded because the NTFS-mounted Kiwix
backend caused 3,737 GraphPTC and 8,519 baseline search timeouts, reducing the intended open-book task
to mostly closed-book behavior. The rerun changed only the shared Kiwix protocol/timeout/storage layer;
method, prompt, model, budgets, and scoring remained fixed.
