# FRAMES adapter

The adapter evaluates the official 824-example `test.tsv` from
`google/frames-benchmark` revision `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`.
It uses the paper's `wikipedia/20230601.en` TFDS snapshot, whole-corpus BM25,
top 10 documents per query, and a 25-query budget corresponding to the reported
5 rounds by 5 queries configuration. Dataset answers, reasoning labels, and gold
Wikipedia links are used only after generation.

`wiki_search(query)` and `wiki_content(doc)` are registered inside the persistent
PTC Python environment. Both arms use the same MiMo model, synthetic planning
demonstration, retrieval service, budgets, and concurrency; the only method
difference is `graph_adaptation_mode = "generic"` versus `"off"`.

The Direct Tool Calling config `configs/frames/direct-tools-test.toml` exposes the same two
retrieval functions as native model tools, disables the PTC demonstration and graph control, and
keeps the same dataset, retriever, query budget, grader, and output limits.

Prepare and serve the official corpus in Ubuntu 22.04:

```bash
bash scripts/frames/setup_retriever.sh
bash scripts/frames/prepare_wikipedia.sh
bash scripts/frames/run_retriever.sh
```

After `probe-frames-wikipedia` succeeds, run the generated three-arm profile through the unified
launcher:

```bash
.venv/bin/python scripts/evaluation/full_suite.py preflight --profile MODEL_PROFILE
.venv/bin/python scripts/evaluation/full_suite.py all --profile MODEL_PROFILE
```

The primary metric is MiMo accuracy using the FRAMES paper's Figure 6 autorater
prompt. It is labeled as MiMo judging, not as a reproduction of the paper's
Gemini-Pro-1.5-0514 autorater. Normalized exact match and atomic reasoning-type
breakdowns are reported as secondary diagnostics.
