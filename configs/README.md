# 配置目录

`configs/` 只保留可直接启动完整评测范围的配置。六个 GraphPTC 领先 benchmark 同时保留
GraphPTC、Fewshot PTC 与 Direct Tool Calling 三组配置；三组的模型、数据、预算与评分协议应对齐。
smoke、pilot、fold、临时 challenger 和示例配置不放在本目录中。

## 正式配置

| Benchmark | GraphPTC | Fewshot PTC | Direct Tool Calling | 完整范围 |
| --- | --- | --- | --- | --- |
| Agent-Diff | `agent_diff/graphptc.toml` | `agent_diff/fewshot-ptc.toml` | `agent_diff/direct-tools.toml` | 224 个任务 × 3 trials |
| ALFWorld | `alfworld/graphptc-valid-{seen,unseen}.toml` | `alfworld/fewshot-ptc-valid-{seen,unseen}.toml` | — | `valid_seen` 140 题及 `valid_unseen` 134 题 |
| APIFlow | `apiflow/graphptc.toml` | `apiflow/fewshot-ptc.toml` | — | v1.0 全部 467 题 |
| AppWorld | `appworld/appworld.graphptc-test-{normal,challenge}.toml` | `appworld/appworld.fewshot-ptc-test-{normal,challenge}.toml` | `appworld/appworld.direct-tools-test-{normal,challenge}.toml` | `test_normal` 168 题及 `test_challenge` 417 题 |
| BrowseComp-Plus | `browsecomp_plus/browsecomp_plus.graphptc-full.toml` | `browsecomp_plus/browsecomp_plus.fewshot-ptc-full.toml` | `browsecomp_plus/browsecomp_plus.direct-tools-full.toml` | 单次完整 830 题，不按 fold 拆分 |
| DeepPlanning | `deepplanning/graphptc.toml` | `deepplanning/fewshot-ptc.toml` | — | 360 题 × 4 runs 全量模板；当前尚无正式结果 |
| FanOutQA | `fanoutqa/graphptc-dev.toml` | `fanoutqa/fewshot-ptc-dev.toml` | `fanoutqa/direct-tools-dev.toml` | 官方 `dev` 全部 310 题 |
| FRAMES | `frames/graphptc-test.toml` | `frames/fewshot-ptc-test.toml` | `frames/direct-tools-test.toml` | 官方 `test` 全部 824 题 |
| InterCode | `intercode/graphptc.toml` | `intercode/baseline.toml` | — | Bash 200 题及 SQL 1,034 题 |
| ToolHop | `toolhop/graphptc.toml` | `toolhop/fewshot-ptc.toml` | — | Mandatory 全部 995 题 |
| ToolSandbox | `toolsandbox/graphptc.toml` | `toolsandbox/fewshot-ptc.toml` | `toolsandbox/direct-tools.toml` | 官方全部 1,032 个场景 |

## 使用约束

- 切换模型时，六个领先 benchmark 要复制三组 matched 配置，同时修改模型字段和三组输出路径。
- 已产生结果的配置不得原地修改；新评测使用新的 profile 与输出目录。
- frozen manifest、任务选择及其 hash 不得为了适配新结果而回调。
- 数据 SHA-256 只保存在对应冻结 manifest 中，由 `inspect-*` 校验，不在成对 TOML 中重复保存。
- AppWorld、ToolSandbox、Agent-Diff 使用 `{repo}/external/...`，由配置 loader 按仓库位置解析；不要再
  写入 `D:`、`/mnt/d`、`wsl.exe` 或个人 home 路径。
- ToolSandbox 的 `[user_model]` 是冻结 user simulator。切换被评测模型只改 `[model]`，不得连带
  修改 `[user_model]`。

新模型不要手工逐份复制，使用：

```bash
.venv/bin/python scripts/evaluation/full_suite.py create-profile \
  --profile PROFILE --model MODEL --base-url BASE_URL
```

BrowseComp-Plus 的上游加密 CSV 位于
`data/browsecomp_plus/browse_comp_test_set.csv`。三份 full 配置都直接读取
`data/browsecomp_plus/questions.jsonl` 的 830 题。
