# Benchmark 结果汇总

证据来自本工作区 `runs/` 下的 report 与 paired-report。所有 matched
对比均使用 `mimo-v2.5`；“领先”表示 GraphPTC 相对仓库内 Fewshot PTC 对照，不是外部排行榜声明。
BrowseComp-Plus 的现有历史结果由四个分片汇总；今后的正式重测改用单个 830 题配置，结果口径不变。

## 完整评测且主要指标领先

| Benchmark | 范围 | 主要指标 | GraphPTC | Fewshot PTC | 差值 |
| --- | --- | --- | ---: | ---: | ---: |
| BrowseComp-Plus | 830/830 | accuracy | 35.66% | 29.40% | +6.27 pp |
| AppWorld normal | 168/168 | TGC / SGC | 78.0 / 67.9 | 67.3 / 41.1 | +10.7 / +26.8 pp |
| AppWorld challenge | 417/417 | TGC / SGC | 69.8 / 54.0 | 52.5 / 30.2 | +17.3 / +23.8 pp |
| ToolSandbox | 1,032/1,032 | official similarity | 74.16 | 43.67 | +30.48 pp |
| Agent-Diff | 224 × 3 | task pass rate | 67.86% | 67.11% | +0.74 pp |
| FanOutQA | dev 310 | official-local loose | 72.13% | 68.09% | +4.04 pp |
| FRAMES | test 824 | paper-style judge | 69.54% | 66.38% | +3.16 pp |

## 完整 paired 评测但没有总体领先

| Benchmark | 范围 | GraphPTC | Fewshot PTC | 结论 |
| --- | --- | ---: | ---: | --- |
| ALFWorld | seen + unseen，274 | 73.36% | 73.72% | -0.36 pp |
| APIFlow temperature 0 | 467 | 86.51% | 86.51% | 完全打平 |
| APIFlow temperature 1 | 467 | 87.15% | 88.65% | -1.50 pp |
| ToolHop Mandatory | 995 | 60.30% | 62.11% | -1.81 pp |
| InterCode Bash + SQL | 1,234 | 74.72% | 76.82% | -2.11 pp |

## 主要证据路径

- `runs/browsecomp_plus/{graphptc-stdout8k-,fewshot-ptc-v1-stdout8k-}{fold1,fold2,fold3,remaining530}/report.json`
- `runs/appworld/{graphptc,fewshot-ptc}-test-{normal,challenge}/report.json`
- `runs/toolsandbox/{graphptc,fewshot-ptc}/report.json`
- `runs/agent_diff/{graphptc,fewshot-ptc}/report.json`
- `runs/fanoutqa/dev/{graphptc,fewshot-ptc}/report.json`
- `runs/frames/test/{graphptc,fewshot-ptc}/report.json`
- `runs/alfworld/valid-{seen,unseen}/{graphptc,fewshot-ptc}/report.json`
- `runs/apiflow/{graphptc,fewshot-ptc}/report.json`
- `runs/apiflow/temperature1-epoch1/{graphptc,fewshot-ptc}/report.json`
- `runs/toolhop/mandatory-temperature0-epoch1/paired-report.json`
- `runs/intercode/paired-report.json`
