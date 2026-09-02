# 运行结果目录

本目录只保存当前结果汇总采用的最终全量响应、评分、报告、checkpoint、日志、图和其他运行
artifact。除本说明外，内容默认被 Git 忽略。

历史运行是只读评测证据。源码重构后 implementation hash 会变化，因此新模型或新代码必须使用
新的输出目录，不能续接旧运行。不要删除成功记录、选择性重试失败项或在看到结果后调整阈值。

`runs/profiles/<profile>/` 保存统一入口为新模型生成的 manifest 和 21 份 resolved 配置；各 benchmark
结果写入 `runs/<benchmark>/<profile>/<arm>/`。这些文件属于新评测证据，不覆盖下表的历史结果。

## 保留的正式结果

| Benchmark | 保留路径 | 全量范围 |
| --- | --- | --- |
| Agent-Diff | `agent_diff/{graphptc,fewshot-ptc}` | 224 tasks × 3 trials |
| ALFWorld | `alfworld/valid-{seen,unseen}/{graphptc,fewshot-ptc}` | 140 seen + 134 unseen |
| APIFlow | `apiflow/{graphptc,fewshot-ptc}`、`apiflow/temperature1-epoch1/` | temperature 0 与 1，各 467 |
| AppWorld | `appworld/{graphptc,fewshot-ptc}-test-{normal,challenge}` | 168 normal + 417 challenge |
| BrowseComp-Plus（历史结果） | `browsecomp_plus/{graphptc-stdout8k-,fewshot-ptc-v1-stdout8k-}{fold1,fold2,fold3,remaining530}` | 四个历史分片合计 830；新评测改为单次 830 |
| FanOutQA | `fanoutqa/dev/{graphptc,fewshot-ptc}` | dev 310 |
| FRAMES | `frames/test/{graphptc,fewshot-ptc}` | test 824 |
| InterCode | `intercode/{graphptc,baseline}` | Bash + SQL，共 1,234 |
| ToolHop | `toolhop/mandatory-temperature0-epoch1/` | Mandatory 995 |
| ToolSandbox | `toolsandbox/{graphptc,fewshot-ptc}` | 1,032 scenarios |

每个保留目录应包含足以审计该轮结果的完整证据链，而不只是汇总 `report.json`。smoke、pilot、
失败启动、诊断重跑、旧实现和被替代的结果目录不属于交接包。
