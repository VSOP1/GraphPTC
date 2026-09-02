# BrowseComp-Plus

正式重测直接使用 `data/browsecomp_plus/questions.jsonl` 的完整 830 题，不再按 fold 或 split 拆分。
GraphPTC、Fewshot PTC 和 Direct Tool Calling 三个 arm 使用相同数据、模型预算、本地 BM25 retriever
和冻结 grader。

## 必需文件

- `browse_comp_test_set.csv`：加密上游问题/答案；
- `questions.jsonl`：完整 830 题运行输入；
- `corpus_parquet/`、`corpus.sqlite3`：本地语料与索引；
- `qrel_golds.txt`、`qrel_evidence.txt`；
- `qwen3-tokenizer/` 和官方索引 manifest。

这些大文件默认不进入 Git。若不从原机器同步，可使用 `scripts/browsecomp_plus/` 下的下载脚本准备
官方索引与 tokenizer。

## 服务与预检

```bash
bash scripts/browsecomp_plus/run_retriever.sh
.venv/bin/python -m graphptc inspect-browsecomp-plus \
  --config configs/browsecomp_plus/browsecomp_plus.graphptc-full.toml
```

预检会加载完整 830 题并请求 `/metadata`，核对 backend、`top_k`、snippet token 限制和索引信息；
`/health` 成功不能替代这一步。

正式新模型运行由 `scripts/evaluation/full_suite.py` 生成独立 profile。三份模板为：

- `browsecomp_plus.graphptc-full.toml`
- `browsecomp_plus.fewshot-ptc-full.toml`
- `browsecomp_plus.direct-tools-full.toml`

历史结果虽然由四个旧分片产生，但只用于审计既有结果，不能作为新评测的启动方式。
