# 数据目录

这里保存可分发的小型数据、任务 manifest、冻结选择和 provenance。大型语料、索引、Kiwix ZIM、
缓存和受许可限制的数据由 `scripts/<benchmark>/` 或官方环境准备，通常不会进入 Git。
从空 Git clone 准备六项正式评测时，统一按
[数据集下载与准备](../docs/dataset-setup.md)中的官方来源、固定 revision 和验收命令执行。

不要移动或改写已经被运行签名引用的数据。新增数据时应记录来源、revision、选择规则、预期数量和
SHA-256；开发子集、合成 fixture 与官方 split 必须明确区分。

## 当前保留规则

- `browsecomp_plus/browse_comp_test_set.csv` 是加密上游问题/答案数据，正式配置和默认 loader
  都会读取它，必须保留。
- `browsecomp_plus/questions.jsonl` 是正式重测使用的完整 830 题数据，不需要拆分。
- `browsecomp_plus/pilot100.questions.jsonl`、`pilot100-fold2.questions.jsonl`、
  `pilot100-fold3.questions.jsonl` 与 `remaining530.questions.jsonl` 虽沿用历史命名，但共同组成
  已有结果使用的四个历史分片；为保留旧运行 provenance 暂不清理，但新评测不再使用。
- benchmark 子目录中的 freeze、manifest、qrel、语料和索引用于固定数据选择或重建本地服务；只有
  在确认没有配置、loader、脚本和最终运行签名引用后才能移除。
