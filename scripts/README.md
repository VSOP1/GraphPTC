# 脚本目录

脚本分为仓库级交付入口和 benchmark 数据/服务工具。正式新模型评测统一从 `evaluation/` 启动，
不再维护各 benchmark 各自的两组 paired launcher。

| 目录 | 内容 |
| --- | --- |
| `evaluation/` | 创建新模型 profile，统一预检、运行和评分六项三组完整评测 |
| `release/` | 生成无密钥 source ZIP、Git bundle、校验和及可选历史结果包 |
| `setup/` | Linux 主环境以及 AppWorld、ToolSandbox、Agent-Diff 隔离环境 |
| `browsecomp_plus/` | 官方索引/tokenizer 与本地 retriever |
| `frames/` | Wikipedia 快照准备与 retriever |
| `intercode/` | 非默认重测项的 Linux/Docker 环境和 gold sanity |

脚本应从自身位置解析仓库根目录，不假定固定的 `D:/GraphPTC` 解压路径。付费运行前先执行对应
CLI 的 `inspect-*` 或 `probe-*`。完整操作见 `docs/server-evaluation.md`；统一入口永远不会自动添加
`--restart`。
