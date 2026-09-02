# Agent-Diff

正式范围是冻结 commit `3bb9c40707df23d89e5dbc0e40c424ba38c69ff8` 对应的全部 224 个任务，
每题运行 3 个 trial，采用 `no-docs` 条件和官方 state-diff evaluator。

## 环境

```bash
bash scripts/setup/agent_diff.sh
.venv/bin/python -m graphptc download-agent-diff \
  --config configs/agent_diff/graphptc.toml
```

隔离 SDK 安装在 `external/agent_diff/.venv/`；三份正式配置通过 `{repo}` 自动解析仓库位置，不依赖
Windows、WSL 或固定服务器用户名。服务凭据写入 `.env`：

```dotenv
AGENT_DIFF_API_KEY=
AGENT_DIFF_BASE_URL=
```

## 预检与正式配置

```bash
.venv/bin/python -m graphptc inspect-agent-diff \
  --config configs/agent_diff/graphptc.toml
```

预检核对 SDK、冻结 commit、数据数量、数据 hash 和服务分布。正式模板为：

- `graphptc.toml`
- `fewshot-ptc.toml`
- `direct-tools.toml`

新模型必须通过统一 profile 入口复制三组配置，保持 224 × 3、预算、数据和 evaluator 不变。运行中
断后按 task/trial checkpoint 续跑；不得只重试失败 trial。
