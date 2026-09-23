# 上传部署包并运行实验

这是调用现有模型服务的 RAG 推理与评测，不更新模型参数。运行入口同时执行 `original` 和 `fusion`；无需在实验机器上部署模型或配置 GPU。实验机器需有 Python 3.10+（建议 3.11）、venv/pip、Bash、`ps`（Linux 的 procps），以及到模型服务和 Python 依赖源的网络。

当前默认服务为：LLM `111.19.156.30:8006`、embedding `111.19.156.74:8001`、reranker `111.19.156.74:8002`。LLM 默认 key 已内置，环境变量可覆盖；无需手工创建凭据文件。部署包不包含模型权重，不启动模型服务。

## 1. 上传并解压

把 `dagv2_bt_deploy_20260923.tar.gz` 上传到服务器。例如在本机运行以下命令，把 `USER@SERVER` 替换为实际 SSH 用户和地址：

```bash
scp /Users/mao/projects/dagv2_bt_deploy_20260923.tar.gz USER@SERVER:~/
```

登录服务器后，在一个新的实验目录中解压：

```bash
mkdir -p ~/dagbt_runs/20260923
tar -xzf ~/dagv2_bt_deploy_20260923.tar.gz -C ~/dagbt_runs/20260923
cd ~/dagbt_runs/20260923/dagv2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-fusion.txt
```

部署包包含全部原始数据和三个旧 NV 向量文件，完整性预检仍需要它们。它们不会被当作新 Qwen3 索引使用。包内不带本机虚拟环境、Git 历史、实验输出或本地凭据文件；BT 源码已经内置，无需再上传另一个 BT 仓库。

## 2. 检查环境，然后小规模试跑

```bash
bash scripts/run_paired.sh preflight --offline-preflight
bash scripts/run_paired.sh --datasets hotpotqa --limit 2 --output outputs/smoke_hotpot
bash scripts/run_paired.sh status --output outputs/smoke_hotpot
tail -f outputs/smoke_hotpot/launcher.log
```

第二条命令先检查三个模型服务，再启动后台任务。无需额外使用 `nohup`；断开 SSH 后任务仍运行。查看日志时按 Ctrl-C 只结束 `tail`。

首次试跑也会先为 HotpotQA 的完整语料建立新索引，显示 `preparing_index` 属于正常阶段。索引存放在 `data/derived/`，后续相同配置可复用。`launcher.log` 不一定持续更新，以 `status` 和 `progress.json` 为主要进度来源；编码批次进度见 `outputs/smoke_hotpot/index_build/hotpotqa/progress.json`。离线预检不调用模型；它通过不代表远程服务可用，也不代表真实推理成功。

等待试跑结束后检查 `outputs/smoke_hotpot/hotpotqa/comparisons.csv` 的两方法状态和 `outputs/smoke_hotpot/summary.json` 的失败率；任务完成可能包含已记录的失败。先排查接口、鉴权、网络或模型输出错误，再开始正式实验。

## 3. 正式运行

```bash
bash scripts/run_paired.sh --output outputs/paired_full
bash scripts/run_paired.sh status --output outputs/paired_full
tail -f outputs/paired_full/launcher.log
```

默认运行包内 HotpotQA、2WikiMultihopQA、MuSiQue 各 1,000 题，每题 original 和 fusion 各一次，共 6,000 个方法任务，每个任务可能包含多次模型调用。已有 HotpotQA 索引会被复用，其余语料索引在运行前自动构建。评测在所有任务均成功或记录失败后生成。

## 4. 结果、停止与恢复

主要结果：

- `outputs/paired_full/summary.json`：总览。
- `outputs/paired_full/<dataset>/summary.json`：数据集分数、失败率和成本。
- `outputs/paired_full/<dataset>/comparisons.csv`：逐题 original/fusion 对比。
- `outputs/paired_full/progress.json`：当前进度。

```bash
# 停止后台任务
bash scripts/run_paired.sh stop --output outputs/paired_full

# 使用原配置和原输出目录恢复未完成任务
bash scripts/run_paired.sh --output outputs/paired_full

# 显式重试失败任务（受配置的累计尝试上限约束）
bash scripts/run_paired.sh --output outputs/paired_full --retry-failed
```

成功任务会跳过；默认也跳过已有失败终态。代码、配置、数据集或题目范围改变后，使用新的输出目录。不要将试跑输出目录用于正式全量运行。

如需改服务地址，复制 `configs/paired.example.json` 为 `configs/paired.local.json` 后修改，并在每次启动命令中加 `--config configs/paired.local.json`。内置 LLM key 只绑定当前地址，改地址后通过 `DAG_LLM_API_KEY` 配置对应服务的 key。

本机仅验证了离线运行和模拟接口；真实模型服务的可达性与真实推理需在目标服务器上通过上述试跑确认。完整运行选项及机制对照见 [PAIRED_RUNNER.md](PAIRED_RUNNER.md)。
