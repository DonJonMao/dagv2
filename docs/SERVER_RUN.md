# 上传部署包并运行实验

这是调用现有模型服务的 RAG 推理与评测，不更新模型参数。运行入口同时执行 `original` 和 `fusion`；无需在实验机器上部署模型或配置 GPU。实验机器需有 Python 3.10+（建议 3.11）、venv/pip、Bash、`ps`（Linux 的 procps），以及到模型服务和 Python 依赖源的网络。

当前入口为 `scripts/run_v3.sh`，默认输出 `outputs/paired_full_reliability_v3`。v3 改变融合选集复核和 PersonaMem 问题协议，须解压到新项目目录、新建结果目录；保留旧实验，不要用新代码恢复旧运行身份。

当前默认服务为：LLM `111.19.156.30:8006`、embedding `111.19.156.74:8001`、reranker `111.19.156.74:8002`。LLM 默认 key 已内置，环境变量可覆盖；无需手工创建凭据文件。部署包不包含模型权重，不启动模型服务。

## 1. 上传并解压

把可靠性 v3 新包 `dagv2_bt_deploy_20260930_reliability_v3.tar.gz` 上传到服务器；它保留全部四套数据，并在 v2 来源引用与响应恢复的基础上，适配 BT `c4b04c9` 的个人历史提示、独立原文复核和有界覆盖恢复。旧包不包含本轮方法修改。例如在本机运行以下命令，把 `USER@SERVER` 替换为实际 SSH 用户和地址：

```bash
scp /Users/mao/projects/dagv2_bt_deploy_20260930_reliability_v3.tar.gz USER@SERVER:~/
```

登录服务器后，在一个新的实验目录中解压：

```bash
mkdir -p ~/dagbt_runs/20260930_reliability_v3
tar -xzf ~/dagv2_bt_deploy_20260930_reliability_v3.tar.gz -C ~/dagbt_runs/20260930_reliability_v3
cd ~/dagbt_runs/20260930_reliability_v3/dagv2
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-fusion.txt
```

部署包包含全部原始数据、三个旧 NV 向量文件，以及已处理的 PersonaMem-v1 32k 全部 589 题、记忆、按题可见范围和独立评测标签；无需在服务器另行下载 PersonaMem。数据来源与 SHA256 见 `data/personamem/manifest.json`。完整性预检仍需要旧 NV 文件，但不会把它们当作新 Qwen3 索引使用。包内不带本机虚拟环境、Git 历史、实验输出或本地凭据文件；BT 源码已经内置，无需再上传另一个 BT 仓库。

## 2. 检查环境，然后小规模试跑

```bash
bash scripts/run_v3.sh preflight --offline-preflight
bash scripts/run_v3.sh start --datasets personamem --limit 2 --output outputs/smoke_personamem_v3
bash scripts/run_v3.sh status --output outputs/smoke_personamem_v3
tail -f outputs/smoke_personamem_v3/launcher.log
```

第二条命令先检查三个模型服务，并用固定虚构问题发送一次真实融合 planner 请求；采用正式 schema/validator，无格式修复和协议切换。检查失败不会启动后台任务。原始请求响应及费用独立保存在 `preflight_calls/planner-*/`，不计入题目成本；通过仅证明这次 planner 请求可用，仍须检查后续试跑。无需额外使用 `nohup`；断开 SSH 后任务仍运行。查看日志时按 Ctrl-C 只结束 `tail`。

首次试跑也会先为 PersonaMem 全部 3,187 条去重记忆建立新索引，显示 `preparing_index` 属于正常阶段。物理共享索引只减少重复编码，每道题仍严格使用其 persona 和时间截断内的记忆。索引存放在 `data/derived/`，后续相同配置可复用。`launcher.log` 不一定持续更新，以 `status` 和 `progress.json` 为主要进度来源；编码批次进度见 `outputs/smoke_personamem_v3/index_build/personamem/progress.json`。离线预检不调用模型；它通过不代表远程服务可用，也不代表真实推理成功。`--offline-preflight` 显式跳过全部端点和 planner 协议检查，但不会让实验离线运行；正常服务器启动不要加这个参数。

等待试跑结束后检查 `outputs/smoke_personamem_v3/personamem/comparisons.csv` 的两方法状态和 `outputs/smoke_personamem_v3/summary.json` 的失败率；任务完成可能包含已记录的失败。PersonaMem reader 要求只返回一个选项标签，`invalid_choice` 表示答案不满足单选协议，按失败和 0 分处理。先排查接口、鉴权、网络或模型输出错误，再开始正式实验。

还可只读查看原文选择、覆盖状态和调用分层：

```bash
bash scripts/run_v3.sh diagnostics --output outputs/smoke_personamem_v3
```

`diagnostics` 不调用模型，也不加载评分标签。`raw_only`/`mixed` 表示选集中存在没有映射的原文，不自动代表来源错误；`coverage_state=unassessed` 表示覆盖标注未完成验证，不应读成已覆盖。相反，`normal` 只说明没有记录输入裁剪或部分映射，不能证明上下文充分。

## 3. 正式运行

```bash
bash scripts/run_v3.sh start
bash scripts/run_v3.sh status
bash scripts/run_v3.sh diagnostics
tail -f outputs/paired_full_reliability_v3/launcher.log
```

不加 `--datasets` 默认运行四套数据：HotpotQA、2WikiMultihopQA、MuSiQue 各 1,000 题，PersonaMem-v1 32k 全部 589 题。每题 original 和 fusion 各一次，共 **3,589 题、7,178 个方法任务**，每个任务可能包含多次模型调用。已有 PersonaMem 索引会被复用，其余语料索引在运行前自动构建。评测在所有任务均成功或记录失败后生成。

PersonaMem 报告严格单选 `accuracy` 和 `persona_macro_accuracy_percent`；原三套数据继续报告原有问答和检索指标。PersonaMem 没有 gold support，不计算其 recall 或文本 F1。完整数据协议见 [PERSONAMEM.md](PERSONAMEM.md)。v3 的原文复核、DAG 支持证书、预算和验证边界见 [FUSION_RELIABILITY_V3.md](FUSION_RELIABILITY_V3.md)。默认 `fusion.response_format=plain`，其他可选模式为 `json_object` 和 `json_schema`，必须先确认当前部署支持；修改后新开运行目录，服务拒绝不会自动降级协议。

融合 PersonaMem 的规划、检索与证据阶段只看用户当前问题；公开选项独立传给最终 reader。`original` 保留既有问题加公开选项的协议，因此双臂分数不是只隔离 BT 的单因素对照。DAG 默认上下文仍为 16,384，包含 reader 的 1,024 输出预留；独立 BT 仓库的 8,192 配置不覆盖这里的预算。

## 4. 结果、停止与恢复

主要结果：

- `outputs/paired_full_reliability_v3/summary.json`：四套数据总览。
- `outputs/paired_full_reliability_v3/<dataset>/summary.json`：数据集分数、失败率和成本。
- `outputs/paired_full_reliability_v3/<dataset>/comparisons.csv`：逐题 original/fusion 对比。
- `outputs/paired_full_reliability_v3/progress.json`：当前进度。

```bash
# 停止后台任务
bash scripts/run_v3.sh stop

# 使用原配置和原输出目录恢复未完成任务
bash scripts/run_v3.sh resume

# 显式重试失败任务（受配置的累计尝试上限约束）
bash scripts/run_v3.sh resume --retry-failed
```

成功任务会跳过；默认也跳过已有失败终态。v3 的结果在 `summary.json` 的各方法 `reliability.completion_cohorts` 下按 normal、truncated、partially_mapped、两者同时发生及 unknown 分组，单列答对数、accuracy/EM 和最新尝试成本；失败成本单列，`cost_all_attempts` 仍累计全部尝试。代码、配置、数据集或题目范围改变后，使用新的输出目录。不要将试跑输出目录用于正式全量运行，也不要复用旧包的 `outputs/paired_full` 或 `outputs/paired_full_personamem` 目录。本次源码和协议改变了运行身份，不能用 v3 resume v2 或 v1；旧项目和日志另行保留。

方法级 `semantic_evidence` 另外按 empty/mapped/raw/mixed 与覆盖 complete/unassessed/unknown 汇总，不与协议可靠性混为一项。逻辑证据调用、预算内物理尝试、缓存、reader 与缺失 usage 单独保留。查看状态或诊断使用的 `--output` 必须与启动相同；`resume` 也须恢复原先的 `--config`、`--datasets`、`--arms`、`--limit` 等参数。例如恢复上述两题试跑：

```bash
bash scripts/run_v3.sh resume --datasets personamem --limit 2 --output outputs/smoke_personamem_v3
```

若要独立检查 raw channel，可在新的消融目录使用 `--arms fusion fusion_no_raw_review`。后者仍建立相同规则的 dense 基线并执行相同复核流程，只关闭独立原文输入；不要用 `selection_review=false` 冒充这个消融，因为它会改变整个选择机制和 baseline 阶段。

如需改服务地址，复制 `configs/paired.example.json` 为 `configs/paired.local.json` 后修改，并在每次启动命令中加 `--config configs/paired.local.json`。内置 LLM key 只绑定当前地址，改地址后通过 `DAG_LLM_API_KEY` 配置对应服务的 key。

`configs/paired.legacy.json` 仅适用于原三套数据，不能用于 PersonaMem；如需运行历史配置，必须同时显式指定 `--datasets hotpotqa 2wikimultihopqa musique`。

本机已完成的真实 BT 导出回放只验证原文可见性，不代表 DAG 新答案或准确率；完整 v3 回归、真实开发子集和新包校验以 [v3 验证记录](FUSION_RELIABILITY_V3.md) 的最终结果为准。这里不预先宣称 Linux 目标机器已实测，目标服务器仍需通过上述试跑检查实际服务与完整路径。完整运行选项及机制对照见 [PAIRED_RUNNER.md](PAIRED_RUNNER.md)。
