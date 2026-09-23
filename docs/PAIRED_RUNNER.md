# 成对实验启动、恢复和结果解释

这是 RAG **推理与评测实验**，不会更新模型参数。默认对包内 HotpotQA、2WikiMultihopQA、MuSiQue 各 1,000 个问题，分别执行原 DAG v2 和融合版；相同问题 ID、原始文本、语料和向量不变。没有把这 3,000 题称为三个数据集的完整官方规模。

## 一键后台运行

先部署原仓库要求的 Qwen3.8 和 NV-Embed-v2 服务，另外准备最新版 BT 所需的 **pointwise reranker**。将 `configs/paired.example.json` 复制为自己使用的配置，填写模型名和端点；示例 reranker 名是明确的占位符。已有 NV-Embed-v2 语料向量不能搭配其他 embedding 模型使用。

凭证只从 `DAG_LLM_API_KEY`、`DAG_EMBED_API_KEY`、`DAG_RERANK_API_KEY` 读取。配置里禁止写 API key，URL 禁止包含账号、密码或 query 参数。不要把凭证传入命令行。新增依赖见 `requirements-fusion.txt`；脚本优先使用仓库 `.venv/bin/python`，可通过 `DAGBT_PYTHON` 指定解释器。支持 macOS/Linux，使用 POSIX 文件锁和信号。

```bash
# 只检查文件、依赖和配置，不访问模型端点，也不进行推理。
bash scripts/run_paired.sh preflight --config configs/paired.local.json --offline-preflight

# 检查模型服务：GET /models；reranker 做 5 次无标注的单条/混合/倒序一致性探测。
bash scripts/run_paired.sh preflight --config configs/paired.local.json

# 先用每个数据集的前 2 题检查实际 structured JSON、模型和 tokenizer 兼容性。
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/paired_smoke --limit 2

# 完整的默认双臂实验：3 个数据集，各 1,000 题，每题原版和融合版各一次。
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/paired_full

bash scripts/run_paired.sh status --output outputs/paired_full
bash scripts/run_paired.sh stop --output outputs/paired_full
```

省略动作表示 `launch`：先同步预检，再用 `start_new_session=True` 脱离终端，标准输入断开，输出写入 `launcher.log`。进程可在终端关闭后运行。不需要 `nohup`，不启动或重启模型服务。`run` 是前台版本；启动成功不等于实验完成，以 `status` 和 `progress.json` 为准。

reranker 预检直接调用当前 BT 源码的 pointwise consistency probe：空集合序列化、两条固定文本的单条分数，要与混合和倒序批次一致。探测报告、请求、重试、usage 保存在 `preflight_calls/`，独立于每题预算。后台进程复用 5 分钟内、完全相同配置的本次 launch 报告，避免重复探测，但重新核验原始文件。通过只证明这组探针的一致性，不证明所有输入上的契约或相关性质量。

离线预检也实际加载包内 tokenizer 并渲染 chat template，检查 NumPy、Transformers、PyYAML、Jinja2，记录依赖版本、模板输出摘要、tokenizer 文件、当前源码和语料/向量 SHA256。在线 `/models` 返回的所选模型 metadata 另存；配置中的 `deployment_identity` 和 reranker 同名字段作为操作者声明记录。模型 ID、服务 metadata 不自动等于权重校验和或不可变 revision，未提供的身份不会被编造。

`--offline-preflight` 仅跳过服务检查，**不会**把 `run`/`launch` 变成离线实验。离线机制测试使用 pytest 的显式模拟服务，生产运行没有假结果回退。

## 固定实验与显式恢复

每个数据集、方法对应一个独立持久工作进程，准备语料和向量一次，然后逐题执行；不同数据集不共享原仓库的模块全局变量。每题轮换方法执行顺序，避免总让同一方法承担服务预热。默认双臂每题各执行一次；HTTP 重试仍受 `max_identical_attempts` 限制。

协调进程给每个方法的每道题设独立 wall-clock 超时。超时、进程退出、异常或非 `ok` 答案保存为单独失败记录，接着运行另一个方法和下一题。超时/崩溃的工作进程会终止并在下题重新准备资源，正常任务不会重新加载向量。单个请求失败不会把全部实验抛弃。

重新执行相同命令会跳过已有成功和已有失败的终态记录，继续尚未完成的题。只有显式指定 `--retry-failed` 才重跑失败终态；每次启动至多给每个失败任务新增一次尝试，累计次数由 `max_question_attempts`（示例为 3）限制：

```bash
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/paired_full --retry-failed
```

每次尝试有独立目录，保留旧失败、请求和重试日志。新尝试仅复用旧尝试中已经成功的完全相同请求缓存；失败 HTTP 请求进入新的、有上限的重试窗口。达到总尝试上限后保留失败并继续。协调进程中断留下的未终态尝试也计入上限，恢复不假装重建搜索中途的 Python 状态，而是从该题起点确定性重放成功缓存。

`manifest.json` 固定配置、问题顺序/摘要、原始包 manifest 和新增源码/vendor 的 SHA256。更改方法、代码、配置、题目范围都必须换输出目录；不能把 smoke 目录直接变成 full 目录。OS `flock` 防止两个写进程使用同一目录；过期 PID 文件不会阻止恢复。`stop` 校验活跃文件锁和进程命令，SIGTERM 触发 finally 终止工作进程，避免只停掉外层留下任务。

## 原版保持什么

`original` 调用未修改的 `e.make_plan`、`e.archive`、`e.native.solve`；`experiment_v6` 仍按原仓库方式装载 controller/reader。保留原版 `e.work` 对 `PlanError` 和 context overflow 的处理，仅将文件恢复和 `Calls` 创建移到协调器，以记录真实缓存/请求开销。原版 planner、原始 dense 查询、节点检索、闭包选择、reader chain 注入和采样参数均不调整。

融合版默认 raw-memory reader，与原版 chain reader 不完全匹配。因此 **original vs fusion 是系统级对比，不能单独证明 BT 或依赖选择的因果贡献**。原始目录 `dagv2/`、`package/`、`data/` 和原有脚本用 `original_manifest.json` 核验；标签文件哈希检查延迟到评分阶段，生成阶段不打开标签。

## 必需的机制对照

相同融合 solver、reader 和预算下，建议完整 2×2：

```bash
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/factorial \
  --arms original fusion dense_dependency bt_flat dense_flat
```

| 方法 | 发现方式 | 选择方式 |
|---|---|---|
| `fusion` | 最新 BT 桥接搜索 | 依赖闭包 |
| `dense_dependency` | Dense | 依赖闭包 |
| `bt_flat` | 最新 BT 桥接搜索 | 平面证据选择 |
| `dense_flat` | Dense | 平面证据选择 |

当四个融合家族方法齐全时，`summary.json` 额外报告 `(fusion - bt_flat) - (dense_dependency - dense_flat)` 的 F1/EM/检索指标交互项，分别使用全任务和四方法共同成功子集；original 失败不会排除四方法均成功的题。保留描述性点估计，并在评分阶段进行 1,000 次、固定 seed=20260918 的同题配对 percentile bootstrap，给出 95% 区间：每次按问题索引同步抽取四臂，不能独立抽各方法。样本数小于 2 时明确报告无法估计。区间假设问题独立、可交换；共享语料可能削弱独立性，区间也不包含模型再次生成的随机性。它不等于因果证据；共同成功筛选也可能改变被评估的题目分布。

可按需加入 `fusion_no_conditions`、`fusion_single_support`、`fusion_no_invalidation`、`fusion_fixed_dag`、`fusion_chain`，分别检查条件审计、多支持方案、失效传播、细化节点和 chain reader。方法名字与最终语义以 `dagbt/config.py` 为准。融合配置预算是**每题、每方法、每次 attempt** 的共享上限，不能按节点重置；显式 `--retry-failed` 开始新 attempt，因此可能额外产生请求和 token，`cost_all_attempts` 累加全部尝试成本。相同 ANN 上限不代表相同总成本，必须报告 LLM、reranker、缓存、HTTP 重试和实际 token。

固定候选池的机制分析可在配置中提供 `fixed_candidate_pools: {"question_id": ["doc_id", ...]}`，让融合家族的方法在该题使用相同可见候选；整个映射进入 manifest，因此不能在恢复时悄悄替换。该设置不改变 `original` 原始流程，只用于融合家族内部的 same-pool 对照。候选池应由与标签隔离的检索阶段产生，不得用 gold support 组装。

`fusion_navigation_closure` 把候选的首次发现导航路径原文也强制放进最终上下文，用于检验将“帮助找到证据的路径”与“真正支持答案的来源”混同的代价。额外原文计入同一 token 与文档数预算，不伪造新的语义支持边。只有候选 ID 的固定候选池没有导航来源，不能用于这一消融。

`fusion_proxy_free` 对应 day2 的 `proxy_mode=none`：复用最新版 BT 的原文条件提案与缺口检索，按需求和 ANN 顺序安排桥接，不创建 set scorer，不请求 reranker。它是显式的无评分代理调度控制，停用了依赖评分的测量与调度机制；不能称为只改了一个 activation 数值而其他搜索行为完全相同。默认 `fusion` 仍是原生完整 `EvidenceBridgeSearcher`。只跑原版与此控制时可以从本地配置中删去 `reranker` 字段；若仍配置 reranker，启动预检会保留 5 次独立探测，其成本单列，不计为该方法的评分调用。

## 日志和评测

- 根目录 `manifest.json`、`preflight.json`、`pid.json`、`progress.json`、`events.jsonl`、`launcher.log`：运行身份、状态、任务终态。
- `<dataset>/<arm>/rows/<question-hash>.json`：每题当前权威结果、ranking/graph、5/10/20 证据选择、答案和 runner 开销。
- `<dataset>/<arm>/attempts/<question-hash>/attempt-NNN/`：不可覆盖的尝试；原始请求/响应、成功缓存来源、每次逻辑调用事件、方法诊断。
- `<dataset>/<arm>/failures/`：失败历史，即使后续成功仍保留。
- `<dataset>/comparisons.jsonl` 和 `.csv`：相同题目的各方法预测、状态、EM/F1、title-group recall、all-support；JSONL 另外含候选/选择模块指标、具体选集、结果路径和最新尝试成本。
- `<dataset>/summary.json` 和根 `summary.json`：全任务分数/失败率、共同成功子集、成对 delta、EM rescue/harm、所有尝试的累计请求成本。

所有配置的数据集、方法、问题必须先达到成功或记录失败的终态，才会第一次打开 `evaluation_only.json`。金答案、gold support 不进入 planner、检索器、selector 或 reader。MuSiQue 保留原协议的 `musique:` doc ID 规范化；检索指标按 **title group**，命中组内任何文档即命中该组，不替换为单一 doc ID recall。

全任务答案失败计零，同时显式报告失败率；检索指标仍按实际保存的选集计算。共同成功结果另报，避免把服务可靠性误当算法差异。多方法时既报全部方法共同成功，也报每个方法与 original 的成对共同成功。rescues/harm 是同题 EM 变化，不是经过干预证明的证据因果效应。

模块诊断把两种证据分开：`gold_candidate_title_group_recall`/`gold_candidate_all_support` 衡量发现阶段是否找到了标注支持，`gold_discovery_minus_selection_recall_at20` 衡量发现后在选择阶段丢失多少标注支持。它们只在评分阶段计算。`structural_complete_required_at20`、节点 unknown/ambiguous、必要需求覆盖、quote/span protocol 错误、ANN/set-score 用量、实际 reader token 单独汇总；这些模型判断与结构校验不等于 gold 正确性。每项汇总都有 observed/missing 分母，失败或未观测不能伪装成完整闭包或零 token。

成本日志区分逻辑调用、成功缓存命中、HTTP 尝试、重试和成功响应的 token usage。不返回 usage 的服务记为未知；服务器已执行但响应丢失的成本无法完全观测，因此总 token 是下界，不能把缺失 usage 解释成零成本。服务预检的轻量探测不混入方法实验预算，其 usage 单独保存在 preflight。

当前交付验证应以离线测试报告为准；没有启动远程完整实验，也没有据此声称融合性能提升。
