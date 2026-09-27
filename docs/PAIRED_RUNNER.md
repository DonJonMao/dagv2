# 成对实验启动、恢复和结果解释

这是 RAG **推理与评测实验**，不会更新模型参数。默认对包内 HotpotQA、2WikiMultihopQA、MuSiQue 各 1,000 个问题，以及 PersonaMem-v1 官方 32k 文件全部 589 题，分别执行原 DAG v2 和融合版，共 **3,589 题、7,178 个方法任务**。同一道题的两方法共享问题、公共选项、可见语料范围和向量。原三套数据仍是包内子集；PersonaMem 的 589 题是所固定官方 32k 文件的全部题目，不代表所有上下文长度设置，也不声称复现 BT 的 train/validation/test 划分。

PersonaMem 的来源、按题可见记忆和单选评测详见 [PERSONAMEM.md](PERSONAMEM.md)。默认省略 `--datasets` 即运行四套数据；仅运行新数据可指定 `--datasets personamem`。

当前融合可靠性版本为 `dagbt_fusion_reliability_v2`，对应 BridgeTree `16809bd` 的协议修订适配。实现变化、来源 ID、局部恢复和输入预算窗口见 [FUSION_RELIABILITY_V2.md](FUSION_RELIABILITY_V2.md)。旧结果没有该诊断时记为 unknown。

## 一键后台运行

默认配置已与 BT 当前部署一致：DeepSeek-V4-Flash、Qwen3-Embedding-8B、Qwen3-Reranker-8B，具体服务和继承来源见 [模型对齐说明](MODEL_ALIGNMENT.md)。可将 `configs/paired.example.json` 复制为本地配置覆盖端点。原版和融合版共用这些模型；原版方法本身不额外调用 reranker。首次运行会在后台先重建 Qwen3 语料索引，保留旧 NV 数组；维度相同也不会混用。

凭证优先从 `DAG_LLM_API_KEY`（或 `BRIDGETREE_CHAT_API_KEY`）、`DAG_EMBED_API_KEY`、`DAG_RERANK_API_KEY` 读取。当前 BT LLM 地址已有代码内置默认 key，无需手工配置；其他端点和服务也支持 Git 忽略的本地凭据文件，严格校验其绑定端点。实验 JSON 配置不接受 API key 字段，URL 不接受账号、密码或 query 参数。新增依赖见 `requirements-fusion.txt`；脚本优先使用仓库 `.venv/bin/python`，可通过 `DAGBT_PYTHON` 指定解释器。支持 macOS/Linux，使用 POSIX 文件锁和信号。

```bash
# 只检查文件、依赖和配置，不访问模型端点，也不进行推理。
bash scripts/run_paired.sh preflight --config configs/paired.local.json --offline-preflight

# 线上检查：GET /models、reranker 五次一致性探测，以及一次真实融合 planner 协议检查。
bash scripts/run_paired.sh preflight --config configs/paired.local.json

# 先用每个数据集的前 2 题检查实际 JSON 输出和模型接口兼容性。
# 首次仍需为所选数据集的完整语料建索引。
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/paired_smoke_v2 --limit 2

# 完整的默认双臂实验：原三套各 1,000 题，加 PersonaMem 589 题。
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/paired_full_reliability_v2

bash scripts/run_paired.sh status --output outputs/paired_full_reliability_v2
bash scripts/run_paired.sh stop --output outputs/paired_full_reliability_v2
```

省略动作表示 `launch`：先同步预检，再用 `start_new_session=True` 脱离终端，标准输入断开，输出写入 `launcher.log`。进程可在终端关闭后运行。不需要 `nohup`，不启动或重启模型服务。`run` 是前台版本；启动成功不等于实验完成，以 `status` 和 `progress.json` 为准。

reranker 预检直接调用当前 BT 源码的 pointwise consistency probe：空集合序列化、两条固定文本的单条分数，要与混合和倒序批次一致。探测报告、请求、重试、usage 保存在 `preflight_calls/`，独立于每题预算。默认 BT profile 的融合臂另执行一次固定虚构问题的真实 planner 请求：正式 Reasoner/schema/validator，无标签、无格式修复、无协议切换；失败阻止后台启动。其原始请求响应、metadata、校验和费用在 `preflight_calls/planner-*/`，不进入题预算。后台进程只复用 5 分钟内配置、源码、数据范围和 arms 一致、且 planner 探测通过的本次 launch 报告，但重新核验原始文件。通过只证明这组探针的一致性，不证明所有输入上的契约或相关性质量。

离线预检检查 NumPy、Transformers、PyYAML、Jinja2，记录依赖、源码和语料身份。BT 配置检查 chat 消息适配和明确标注的 token 估算，报告派生索引 `ready/needs_build`；历史配置则加载包内 tokenizer 渲染模板。在线 `/models` 返回的所选模型 metadata 另存；配置中的 `deployment_identity` 和 reranker 同名字段作为操作者声明记录。模型 ID、服务 metadata 不自动等于权重校验和或不可变 revision。

`--offline-preflight` 显式跳过全部端点和 planner 协议检查，**不会**把 `run`/`launch` 变成离线实验。离线机制测试使用 pytest 的显式模拟服务，生产运行没有假结果回退。

## 固定实验与显式恢复

每个数据集、方法对应一个独立持久工作进程，准备语料和向量一次，然后逐题执行；不同数据集不共享原仓库的模块全局变量。每题轮换方法执行顺序，避免总让同一方法承担服务预热。默认双臂每题各执行一次；HTTP 重试仍受 `max_identical_attempts` 限制。

协调进程给每个方法的每道题设独立 wall-clock 超时。超时、进程退出、异常或非 `ok` 答案保存为单独失败记录，接着运行另一个方法和下一题。超时/崩溃的工作进程会终止并在下题重新准备资源，正常任务不会重新加载向量。单个请求失败不会把全部实验抛弃。

重新执行相同命令会跳过已有成功和已有失败的终态记录，继续尚未完成的题。只有显式指定 `--retry-failed` 才重跑失败终态；每次启动至多给每个失败任务新增一次尝试，累计次数由 `max_question_attempts`（示例为 3）限制：

```bash
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/paired_full_reliability_v2 --retry-failed
```

每次尝试有独立目录，保留旧失败、请求和重试日志。新尝试仅复用旧尝试中已经成功的完全相同请求缓存；失败 HTTP 请求进入新的、有上限的重试窗口。达到总尝试上限后保留失败并继续。协调进程中断留下的未终态尝试也计入上限，恢复不假装重建搜索中途的 Python 状态，而是从该题起点确定性重放成功缓存。

`manifest.json` 固定配置、问题顺序/摘要、原始包 manifest 和新增源码/vendor 的 SHA256，并记录 PersonaMem 的数据 manifest。更改方法、代码、配置、题目范围都必须换输出目录；不能把 smoke 目录直接变成 full 目录，也不能沿用加入 PersonaMem 前的全量输出目录，也不能用可靠性 v2 续跑旧 PersonaMem 包的 `outputs/paired_full_personamem`。OS `flock` 防止两个写进程使用同一目录；过期 PID 文件不会阻止恢复。`stop` 校验活跃文件锁和进程命令，SIGTERM 触发 finally 终止工作进程，避免只停掉外层留下任务。

## 原版保持什么

`original` 保留原 planner、dense 查询、节点检索、闭包选择和 reader chain 算法，`experiment_v6` 仍装载原 controller/reader。历史配置继续直接调用原 `e.archive`；BT 配置的等价 archive 适配仅将固定 4096 维校验改为当前索引维度。两方法共同使用新的向量、chat 适配和 BT 模型部署；原版继续使用此前 token 估算与 JSON schema 格式指令。融合 v2 的证据请求使用独立的保守估算和固定 `response_format`，默认 plain，亦可显式 json_object/json_schema；最终 reader 不继承该证据格式约束。原始文件未修改，但不能把新模型运行称为旧 Qwen/NV 实验的直接复现。

PersonaMem 接入当前成对入口的 `original` 臂，保留其原求解流程，同时增加按题语料隔离和单选 reader 协议；没有修改历史 `original-dagv2` 分支、原始 57 个文件或 BT vendor 快照。两臂接收同样的公共选项，并要求最终只输出一个选项标签。`configs/paired.legacy.json` 不支持 PersonaMem，使用历史配置时须显式加 `--datasets hotpotqa 2wikimultihopqa musique`。

融合版默认 raw-memory reader，与原版 chain reader 不完全匹配。因此 **original vs fusion 是系统级对比，不能单独证明 BT 或依赖选择的因果贡献**。原始目录 `dagv2/`、`package/`、`data/` 中的原文件和原有脚本用 `original_manifest.json` 核验；新增 PersonaMem 文件用 `data/personamem/manifest.json` 核验。标签文件哈希检查延迟到评分阶段，生成阶段不打开标签。

## 必需的机制对照

相同融合 solver、reader、模型和 token 计量方式下，建议完整 2×2：

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

当四个融合家族方法齐全时，`summary.json` 额外报告 `(fusion - bt_flat) - (dense_dependency - dense_flat)` 的指标交互项：原三套数据使用 F1/EM/检索指标，PersonaMem 使用 accuracy，分别使用全任务和四方法共同成功子集；original 失败不会排除四方法均成功的题。保留描述性点估计，并在评分阶段进行 1,000 次、固定 seed=20260918 的同题配对 percentile bootstrap，给出 95% 区间：每次按问题索引同步抽取四臂，不能独立抽各方法。样本数小于 2 时明确报告无法估计。区间假设问题独立、可交换；共享语料和 PersonaMem 同 persona 的问题相关性可能削弱独立性，区间也不包含模型再次生成的随机性。这不是按 persona 聚类的 bootstrap，也不等于因果证据；共同成功筛选也可能改变被评估的题目分布。

可按需加入 `fusion_no_conditions`、`fusion_single_support`、`fusion_no_invalidation`、`fusion_fixed_dag`、`fusion_chain`，分别检查条件审计、多支持方案、失效传播、细化节点和 chain reader。方法名字与最终语义以 `dagbt/config.py` 为准。融合配置预算是**每题、每方法、每次 attempt** 的共享上限，不能按节点重置；显式 `--retry-failed` 开始新 attempt，因此可能额外产生请求和 token，`cost_all_attempts` 累加全部尝试成本。相同 ANN 上限不代表相同总成本，必须报告 LLM、reranker、缓存、HTTP 重试和实际 token。

固定候选池的机制分析可在配置中提供 `fixed_candidate_pools: {"question_id": ["doc_id", ...]}`，让融合家族的方法在该题使用相同可见候选；整个映射进入 manifest，因此不能在恢复时悄悄替换。该设置不改变 `original` 原始流程，只用于融合家族内部的 same-pool 对照。候选池应由与标签隔离的检索阶段产生，不得用 gold support 组装。

`fusion_navigation_closure` 把候选的首次发现导航路径原文也强制放进最终上下文，用于检验将“帮助找到证据的路径”与“真正支持答案的来源”混同的代价。额外原文计入同一 token 与文档数预算，不伪造新的语义支持边。只有候选 ID 的固定候选池没有导航来源，不能用于这一消融。

`fusion_proxy_free` 对应 day2 的 `proxy_mode=none`：复用最新版 BT 的原文条件提案与缺口检索，按需求和 ANN 顺序安排桥接，不创建 set scorer，不请求 reranker。它是显式的无评分代理调度控制，停用了依赖评分的测量与调度机制；不能称为只改了一个 activation 数值而其他搜索行为完全相同。默认 `fusion` 仍是原生完整 `EvidenceBridgeSearcher`。只跑原版与此控制时可以从本地配置中删去 `reranker` 字段；若仍配置 reranker，启动预检会保留 5 次独立探测，其成本单列，不计为该方法的评分调用。

## 日志和评测

- `preflight_calls/planner-*/`：独立部署协议探测，含真实原始请求响应、验证、metadata 和成本；不与题目混合。
- 根目录 `manifest.json`、`preflight.json`、`pid.json`、`progress.json`、`events.jsonl`、`launcher.log`：运行身份、状态、任务终态。
- BT 配置另有 `index_artifacts.json` 和 `index_build/<dataset>/`：共用派生向量身份、批量编码请求、恢复进度和独立索引费用。`preparing_index` 阶段还未进入问题生成。
- `<dataset>/<arm>/rows/<question-hash>.json`：每题当前权威结果、ranking/graph、5/10/20 证据选择、答案和 runner 开销。
- `<dataset>/<arm>/attempts/<question-hash>/attempt-NNN/`：不可覆盖的尝试；原始请求/响应、成功缓存来源、每次逻辑调用事件、方法诊断。
- `<dataset>/<arm>/failures/`：失败历史，即使后续成功仍保留。
- `<dataset>/comparisons.jsonl` 和 `.csv`：相同题目的各方法预测、状态与逐题分数；原三套数据为 EM/F1、title-group recall、all-support，PersonaMem 为 accuracy 和选项标签；JSONL 另外含候选/选择模块指标、具体选集、结果路径和最新尝试成本。
- `<dataset>/summary.json` 和根 `summary.json`：全任务分数/失败率、共同成功子集、成对 delta、答案正确性 rescue/harm、所有尝试的累计请求成本。PersonaMem 另有 `persona_macro_accuracy_percent`。

所有配置的数据集、方法、问题必须先达到成功或记录失败的终态，才会第一次打开 `evaluation_only.json`。金答案、gold support 不进入 planner、检索器、selector 或 reader。MuSiQue 保留原协议的 `musique:` doc ID 规范化；检索指标按 **title group**，命中组内任何文档即命中该组，不替换为单一 doc ID recall。

PersonaMem 对每道题先按 `messages[:end_index]` 截断，再采用 BT 的 `user_assistant_pair` 切分并保留 system persona。向量库存储全部去重记忆，但检索、reranker、桥接、证据校验和 reader 只能访问 `scopes.json` 为当前题列出的文档 ID；全库索引不等于全库可见。两臂都不能检索另一 persona 或该题的未来消息。

全任务答案失败计零，同时显式报告失败率；原三套数据的检索指标仍按实际保存的选集计算。PersonaMem 仅接受单个选项标签，歧义、多选、超范围标签或带解释的答案记录为 `invalid_choice`，计失败和 0 分。其主指标为 `all_task_metrics_percent.accuracy`，同时报告先在每个 persona 内平均、再对 persona 等权平均的 `persona_macro_accuracy_percent`。没有 gold support，因此不报告 recall、all-support 或文本 F1。共同成功结果另报，避免把服务可靠性误当算法差异。多方法时既报全部方法共同成功，也报每个方法与 original 的成对共同成功。rescues/harm 是同题 EM（原三套）或 accuracy（PersonaMem）变化，不是经过干预证明的证据因果效应。

原三套数据的模块诊断把两种证据分开：`gold_candidate_title_group_recall`/`gold_candidate_all_support` 衡量发现阶段是否找到了标注支持，`gold_discovery_minus_selection_recall_at20` 衡量发现后在选择阶段丢失多少标注支持。它们只在评分阶段计算，不为 PersonaMem 伪造这些 gold 指标。`structural_complete_required_at20`、节点 unknown/ambiguous、必要需求覆盖、quote/span protocol 错误、ANN/set-score 用量、reader token 单独汇总。BT 本地计量是估算，`local_token_accounting` 明确标记；API usage 另存，未观测不当作零。模型判断与结构校验不等于 gold 正确性。

成本日志区分逻辑调用、成功缓存命中、HTTP 尝试、重试和成功响应的 token usage。不返回 usage 的服务记为未知；服务器已执行但响应丢失的成本无法完全观测，因此总 token 是下界，不能把缺失 usage 解释成零成本。服务预检不混入方法实验预算，其 usage 单独保存在 preflight。

融合 v2 的每题 `diagnostics.reliability` 记录 `version=dagbt_fusion_reliability_v2`、`cohort`、`mapping_complete`、`input_truncated` 等；失败时保留已落盘 `fusion_partial.json`。每方法摘要新增 `reliability.completion_cohorts`，仅按当前权威成功 rows 统计 normal、truncated、partially_mapped、truncated_and_partially_mapped、unknown 的题数、答对数、accuracy（PersonaMem）或 EM（其他数据集）及 `cost_latest_attempt`；`failed_tasks` 和 `failure_cost_latest_attempt` 单列失败。旧诊断缺失不填 normal，成本缺失不填零，失败历史与 retry 日志不重复算题；`cost_all_attempts` 继续保留所有尝试费用。各组问题不同，这些分组是描述性结果，不能直接归因截断或部分映射的效果。

证据推理默认每请求至多 2 次局部修复、每题至多 6 次，所有 planner/map/resolve/audit/select/repair/rebatch 共用每题每方法每 attempt 的 24 次 LLM attempt 额度，reader 单列。沿用 DAG 的保守计费，每次物理 HTTP 重试也扣 LLM 余额，成功缓存重放按历史已记尝试数扣额度；实际可用逻辑调用可能更少，request 日志中的逻辑调用数另计。预算窗口保留完整记录并限定实际可引用 ID；被省略或未完成的证据不当作未检索到或无关。证据输入估算为 `regex_or_utf8_bytes_div3_v2`，包含 schema、输出预留及 margin，但不是实际服务 tokenizer 上界；原始 reader 的可行性检查仍独立执行。

当前交付验证应以离线测试报告为准；没有启动远程完整实验，也没有据此声称融合性能提升。
