# BT reranker 精确前缀 KV 复用

| 验收问题 | 2026-10-09 当前证据 |
|---|---|
| 哪个后端 | 采集时 `/rerank` 代理报告 `http://127.0.0.1:18002`、`qwen3-reranker-8b`、`/models/Qwen3-Reranker-8B`、8192；匹配历史源码与契约的是 canonical yes/no `/v1/completions` 路径 |
| 缓存原先是否开启 | unknown；代理没有暴露引擎 APC 状态，不能从 vLLM 的默认值推断 |
| 实际模型样本数是否不变 | 真实十题源已终态并冻结：539 次成功 rerank HTTP、854 个不同完整集合评分；off/on 尚未执行，不能证明两臂样本数一致 |
| 复用了多少 | unknown；没有取得内部引擎 KV 指标，结构估计也未作为实测 |
| 耗时降低多少 | unknown；没有部署隔离 APC 双臂，不报告加速或节省比例 |
| 分数/动作变化 | 真实 off/on 未验证。离线测试覆盖误差、排序、阈值翻转与冻结 BT 的实际队列/四项比较 |
| 能否部署 | 工具与隔离配置可交付，生产不启用；模型 HTTP 当前不可达，隔离引擎执行入口/设备仍缺失 |

**状态：`IMPLEMENTED_NOT_HARDWARE_VERIFIED`。** 这不是已验证加速。当前工作区 HEAD 是 `972625124e113aa542d760ce81928c1502af739c`，分支 `feat/dag-resolve`。保留原有未跟踪研究文档、PDF、结果压缩包及 `tmp/`；未 reset、切换分支、重启生产、清除生产缓存或修改驱动。

## 实际链路与验证边界

`dagbt.bridge._Scorer` 按 timestamp/ID 规范顺序序列化完整集合；`_Reranker` 每批最多 4 个集合，用 `top_n=len(batch)` 请求 `/rerank`。空集合也是一个完整评分样本，文本为 `[No evidence passages]`。`vendor/bridgetree` 的内存/持久集合结果缓存与 Transport 响应缓存保留。优化工具只在独立回放中绕过结果缓存。

在线 `/health` 返回 `canonical_yes_no_raw_logprobs_v1`、yes ID 9693、no ID 2152，`submitted_prompt_token_ids`，显式拒绝截断。`backend_tokenizer_alignment` 仍是 `unverified`。线上合成调用 HTTP 200，2 个索引完整，报告上游请求数 2、完整 prompt tokens 352、completion tokens 4。这支持当前评分协议，不证明权重校验和、模板一致性或 KV 生效。

找到的 2026-09 历史 `label_score_fix/reranker_proxy_api.py` 与当前 health 契约相符：用原 tokenizer/chat template 加完整评分后缀生成 token IDs，分别约束 canonical yes/no 的采样，读取 **raw** logprob，再按原函数归一化。这两次物理前向不能合并或改成另一评分头。线上源码是否与历史文件完全相同尚未核验。

配置声明 `v0.20.2rc1-310p`、FP16、Ascend、seqs4，仅作线索。本机是 macOS arm64，没有 vLLM、torch-npu、Ascend 设备工具；Docker daemon 未运行。代理的 `/metrics`、`/v1/models` 返回 404，内部 18002 从外部直连无有效响应（已在沙箱外复核）。因此实际引擎/Ascend/CANN 版本、设备、dtype/KV dtype、block size、attention/RoPE、APC 状态与容量均为 unknown。

只读核查了官方 APC 机制文档、v0.18.0 Ascend reranker 教程，以及上游 v0.20.2 的参数/Completion 协议源码。它们不是已安装 `v0.20.2rc1-310p` 的证据，不能据此决定部署参数或 salt 支持。

## 工作负载证据

已有 PersonaMem 真实事件日志共 490,339 行，含多个事件副本、逻辑请求、物理尝试和文档 hash 描述，**没有完整 query/documents payload**。不能把每行当作模型样本。历史 regression 包有 194 条完整 payload、103 个问题身份，但专门选择 singleton 500/timeout 的失败案例，不符合本轮主基准固定抽样规则；没有把它用作真实收益实验。紧凑答案包也不包含完整模型请求。

新提取器支持已有 DAG-BT `call_events.jsonl` 与 `requests/<ref>.json`，校验原 Transport 身份，按稳定 dataset/question hash 选最多 50 题，前 10 题为 smoke。不按分数、成功或命中率选题。失败、HTTP 重试、缓存命中分别记录；只使用实际物理时间排序，磁盘 SQLite 排序避免把全量原文载入内存。不会从答案或 doc ID 猜正文。

`collect-smoke` 已通过现有在线服务，在新私有目录 `outputs/bt_prefix_cache_20261009/real_smoke_first10_20261009/` 采集 **预先固定的前 10 题**。复用原 runner 的 preflight、ensure_index、单个 fusion Worker 和 generate_dataset，绕过仅允许成对算法实验的 CLI 限制，不改评分和预算，不读取评测 labels。保持原 Transport 重试，每题只运行一个 attempt。为真实输入准备的 PersonaMem 派生 embedding index 已完成 100 批、3187 文档，存于忽略的 `data/derived/`；原索引不变。

采集已进入 `complete_with_failures`，原进程 85837 已退出，会话 52567 正常结束。十题为 **1 个 `ok`、9 个 `execution_failed`**：7 个原 `ServiceError`、1 个 `InputOverflow`、1 个 `ProtocolError`。失败题全部保留，没有改容量、截断或额外重跑。成功题的原逻辑用量为 ANN 29/36、set-score 181/512、LLM 23/24、Reader 1/1，完整非 reranker 观测已保留。采集期间 wall clock 曾跨越长时间间隔；不把源服务的耗时用作 APC 性能基准。

终态后通过原 writer lock 门禁冻结到 `outputs/bt_prefix_cache_20261009/frozen_real_smoke10_20261009/`。抽样分母仍为 10；四题到达 reranker，六题未到达。**539 条物理 rerank attempt 全部 HTTP 200，共 854 个不同 query/完整集合评分样本**；539 条请求 hash、scorer 集合 ID/正文关联和完整索引响应通过校验，时间排序单调，原 scorer 的 `scored_sets` 合计恰为 854。批量分布为 236 个单样本、295 个双样本、4 个三样本、4 个四样本。原 scorer 内存结果缓存命中合计 2778、Transport rerank 响应缓存命中 0，均不计作 KV 命中。

代理报告 1078 次上游请求、2503122 个完整 prompt tokens、1708 个 completion tokens；这是原响应 usage，不能当作引擎 KV、新计算 token 或独立观测的计算指标。所有 539 条 trace 的最终 token IDs 仍为 null，tokenization/attention/模型身份未由实际引擎核验，结构复用分析与硬件收益也未完成。

十题私有资源包已导出到 `outputs/bt_prefix_cache_20261009/search_bundles_real_smoke10_20261009/`，包括失败题；文档可见范围、顺序、metadata、向量与源资源 hash 全部通过校验，不读 labels、不访问模型。数据集共有 589 题，当前的十题是明确 smoke 范围，不是全部可用题。固定 50 题主采集/主回放仍待执行，没有启动全量实验。

## 工具与安全边界

- `scripts/benchmark_bt_reranker_prefix_cache.py`：固定 10 题真实采集、独立门禁控制的固定 50 题采集、freeze、隔离 capture、token 前缀分析、单臂原生 token 回放、raw/归一化分数比较、原题可见范围的私有资源包导出、实际 Engine 冻结观测回放与队列/动作比较、合成协议输入。
- `serving/bt_prefix_cache_probe.py`：读取**已安装**版本/source/help/device，生成现有 vLLM argv 的隔离 dry-run 覆盖，以及加载原代理的观测包装器。观测器原样透传最终 payload/响应，不增添模型重试、不重写模板或评分函数。
- 输出目录 0700，工具生成文件 0600，目录必须新建，禁止恢复/覆盖旧实验。敏感正文与 tokens 只进入忽略的私有 `outputs/`。Authorization 不写日志；拒绝凭据 URL 和带 key 的启动 argv。
- 回放/观测只接受 loopback 隔离端口，排除生产 8002/18002；远端通过已授权 SSH tunnel 使用。启动 plan 不会被当作真实运行身份；回放另需 `runtime_verified`、健康检查和匹配 hash 的实际运行证据。
- 个性化输入采用部署已有的**单租户专用实例**，每个租户独立引擎、稳定 namespace。没有从 chat 文档照搬 `cache_salt`，也不声称当前完成接口已经验证 salt。跨租户共用实例不在本工具可执行范围内。
- 结构分析用完整 token 前文链、计算身份及 namespace 隔离，按实际 block 对齐并预留最后位置重算。只有开始前已经完成的请求可贡献命中；同批/在途/未来样本不能预热。理想无限容量量与实际引擎指标分开。
- 预算保持 ANN=36、set-score=512、LLM=24、Reader=1；旧 `logical_input_tokens_estimate`、`scored_sets`、`set_score_calls` 字段未改。上游 yes/no prompt 样本、完整集合评分样本、HTTP 次数另记。

## 可执行的本地与目标机步骤

以下为采证/dry-run/隔离测试命令，不是生产启用命令。目录名称应每次更新。

```bash
# 本机可运行，不发送模型请求。
.venv/bin/python serving/bt_prefix_cache_probe.py inspect --output outputs/apc_local_inspect_new
.venv/bin/python scripts/benchmark_bt_reranker_prefix_cache.py synthetic --output outputs/apc_synthetic_new

# 只有尚无真实完整 trace 时运行；本次十题已完成，不重复启动。
.venv/bin/python scripts/benchmark_bt_reranker_prefix_cache.py collect-smoke \
  --config configs/paired.example.json --output outputs/apc_real_smoke_new

# 先检查十题终态/失败日志并恢复可达服务；此命令没有自动启动。
# 固定前 50 题另行采集，包括 smoke 中失败的题，不按成功或收益选题。
.venv/bin/python scripts/benchmark_bt_reranker_prefix_cache.py collect-main \
  --config configs/paired.example.json --smoke outputs/apc_real_smoke_new \
  --output outputs/apc_real_main50_new

# 在有完整 DAG-BT 原始日志的目标机流式冻结，少于 50 题如实记录。
python scripts/benchmark_bt_reranker_prefix_cache.py freeze \
  --source /authorized/dagbt_run --questions 50 --output outputs/apc_frozen_new

# 在原引擎的环境/镜像内部执行 inspect，不能用 Mac 或另一 vLLM 的 help 替代。
python serving/bt_prefix_cache_probe.py inspect --output outputs/apc_installed_runtime_new
```

`freeze` 应指向同一真实 fusion 运行，排除其他方法。完全相同 journal 副本和物理 attempt 副本分别去重；相同 payload 的后续失败重试按真实时间归属原 stage。集合 ID 来自 scorer cache-miss/batch 事件，不解析不可信正文中的 passage 标头。原 manifest/task 中未进入模型的问题仍保留在抽样分母，不能只选成功题。

冻结读取期间持有 runner 已有 `writer.lock` 的共享锁；有活跃 writer 或非终态 progress 时拒绝并且不创建输出目录。没有该锁协议的外部日志必须先取得静止导出。`collect-main` 也拒绝活跃/未完成 smoke，要求十题全部终态，核对原配置文件、数据集 manifest 和固定前十题身份；失败题不删去。两次源采集分别有新 manifest，不恢复 smoke 输出。journal 无请求身份时记录 path fallback，dataset 无标准 attempt 路径时为 unknown；这些缺失在采集源补齐后才能正式验收。原 scorer 缓存累计命中单列，不能计为 KV；实际完成 trace 再与原成本日志核对。

目标机准备私有 `inventory.json`，由真实源码、模型文件和运行日志取证，不能用示例值填成 verified。所需字段见 `serving/bt_prefix_cache_probe.py:IDENTITY_FIELDS`：权重/tokenizer/template/代理 hash、adapter、dtype/KV dtype/量化、attention/RoPE、runner/pooling、引擎/Ascend/CANN/device、最大长度/block size、score contract/logprobs mode。明确无 adapter/量化/pooling 时记录 `none`，未知不能填 `none`。另需原 `engine_argv`、`causal_prefix_supported`、`private_single_tenant_instance`、`tenant_namespace_hash`、`label_token_ids` 和可用指标的实际名称/单位。

```bash
# 仅生成计划；两个计划除 APC、loopback 绑定外保持现有 argv。
python serving/bt_prefix_cache_probe.py plan --inventory /authorized/inventory.json \
  --help-file /authorized/installed_vllm_help.txt --mode off --port 28003 \
  --output outputs/apc_plan_off_new
python serving/bt_prefix_cache_probe.py plan --inventory /authorized/inventory.json \
  --help-file /authorized/installed_vllm_help.txt --mode on --port 28003 \
  --output outputs/apc_plan_on_new
```

若已安装 help 不能证明开/关参数，或实际评分 attention 不支持，plan 会拒绝；保留证据并报告能力阻塞，不能强改 pooling 或升级生产。独立升级需先比较升级后 off 与原服务，再比较同一升级环境 off/on；两个来源的收益不能混算。

在已有 Ascend 启动方式里使用同一镜像/环境/hotfix、同一授权空闲设备和新端口，**依次**运行测试臂。不能直接在本机照搬 NVIDIA 启动脚本，也不能同时让两个完整实例抢同一设备。启动 argv 的实际执行与设备分配依赖目标环境；当前没有执行。

```bash
# 仅当已有隔离后端运行且代理原源码 hash 已核验时，启动其独立观测进程。
python serving/bt_prefix_cache_probe.py observe-proxy \
  --inventory /authorized/inventory.json --proxy-source /authorized/original_reranker_proxy_api.py \
  --backend http://127.0.0.1:28003 --port 28004 \
  --frozen-trace outputs/apc_frozen_new/rerank_trace.jsonl --smoke-only --output outputs/apc_observer_smoke_new

# 另一个终端：完整集合请求原样进入观测代理；历史失败单列，不推断已执行。
python scripts/benchmark_bt_reranker_prefix_cache.py capture \
  --trace outputs/apc_frozen_new/rerank_trace.jsonl --proxy http://127.0.0.1:28004 \
  --smoke-only --output outputs/apc_capture_smoke_new
python scripts/benchmark_bt_reranker_prefix_cache.py analyze \
  --trace outputs/apc_observer_smoke_new/engine_trace.jsonl --block-size ACTUAL_BLOCK_SIZE \
  --output outputs/apc_prefix_structure_new
```

先检查这 10 题 smoke 的实际响应、失败和 token 身份。通过后，在新的 observer/capture 目录去掉两个 `--smoke-only`，冻结完整 50 题模型流（或全部可用题），保存为后续命令里的 `outputs/apc_observer_new/engine_trace.jsonl`。不能把 smoke 混入主回放计数。

主回放采用原批量与并发：当前已知 DAG-BT 是串行 HTTP、每批最多 4 个集合；如果实际源 trace 有 HTTP 重叠，串行 replay **不能**用于主结果，须先补充相同并发调度，记录这个未实现的条件。并发观测的文件可能按完成顺序落盘，分析前按 ordinal 排序仅整理日志，不能改变实际请求流。不同原始失败/重试按冻结规则单列，不能暗中重试补齐。

实际启动后为每臂保存健康检查和引擎配置/源码/日志证据，生成运行身份文件：与计划相同的计算/设备/内存/并行设置，并增加 `runtime_verified=true`、`health_verified=true`、`runtime_evidence_file` 及 SHA256。APC 状态必须是实测运行配置。回放的 `metrics` 映射只引用当前引擎观测过的 metric 名、counter/unit；缺失、多 replica 标签或 counter 回退均记 null，不能自动相加冒充设备指标。

```bash
python scripts/benchmark_bt_reranker_prefix_cache.py replay \
  --trace outputs/apc_observer_new/engine_trace.jsonl --backend http://127.0.0.1:28003 \
  --identity /authorized/observed_off.json --condition off --output outputs/apc_pair1_off_new

# 单独启动并核验 on 臂，完成相同模型/编译/算子预热；只清隔离实例的前缀缓存。
python scripts/benchmark_bt_reranker_prefix_cache.py replay \
  --trace outputs/apc_observer_new/engine_trace.jsonl --backend http://127.0.0.1:28003 \
  --identity /authorized/observed_on.json --condition cold --reset-isolated-prefix-cache \
  --output outputs/apc_pair1_cold_new
python scripts/benchmark_bt_reranker_prefix_cache.py replay \
  --trace outputs/apc_observer_new/engine_trace.jsonl --backend http://127.0.0.1:28003 \
  --identity /authorized/observed_on.json --condition warm --output outputs/apc_pair1_warm_new
python scripts/benchmark_bt_reranker_prefix_cache.py compare \
  --off outputs/apc_pair1_off_new --on outputs/apc_pair1_cold_new --output outputs/apc_pair1_compare_new
```

prefix reset 必须另有已安装版本的接口证据，运行身份 `prefix_reset_verified=true`。这个 reset 不清编译缓存。先在 off 上至少重复两次同输入，测原服务绝对/相对波动和批量/并发影响。数值标准预先冻结为现有 pointwise 默认 `atol=1e-6, rtol=1e-5`，不事后放宽。采样头的 raw logprob 与原归一化分数分开核验，同题排序和 exact tie 单列。

执行至少 5 组成对 off/cold，交替 off/on 与 on/off 顺序，并报告每组及分布；warm 只作补充。负对照用预先冻结的合成无长公共前缀流，在 off/on 都运行，不用它替代主流。动态 batching、并发、dtype、设备数量、KV 内存额度不混入 APC 对照。工具不自动承诺这 5 组已完成。

## 四项与实际轨迹

BT 源码用 `A=(PGe-PG)-(Pe-P)`、目标边际 `M=PGe-PG`，而记录字段 `context_marginal=PGe-Pe` 是另一项；工具明确区分。retain/speculate/pivot、epsilon、深度/状态限额与 tie-breaking 仍由冻结源码执行。

`search-replay` 的私有 bundle 包含原 question（仅 id/question）、config、method、原 attempt_dir、带完整 metadata 的 documents、规范 ids、vectors_path/SHA256、可选 reader_question。必须是该题原来可见的 corpus scope，不得从全库补未来记忆。`build-search-bundles` 复用原 prepare_resources/scope_resources，校验原算法、数据集与实际记录的 corpus/index 资源哈希；按原 scope 顺序复制该题向量和完整 provenance，失败题同样登记，不读评测 labels、不访问网络。它也要求源采集终态，当前不对活跃源导出。

导出的 PersonaMem bundle 带 `dataset` 与原有效 config。回放按原 worker 导入 pipeline、初始化配置与 PersonaMem Reader 适配，使 MCQ Reader 输入与原流程一致；不能只初始化通用 Engine 而漏掉这些步骤。现有日志中没有的新请求或未评分集合立即报分歧，无网络 fallback，无旧分数 fallback。正式预算按原记录的尝试次数记账。原流程失败题的完整成功轨迹仍不能验收：失败单列，不能把其已有评分样本当作已完成题。

```bash
# 仅在源十题或五十题采集已终态后执行；保留原失败题。
python scripts/benchmark_bt_reranker_prefix_cache.py build-search-bundles \
  --source outputs/apc_real_main50_new --output outputs/apc_search_bundles_new

python scripts/benchmark_bt_reranker_prefix_cache.py search-replay \
  --bundle /authorized/question_bundle.json --scores outputs/apc_pair1_off_new/results.jsonl \
  --label-ids '{"yes":9693,"no":2152}' --output outputs/apc_search_off_new
python scripts/benchmark_bt_reranker_prefix_cache.py search-replay \
  --bundle /authorized/question_bundle.json --scores outputs/apc_pair1_cold_new/results.jsonl \
  --label-ids '{"yes":9693,"no":2152}' --output outputs/apc_search_on_new
python scripts/benchmark_bt_reranker_prefix_cache.py compare-search \
  --off outputs/apc_search_off_new --on outputs/apc_search_on_new --output outputs/apc_search_compare_new
```

轨迹比较保留真实候选/队列顺序、请求身份、分支及逻辑计数，四原始分数/A/M 单列；只有阈值近旁动作或实际排序改变就报分歧。重复相同 payload 若观察到分数不同，聚合 score bank 会拒绝，须进一步按物理身份重放，不会偷偷用缓存抹平波动。已验证实际冻结 BT search 的协议 fixture、完整 Engine 的固定候选 fixture 只读观测回放，以及拒绝未知请求的路径；**没有真实模型的完整 DAG-BT 轨迹验收**。

## 报告、测试与回滚

输出保留 manifest/trace hash、逐 HTTP/样本比较、请求耗时均值/中位/P95、问题服务时间、吞吐/总墙钟及原生指标快照。批量 HTTP 耗时不能冒充单样本引擎 latency，暂记 null。设备时间、峰值内存、淘汰/OOM、实际新计算 token 若没有指标同样记 null。不得把 `full_input_tokens - 理想可复用量` 写成实际重算量。

`speedup=off_wall/on_wall`；`reranker_time_reduction=1-on_wall/off_wall`。单对比较不会自动升级成 VERIFIED：仍须真实命中、完整评分/动作/轨迹、至少 5 组稳定收益及无新增失败。现有按完整 tokens 计费 API 不会自动优惠；自部署仅能据实报告设备时间/吞吐容量，没有小时费率和可减计费时长不算货币节省。

最终本地审计见私有 `outputs/bt_prefix_cache_20261009/final_local_audit_20261009/`；较早的根目录 `verification.json` 和 continuation audit 是历史检查。最新相关测试 **175 passed in 5.05s**，其中新增测试 41 项；当时的三份 Python 代码 hash 与最终代码一致，之后只更新文档和私有采证产物，没有重复运行同一组测试。真实终态 trace 的 539 条请求与十题资源包已另行校验。这些仍是协议和源日志检查，不证明 KV 或加速。第一次 `.venv/bin/pytest` 入口未包含项目根，改为 `python -m pytest`。冻结原始文件 57 项与 BT snapshot 57 项均无差异，tracked/untracked 文件 whitespace 检查通过。合成测试包括空/singleton/pair/长前提、前缀与中间块隔离、反转 batch、输入身份、故障与统计边界、私有 scope/向量/provenance 导出和 PersonaMem 完整 Engine 协议回放；真实设备的近容量、生产并发、淘汰、重启和租户变更仍待执行。

```bash
.venv/bin/python -m pytest -q tests/test_bt_reranker_prefix_cache.py \
  tests/test_bridge.py tests/test_bridge_http.py \
  tests/test_bridge_upstream_dependency_scoring.py \
  tests/test_bridge_upstream_evidence_search.py \
  tests/test_model_runtime.py tests/test_config.py --tb=short
git diff --check
```

验证产物目录：`outputs/bt_prefix_cache_20261009/`，敏感产物不提交。生产未改变，回滚无需生产操作。测试退出只针对操作者本次启动的隔离 PID/container，保留日志；禁止重启/停止既有 reranker 或清生产全局 cache。

截至 12:09 UTC，LLM、embedding、reranker 的只读探测均不能取得有效服务响应：直连为 `RemoteDisconnected`，系统环境路由为 HTTP 502。原 source 包中另有 23 个失败的非 rerank 物理尝试（19 个 `RemoteDisconnected`、4 个 `ConnectionResetError`），模型是否实际执行未知；它们不能冒充有效模型样本。SSH 22 的 TCP 探测成功，但实际只读 SSH 在握手时被关闭；agent 身份为 0，没有匹配的授权别名或远程执行工具。没有重启服务、猜测凭据、使用历史生产 apply 脚本或开启新采集。

唯一必要的外部输入仍是**已授权隔离测试环境的 SSH 主机别名或执行入口**；无需发送凭据。本机不能执行 Ascend KV 测试。进入可执行环境后先探测设备、已安装源码/版本、实际部署身份与模型 API 可达性，再完成十题最终 token/服务 smoke、固定 50 题主流、baseline 波动、五组 off/cold、warm/negative、数值及动作/轨迹验收。当前缺证据的项目全部保持 unknown/null，目标尚未完成，生产没有启用命令。
