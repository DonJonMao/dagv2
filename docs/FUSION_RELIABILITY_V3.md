# DAG + BridgeTree：个人历史与选集复核 v3

日期：2026-09-30。融合诊断版本为 `dagbt_fusion_reliability_v3`。本轮将上游 BT `16809bd..c4b04c9` 的个人历史任务提示、原文复核、覆盖标注恢复和诊断适配到 DAG；上游交付记录 HEAD 为 `93a56db`。这是冻结模型的检索、推理与评测，模型参数不更新。

v3 默认启用 `selection_review=true`、`raw_memory_review=true`、`allow_unassessed_coverage=true`。它改变模型可见信息和最终选集，不属于纯接口修复；必须新开运行身份。原始 DAG 文件及冻结搜索 vendor 不因本轮适配覆盖，`original` 臂保留既有方法。前一版本协议见 [v2 记录](FUSION_RELIABILITY_V2.md)。

## 现在的执行流程

1. DAG 从固定原问题规划可执行节点，冻结终端需求。对 PersonaMem，规划、检索、mapping、节点求解与最终选集复核只看到当前用户问题，不看到候选答案或 gold。
2. 用原问题做一次 dense 检索，按排名逐条检查完整 reader 输入可行性，得到 `baseline_ids`。这次检索计入同一题的 36 次 ANN 额度，不是额外赠送的候选。基线候选加入正常候选池，继续参与映射；固定候选池实验改用固定池顺序，不额外检索。
3. 各就绪子问题执行既有 BT 搜索、来源映射、节点求解、缺口回查、条件审计和失效传播。检索导航来源不会自动成为逻辑支持前提。
4. dependency 模式用 `select_support` 得到 DAG 来源闭包提案，保留原有必要需求覆盖、可行性和最小原文成本目标；不能完全覆盖时仍有既有 partial 补充。该提案交给新的 `FinalSelector`，不直接作为最终答案上下文。flat 模式使用相同复核器，但没有 DAG 闭包提案和 dependency 专属覆盖要求。
5. 复核输入先容纳完整 dense 基线原文及权威来源 metadata，再容纳预算内完整映射记录与 DAG 支持路线。模型可以保留、移除或替换提案中的文档，也可以选入没有任何有效 mapping 的可见原文；不要求最小文档数，也不强制并入全部 dense 原文。
6. 代码校验实际选集的可见性、文档数量、完整 reader 预算和适用硬约束。选定后重新计算选集内能够成立的 DAG 支持证书。最终 reader 接收所选完整原文，默认不注入模型生成的中间答案链。

`FinalSelector` 的自然语言 coverage 与程序重新计算的结构证书是两层结果。raw-only 文档可以供 reader 阅读，但不会凭空产生 span、support route 或 `covered`。dependency 模式的 `covered` 还要求对应节点的完整 eligible route 在实际选集内；同节点、可见且属于所选文档的证据引用才合法。被争议来源文档仍须按保护组整组保留；`fusion_navigation_closure` 仍执行完整导航来源闭包约束。最终 reader 使用完整的 20 文档上限选集，5/10 文档结果另行重算证书，不能将子集视为已经通过最终覆盖证明。

## P1：问题理解与 PersonaMem 协议

`dagbt/prompts.py` 补充个人历史任务语义：偏好、经历、原因、约束和历史阶段本身可以支持个性化回答；推荐类问题不要求记忆中已经出现完整推荐答案；不把未询问的活动名、时间、地址或日程自动设为必要事实。一般任务仍使用 DAG steps/schema，PersonaMem planner 另有明确的个人历史说明。跨领域相关性仍是模型预测，不因提示改进而被证明。

融合 PersonaMem 经 `dagbt/runner.py` 将 `user_question` 交给 Engine，将含全部公开选项的原始题目作为独立 `reader_question` 交给最终 reader。当前用户问题是规划与证据阶段的唯一问题文本；gold 只进入评分阶段。记忆范围继续按 persona/shared context 与 exclusive cutoff 隔离。

`original` 臂保留先前的“当前问题加公开选项”输入协议，不随融合臂改写。因此双臂系统比较同时包含规划/证据协议、检索选择、reader 答案链等差异，不能解释成仅有 BT 一个因素。v3 内的融合消融共享 query-only 协议。四套数据和任务总量保持 3,589 题、7,178 个默认双臂方法任务。

## P2：原文复核与消融

独立 raw channel 的完整记录为 `doc_id`、`passage` 和 `metadata`；角色依赖权威 `source_segments`，不从文本里的“User:”等字样猜测。原文观察顺序不等于事件日期或因果顺序。预算不足时省略整条记录，记录省略 ID，不截断 JSON 或源文本。只有实际显示的 raw 或 ledger 文档 ID 可选，已经发现但未显示的候选不获得选择资格。

| 设置/方法 | 差异与解释 |
|---|---|
| 默认 `fusion` | 原问题 dense 基线、DAG 闭包提案、完整选集模型复核、允许 raw-only 文档 |
| `fusion_no_raw_review` | 相同 dense 基线构造、候选来源、预算和复核流程，仅关闭独立 raw 输入；已有映射的文档仍可通过 ledger 进入选集 |
| `fusion_strict_coverage` | 与默认相同，但覆盖标注修复耗尽后不允许 `unassessed` 继续 |
| 配置 `selection_review=false` | 回到旧选集机制，并关闭新复核调用预留；这是机制回归开关，也改变 baseline 阶段，不是隔离 raw channel 的消融 |

搜索存在有限调用与模型随机性；相同候选来源/预算设计不等于不同真实运行的最终搜索池必然逐字相同。要做严格同池选择对照，应使用冻结候选池并记录身份，不把固定池作为在线召回成绩。

## P3/P4：覆盖恢复和调用预留

全局 selected IDs/header、可见性、完整 reader 容量与闭包硬约束必须合法。尚无合法 header 时，可以在限额内修订整个选集；合法 header 固定后，覆盖修复不能变更选集、reason 或 conflicts。已通过的 coverage 行保留，只发送待修需求及其所选文档内的相关 ledger、允许引用的别名和选集内路线状态，不重复发送全部 raw 原文。

覆盖类型只允许保守降低：引用隐式 assessment，或由独立 partial 前提联合支持时，错误的 `explicit` 声明可改为 `inference`；不能把 partial、跨节点引用、选集外引用或伪造 ID 改写成合法支持。没有同节点有效 mapping 的 raw 文档可以被选中，但不能用其文档 ID 冒充 evidence alias。

仅当合法选集已经固定、错误确实限于覆盖标注且有界恢复耗尽时，默认允许剩余行成为 `status=unassessed`、`kind=null`、`validation_complete=false`。这不代表 covered 或检索缺口，也不会再据此触发缺口检索。服务错误、拒绝、明确输出截断、不完整/非法全局 JSON，以及破坏固定 header 的响应仍失败；不能被未评估覆盖吞掉。`fusion_strict_coverage` 可关闭该继续执行策略。

默认仍为每题 24 次预算内 LLM 物理尝试、6 次全局 JSON 修复、每操作最多 2 次修复。新复核提前预留 3 次调用和 2 次修复；映射/非选择修复与传输重试不能耗掉这些预留。24 是 DAG 的保守 attempt 口径：物理 HTTP 重试和成功缓存重放按已有规则扣额，不等同于独立 BT 的 24 次逻辑调用；逻辑调用、物理请求、缓存及 reader 分开报告。

上下文保持 DAG 的 16,384，reader 输出预留 1,024；没有照搬独立 BT 的 8,192 reader 输入预算。证据输出预留 4,096、margin 256，选择初始输入另留 384 的修订空间，当前选择视图限额为 **11,640**。最终 reader 可行性仍按实际序列化模板及输出预留重新计量。证据估算 `regex_or_utf8_bytes_div3_v2` 和 reader 的既有 BT regex 估算都是估算，不能视作服务 tokenizer 的精确值或上界。

## P5：观察什么结果

`diagnostics.semantic_evidence` 记录 baseline/candidate/fully-mapped/eligible/raw-visible/selected IDs、原文省略、候选去向、baseline 保留比例、覆盖完成状态和恢复过程。按实际选集分别标记 `empty_context`、`mapped_only`、`raw_only`、`mixed`，旧数据或不一致状态为 `unknown`；有映射与完整映射均不等于语义相关或充分。

原 `reliability` 的 normal/truncated/partially_mapped 分组继续保留，不能用 normal 代替“证据充分”。`summary.json` 的方法级 `semantic_evidence` 另列证据状态、coverage complete/unassessed/unknown、交叉计数和成本。成功与失败分开，当前权威结果不重复计题；旧失败费用仍进入 all-attempt 成本，usage 缺失仍记未知。

`bash scripts/run_v3.sh diagnostics` 只读当前权威产物与失败快照，不加载评分标签、不调用模型，不因查看进度创建新运行。最终准确率、失败率、共同成功比较和全部尝试成本仍需一起看，不能从不同问题组成的 cohort 分数推导因果收益。

## 已执行的来源核验与实际导出回放

上游核验对象是 BT 的 `dist/evidence-bridge-v3-server/bridge-tree-evidence-v3.tar.gz`，**不是本仓库的新部署包**。SHA256 为 `d5826ed88506a01c56eaa8ee3df96ca9f5aaa96c8fec915797af7d5ebefc2bb1`，与配套 sha256 和 manifest 一致。流式核对 222 个普通文件，全部与当时 BT 工作区逐字相同，无链接成员；文档中的 220 是首包历史数。BT 核心源码未在审计中改动。

本仓库已执行：

```bash
.venv/bin/python scripts/replay_bt_v3.py \
  --run-dir /path/to/evidence_v2_20260930_logs/run \
  --output-dir outputs/diagnostics/bt_v3_raw_view_replay_final_20260930
```

源为 BT 实际 v2 导出。73 个 normal 任务中，选择全部 10 个零映射/空选集案例，冻结原始 query、requirements、dense 选集和完整来源 metadata，构造未知 DAG 节点后经过当前 `FinalSelector.prepare_view()`。**模型调用 0、新答案 0**。报告只含 ID、hash、计数与边界说明，路径为 `outputs/diagnostics/bt_v3_raw_view_replay_final_20260930/replay_report.json`。

| 已观测项 | 结果 |
|---|---|
| 符合零映射条件的真实案例 | 10 个全部回放 |
| exported dense 原文 | 120 条，91 条进入 raw view、29 条整条预算省略 |
| 全部 baseline 可见的案例 | 0/10，不宣称在 DAG 复现上游 73/73 全可见 |
| 保留原文和 metadata | 全部完整一致 |
| 新选择 payload | 10/10 在当前输入预算内 |
| 关闭 raw channel 的对照 | 保持相同 baseline，raw 可见条数为 0 |
| 图书题 `40d94e80-9557-48dc-9101-1cc6c12486a9` | 12 条保留 10 条，关键 m00003/m00031 可见；11206/11640 tokens |

此回放仅验证真实来源在新 raw 通道中的可见性。旧 BT requirements 被表示成独立未知节点，baseline 是旧 BT 导出值，没有重新检索 DAG baseline、转换原 BT 映射为 DAG 支持、调用 selector、生成新答案或证明准确率收益。非零映射案例不在该脚本范围内。上游输入额度 16,128 与此处 11,640 不同，因此不能直接搬用上游完整可见率。

对应独立测试 `tests/test_replay_bt_v3.py` 已执行 **5 passed**；该脚本和测试 compileall 通过。测试使用虚构文本，不把真实个人原文复制进仓库。上游三道真实开发题 dense 3/3、BT v3 2/3，亦不是本仓库 DAG v3 的成绩；关键原文到达 reader 仍不保证答对。

## 已执行的有界真实服务探测

独立 planner 探测通过：一次逻辑调用、一次物理 HTTP 请求，无重试；使用固定虚构问题与正式 schema/validator。报告为 `outputs/verification/reliability_v3/live_probe/preflight_calls/planner-1790751300880543000/report.json`。这只证明该次 planner 请求通过，不证明后续阶段兼容或答案质量。

随后的一次真实 Engine 虚构小样本烟测**失败**。报告为 `outputs/verification/reliability_v3/live_smoke/smoke_report.json`，输入为四条人工记忆，没有加载公共数据集或 gold。planner 和实际 embedding 成功，但首个 mapping 请求 `map/2` 达到该次烟测配置的 20 秒超时，底层为 `TimeoutError`，最终记录 `ServiceError`。只允许一次尝试，`max_identical_attempts=1`，没有为取得成功而重试。

本次语料 embedding 独立请求 1 次；任务内共 4 次逻辑/物理请求：LLM 2 次（planner 成功、mapping 超时）和 query embedding 2 次，无缓存、无 HTTP 重试。没有到达 selector、reader 或 reranker，也没有新答案。已观测的两个 LLM 输入预算检查均通过，不代表超时 mapping 返回了合法内容。非 reader 请求中未发现公开选项，但 reader 根本未执行，因此不能把报告中的 reader 原文/选项检查未通过解读成已经证明泄漏，也不能宣称 reader 路径验证完成。

该烟测使用缩小的 ANN=8、set-score=32、feedback=0 额度；LLM=24、context=16,384、证据输出=4,096、reader 输出=1,024。它不是默认全量运行。运行起止 source hash 同为 `6e7ffec57854dfff686034e703f1f2c0066ad3e8321983c229102f262a4537ab`，记录中没有运行期间源码变动；该 hash 不能自动视为最终交付包身份。**当前没有真实服务完整 map→select→reader 路径通过的证据**，其兼容性与语义收益仍需后续完整验证。

## P1–P5 核对矩阵

| 要求 | 当前源文件/路径 | 已有证据与限制 |
|---|---|---|
| P1 任务理解与问题隔离 | `dagbt/prompts.py`、`engine.py`、`runner.py` | 已核对个人历史提示及 PersonaMem query/reader 分路；worker 回归入口在 `tests/test_runner_v3_diagnostics.py`。语义改善须真实配对实验 |
| P2 原文独立复核 | `dagbt/final_selection.py`、`evidence_views.py`、`engine.py` | 实际 10 案例回放及 5 项独立测试；91/120 可见，不能宣称全部 dense 原文可见或已被模型选中 |
| P3 合法选集与未评估覆盖 | `dagbt/final_selection.py`、`support.py` | 源码保留 ID/预算/同节点引用/选集中路线约束；未评估覆盖不形成支持证书。`tests/test_final_selection_v3.py` 的 42 项回归通过，包含正式 Engine 失败落盘；本次真实烟测未到达该阶段 |
| P4 紧凑修复、保留状态与额度预留 | `final_selection.py`、`row_recovery.py`、`reasoning.py`、`transport.py`、`config.py` | 有独立恢复与重试预算回归入口 `tests/test_recovery_v3.py`；不能把请求可发送等同于真实模型修复成功 |
| P5 诊断、成本和运维 | `dagbt/runner.py`、`scripts/run_v3.sh` | 证据状态与协议可靠性分开；回归入口 `tests/test_runner_v3_diagnostics.py`。全套 529 项测试通过；真实完整运行尚未通过；包核验记录见下文 |

## 最终本地回归与交付检查

2026-09-30，最终生产代码与测试树执行 `.venv/bin/python -m pytest -q tests`：**529 passed, 6 warnings in 43.99s**，退出码 0。日志 `outputs/verification/reliability_v3/pytest_final.txt`。6 个 NumPy matmul 数值警告来自既有 resources 回归；不宣称无 warning。先前无目录限制的 pytest 将 `outputs/verification/reliability_v2/extracted/` 历史解包副本也纳入收集，产生 23 个同名模块冲突；保留历史产物后改为明确收集正式 `tests/`，不是跳过失败测试。

- `compileall -q dagbt scripts tests`、两份运维脚本的 `bash -n`、`git diff --check` 均通过；没有全仓库 Ruff 清零的声明。
- 原始 manifest 的 57 个文件和冻结 BT vendor 的 57 个文件分别逐个 SHA256 核对通过。本轮不覆盖原方法与 vendor；BT 工作区既有未跟踪文件未改动。
- `bash scripts/run_v3.sh preflight --offline-preflight` 退出 0：四套数据分别 1000/1000/1000/589 题，原始索引完整性通过，Qwen3 派生索引仍按正式启动流程构建。此检查没有调用模型。
- `run_v3.sh help` 与不存在 run 的 `diagnostics` 实际执行通过；后者报告 `no_manifest`、`labels_read=false`，没有创建运行。运维协调器和本地 HTTP 流水线测试均包含于完整 suite。
- 新包构建使用最终已提交文件与 `original_manifest.json` 的并集，包括三个被 Git 忽略但离线完整性所需的旧 `.npy`；逐成员哈希、文件类型和实际解包预检均由构包脚本验证。最终包 SHA256 见包旁 `.sha256`，逐成员与提交身份见本地 `outputs/verification/reliability_v3/archive_verification.json`。旧 v2 包保留。

最终选择注释恢复还区分“需求行未验证”和“额外 coverage 容器行非法”：不为非法额外行创造虚假 requirement；已通过行冻结保留，必要时整体覆盖状态仍记未完成。选择尚未产生合法 header 的故障保留 `unknown`，不能由空占位列表推导为空上下文。异常快照刷新逻辑调用与预算尝试，包含最后失败请求。

真实 smoke 的冻结源码身份与最终交付不同：之后完成了 coverage 容器异常处理、失败计数和 unknown 诊断修复，并显式写入默认 v3 配置；最终树由上述全套回归验证，未再次调用真实服务。因此本次仍不能宣称真实完整路径成功、全量准确率或效果提升。旧 v2 的 437 项与上游 BT 的 981 项不替代本轮 529 项结果。

## 新包与启动入口

v3 部署文件名为 `dagv2_bt_deploy_20260930_reliability_v3.tar.gz`。使用 `scripts/run_v3.sh` 管理 `start/status/stop/resume/diagnostics/preflight`，默认输出 `outputs/paired_full_reliability_v3`。解压到新项目目录，保留旧 tar、项目和输出，不用 v3 恢复 v2 运行身份。派生索引仍需通过既有模型/数据身份检查后才能复用。

安装、小规模试跑和全量启动命令见 [SERVER_RUN.md](SERVER_RUN.md)。模型仍为 DeepSeek-V4-Flash、Qwen3-Embedding-8B、Qwen3-Reranker-8B；原三套 QA 与 PersonaMem 默认范围不变。
