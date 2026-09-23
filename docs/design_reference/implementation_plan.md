# DAG v2 × Evidence BridgeTree：融合实施计划

日期：2026-09-23。状态：**研究与待实施设计**；本文不表示融合代码已经实现，也不报告新实验结果。

## 1. 建议与研究问题

建议融合，但以当前 **Evidence BridgeTree** 为底座，加入查询时的依赖解析和证据闭包选择，不串联两套完整系统。

核心问题：在固定上下文预算下，系统能否发现支持条件偏好的必要前提，区分检索导航与结论支持，保留足够完整且当前适用的证据，并在新条件出现时重算受影响的结论？

研究对象是用户历史中的偏好理解、偏好变化和个性化回答；不是用偏好对训练通用 RAG 的 DPO/RLHF。如果后续改为后一种定义，本文基线和任务需要调整。

**创新判断**：直接拼接树检索和 DAG 推理，新增贡献有限；将“前提发现、支持替代项、预算闭包、条件失效”做成可测试的联合机制，有形成方法贡献的潜力。能否达到顶会标准取决于与近邻方法的充分比较及独立实验，不宜预先给接收概率或宣称首创。

## 2. 两个输入方法与当前工程事实

### 2.1 DAG v2

来源：`dagv2_package_20260921.tar.gz`；本地解包位于 `/private/tmp/dagv2_review_20260923/dagv2_package_20260921/`。

- 仅根据问题规划 1–6 个节点，每个节点带 question、output_slot、answer_type、inputs。
- 按拓扑顺序，用已解析的父节点答案填充子问题，然后检索并生成节点答案和来源。
- 节点闭包包含本节点来源与已解析父节点的闭包；枚举小图上的节点集合，在文档数预算下保留较多完整闭包。
- 原包 reader 同时读取所选原文和已解析节点答案；部分中间答案的来源可能不在最终原文上下文内。
- 代码中 resolved 主要依赖非空答案和来源，不能等价为语义正确、条件有效或证据蕴涵成立。

可借鉴：显式输入绑定、拓扑求解、来源闭包。需要重做：偏好时间/情境、支持替代项、token 预算、语义状态，以及 reader 的中间答案泄漏风险。

### 2.2 BridgeTree 的三个版本

| 版本 | 实际机制 | 本方案地位 |
|---|---|---|
| 原始 BridgeTree | ANN、球面聚类、质心探测后落到真实父记忆、瓶颈可达性、路径条件特征、logdet 选择 | 保留低成本导航思想；质心不是证据 |
| Conditional Activation | 比较加入前提集合前后的候选边际分数变化 | 仅作为可选提案排序启发式 |
| 当前 Evidence BridgeTree | q-only 需求规划、多根预算搜索、原文证据映射、可删换的整组选取、缺口回查 | 融合的工程底座及最重要基线 |

当前代码已经有 `joint_inference`、支持/反对/部分支持、精确引用和需求覆盖；不能把“整组证据推理”重新包装为本次新增。

真正新增的是：执行节点输入绑定、显式支持超边、AND/OR 闭包约束、替代支持的选择与局部失效，以及与这些状态耦合的缺口搜索。

### 2.3 已有结果能说明什么

历史 partial audit 涉及 261 问、9 个 persona，并非完整 589 问的最终实验。共同成功的 257 问中，dense 答对 170、旧 activation 答对 159；19 个救回、30 个损害，净少 11 个正确。227/257 个 activation 任务只探索一个 target root。

其中 flashcards 案例的旧偏好原文已经位于 dense rank 4，却因加入后集合相关性分数下降而被舍弃。这支持“检查目标函数和证据选择”的动机，不证明新算法有效。当前 Evidence BridgeTree 文档记录 641 个离线测试通过；本轮未重新执行这些测试，也没有真实模型准确率验证。

## 3. 近两年文献如何约束创新叙事

检索/阅读截止 2026-09-23，重点覆盖 2025–2026 已能核实会议信息的相关工作；这是定向方法调研，不是穷尽式系统综述。主会与 Findings 分开标注。

| 文献 | 已有能力 | 对本方案的约束 |
|---|---|---|
| SETR [1]，ACL 2025 主会 | 信息需求、证据映射、集合选取 | 不能主张首次从 ranking 转为 set selection；其发表方法包含约 4 万教师样本蒸馏，不能称论文原方法免训练 |
| Associa [2]，Findings ACL 2025 | 事件图、PCST 子图、缺失线索预测与递归补全 | 图检索加缺口回查已经存在；需比较支持闭包是否带来独立收益 |
| TaciTree [3]，EMNLP 2025 主会 | 事实/persona 层次树、隐式偏好推理 | 树结构和隐式偏好不是新的组合 |
| PREMem [4]，Findings EMNLP 2025 | 写入前推理、跨会话关联、演化 | 偏好演化不是独有能力；可比较离线推理与查询时依赖的取舍 |
| RF-Mem [5]，ICLR 2026 | 熟悉度自适应检索、聚类与查询-质心混合、受限分支扩展 | BridgeTree 的聚类/分支检索须做强对照 |
| REMem [6]，ICLR 2026 | gist/fact 混合图、时间限定工具、多步检索 | 时间条件、多步图检索不能单列为创新 |
| MRAgent [7]，ICML 2026 | Cue–Tag–Content 图、由中间证据调整路由 | “主动重构而非静态召回”已有明确主张 |
| HyperMem [8]，ACL 2026 主会 | topic/episode/fact 超图、多粒度选择与溯源 | 高阶关联不新；需解释语义依赖与主题超边的差别 |
| S2Pref [9]，Findings ACL 2026 | 稳定/情境偏好、冲突澄清、序列识别 | 需要测试情境差异，不能把所有冲突当作覆盖旧偏好 |
| TSM [10]，Findings ACL 2026 | 事件时间、分层记忆、时间过滤/重排 | 不可仅凭“更近的会话”判断当前适用 |
| PURPLE [11]，ACL 2026 主会 | 依赖于生成效用的用户记录集合与顺序优化 | 不能泛称首次建模记忆之间的相互作用 |

应主张的差异：**在查询时构建可追踪的支持依赖，让同一依赖结构同时约束搜索缺口、最终上下文和更新失效**。这里的“可追踪”指来源与程序约束可核验，不是自然语言蕴涵的形式化证明。

## 4. 四类对象必须分离

1. `Requirement`：问题需要回答什么。原始顶层需求只由 q 得到，运行中不可为迎合候选而重写。
2. `ExecutionNode`：需要求解哪个子问题；输入依赖是预先计划，不代表这些父节点最后都成为必要证据。
3. `ProposalEdge`：通过什么线索发现候选。保留现有 `retrieval_proposal` 与 `dependency_claim=False`。
4. `SupportAlternative`：某结论真正用了哪些来源和父结论。单个替代项中的前提为 AND；同一结论的不同替代项为 OR。

发现图和支持图可以引用同一 memory_id，但边不能自动互相转换。正向支持图应为 DAG；冲突/更新关系存成旁路约束，不为维持 DAG 而删除真实冲突。

## 5. 数据结构草案

以下是接口设计，不是已有 API。

```python
Requirement(id, description, necessary, time_scope, version=0)
ExecutionNode(
    id, requirement_ids, question_template, answer_type,
    input_bindings, planned_parent_ids, refinement_of, version
)
EvidenceSpan(
    id, memory_id, start, end, exact_quote, speaker,
    event_time, mention_time, time_basis, entity_ids,
    scope, stance, evidence_kind, raw_text_hash
)
SupportAlternative(
    id, conclusion_id, used_parent_ids, used_parent_versions, source_span_ids,
    guard_span_ids, applicable_scope, semantic_status,
    structure_valid, invalidated_by, version
)
NodeState(
    id, answer, status, alternatives, unresolved_inputs,
    unresolved_guards, version
)
ConflictConstraint(
    id, target_alternative_ids, source_span_ids,
    conflict_type, time_scope, resolution_status
)
FusionRunState(
    requirements, execution_nodes, discovery_log, candidate_ids,
    evidence_spans, support_graph, conflicts, budget_ledger,
    visible_history_hash, query_time, config_hash
)
```

枚举约束：`status ∈ {unknown, partial, supported, ambiguous, invalidated}`。

- `unknown`：没有足够证据；不构造答案占位符供子节点当事实使用。
- `partial`：有相关片段，但前提、指代或条件未闭合。
- `supported`：模型判断支持成立，且程序校验通过、当前没有未解决的关键冲突。
- `ambiguous`：适用同一时间/情境的不同证据尚不能一致解释。
- `invalidated`：过去有支持，但支持替代项全部失效。仍有另一个有效替代项时，节点不进入此状态。

每个顶层需求在冻结规划时绑定一个或多个终端执行节点，并写明这些输出是联合需要还是可替代；不能任取一个贴上 requirement_id 的片段就算覆盖。细化节点只能服务于这些原有终端判据。覆盖判据由模型语义判断加结构检查共同给出，保留具体来源。

同一节点的 OR 替代项必须支持同一归一化结论和适用范围；若输出值相互冲突，应进入 ambiguous，不作为可随意替换的同义支持。父节点版本进入 used_parent_versions，父结论或情境变化后重新解析，而不是只替换 source ID。

`structure_valid` 与 `semantic_status` 必须分开输出。精确字符串存在、引用属于用户、父节点存在、无环和预算可行，都不能证明解释正确。

时间同时存事件时间和会话时间；未知时间为 null，不从排序位置推断。实体归一化必须有原文或显式绑定支持。助手建议不能冒充用户偏好；用户明确接受该建议时，引用接受语句及必要上下文。

## 6. 运行流程

### 6.1 规划与初始召回

1. 输入 q、截至 query_time 对该用户可见的历史、统一预算。
2. planner 仅看 q；生成不超过 6 个顶层需求及不超过 6 个初始执行节点。
3. schema 检查 ID 唯一、无环、绑定槽存在、无答案预填；不合格使用共享修复额度，不静默退化成另一方法。
4. 复用 dense + bridge 初始池。所有原始 memory_id 去重，保留发现路径审计。
5. 分配多根搜索队列：先保障未覆盖必要需求的轮转机会，再继续当前有希望的分支。

### 6.2 有条件的 Bridge 搜索

- 已 supported 的父结论及其原文前提可填入子问题；unknown/ambiguous 父结论不可直接绑定为真。
- 用原始 q、当前 target、真实前提记忆、冻结的信息需求构造提案；不能加入选项、gold 或 oracle memory IDs。
- ANN 候选来自本用户当前可见历史；路径回访与重复集合使用全局缓存。
- 沿用多根 quantum、公平调度、pair tests、受限 speculative paths；它们属于已有能力。
- 新增调度信号是“哪个必要需求缺哪项输入/条件”，不是单纯提高 reranker 分数。
- 可允许最多 2 个证据驱动细化节点，总执行节点不超过 8；必须记录 refinement_of、来源和版本，不能改变顶层问题。
- 收集所有发现候选，不只保留搜索途中得分最高的集合，再交由证据解析。

可选 activation proxy：

```text
R(S) = frozen_reranker(original_q, canonical_raw_memories(S))
A(e; G | P) = [R(P+G+e)-R(P+G)] - [R(P+e)-R(P)]
```

A 仅参与提案优先级消融。它依赖评分尺度，正值不代表因果依赖、实际答题收益或必要前提。默认依赖合法性不使用 A 阈值。

### 6.3 映射、求解与依赖编译

1. 复用 mapper：为候选原文抽取精确引文、偏好角色、时间/情境、支持或反对关系。
2. 校验 `[start:end] == exact_quote`，ID、speaker、可见历史、文本 hash 一致。
3. resolver 按拓扑顺序执行；输入 q、当前子问题、已支持的父节点及来源、映射片段。
4. 输出短结论、所用父节点、原文片段、条件、替代支持及 unresolved 信息。
5. 新边只能来自 resolver 显式报告的实际使用关系；计划中的父节点不自动进入证据闭包。
6. 编译器做无环检查、父节点可用性检查、支持替代项完整性检查。无效项保留审计，但不能贡献覆盖分数。
7. 对同一实体与适用情境的相反证据进行审计；若条件不足以分解冲突，保留 ambiguous。

建议 resolver 输出模板：

```json
{
  "node_id": "n3", "status": "partial", "answer": null,
  "alternatives": [{
    "id": "n3.a1", "used_parent_ids": ["n1"],
    "source_span_ids": ["s8"], "guard_span_ids": [],
    "semantic_status": "partial", "applicable_scope": "new_workshop"
  }],
  "unresolved_inputs": ["user_stated_motivation"],
  "unresolved_guards": ["current_applicability"]
}
```

不能强迫模型为每个问题输出支持。没有用户动机证据时，只能说“两个经历存在差异”，不能说“该差异导致改变”。

### 6.4 支持替代项、闭包与 token 预算

对一个已选替代项 a，其闭包为本项所有 source/guard 原文记忆，加上实际使用的父节点所选替代项闭包：

```text
C(a) = memories(sources(a) ∪ guards(a))
       ∪ union(C(selected_alternative(p)) for p in used_parents(a))
```

必须同时满足：所有 AND 父项可用、条件适用、来源有效、没有未解决的阻断冲突。一个节点只需一个足够的 OR 支持；不能把所有替代项历史无差别并入。

设计上每节点保留不超过 2 个候选支持替代项。保留与截断顺序固定，并记录被截断项；不能称为全局最优。8 个节点，每个选择“不选/支持一/支持二”，原始枚举最多 `3^8 = 6561`，设置 10000 个枚举状态硬上限。枚举不再次调用 LLM。

在这个**已发现、已编译、有界图**内，采用拓扑剪枝并调用真实 ContextPlan feasibility callback。目标按字典序：

1. 完整覆盖的 necessary 顶层需求数量；
2. 完整覆盖的 optional 顶层需求数量；
3. 更少的原文 token；
4. 稳定 ID 打破平局。

不奖励中间节点个数，避免把一个需求拆碎就提高目标值；不把未经校准的模型置信度直接作为加权收益。共享来源只记一次 token，多个结论可使用同一个证据片段。

这个有限选择器保证的是：在固定可行性规则和候选图内找到目标最优的赋值。它不保证召回完整、模型判定正确、世界中不存在例外，也不继承旧 logdet 的证书。固定 PSD logdet 的边际递减性质不能直接证明 AND 前提互补目标。

若完整闭包均放不下：先按同一目标选择其他可行完整闭包，再确定性保留与未解需求有关的 partial 原文作为补充；每次补充重新检查实际 token。partial 不计为已覆盖，补充不能挤掉已经选定的完整支持。不得逐段截断必要原文却继续宣称闭包成立。

### 6.5 缺口回查与失效更新

- 缺口记录包括 requirement/node ID、缺失槽/条件、已有候选 hash、提案及本轮结果。
- 最多 2 次保留 ANN 调用，包含在全局 36 次内；无新增证据或预算耗尽则停止。
- 针对明确未解决的条件检索反证；有限回查无法证明例外不存在，日志中保留 audit 范围。
- 新反证先作用于匹配实体、时间和情境的支持替代项。
- 若某替代项失效，尝试同节点其他可用替代项；之后再沿实际支持边重评估后代。
- 只有所有支持都失败时才使该结论 invalidated；不能一次把所有发现树后代删除。
- 新旧事件属于不同场景时允许共存，不做 unconditional latest-wins。
- 图版本、query_time 和历史 hash 进入缓存键，避免复用条件已经改变的结论。

存在尚未解决的相反证据时，不能只选支持一侧并把另一侧丢弃来制造确定性。拟纳入 reader 的 disputed 片段须与其反证组成受保护的原文组并一并计费；放不下时不保留单边片段，该需求仍未覆盖。审计未覆盖的候选不默认为没有反证。

### 6.6 最终 reader

默认只传选中 raw memories、原始问题和 benchmark 允许的公开答案选项。planner、mapper、resolver、selector 不接触选项或 gold；feasibility callback 仅返回可行性、token 数、预算、hash。

内部节点答案只用于检索绑定与支持选择，默认不传 reader，保持与现有 Evidence BridgeTree 相同的信息接口。reader 得到“证据不足时不补造偏好理由”的统一指令；所有基线使用相同指令。

单独消融 `reader_chain=true` 时，所有中间结论必须附完整已选来源并计入相同总上下文预算；严禁传来源不在最终上下文内的结论，严禁提示模型优先照抄终端节点答案。

基础设施错误返回 typed error；证据不足返回 unknown/partial。不能把 API/解析失败记为语义零分而隐去，也不能静默使用 dense 答案补齐。

## 7. 合成例子：导航路径与证明路径不同

这是机制示例，不是数据集实测案例。

问题：“为什么我现在愿意重新学画画？”

| 记忆 | 用户原文（合成） | 作用 |
|---|---|---|
| m1 | “以前的课每一步都要照着画，我觉得很受限制，所以后来不去了。” | 过去经历和不喜欢的原因 |
| m2 | “小林向我推荐了河岸画室。” | 发现画室实体的导航线索 |
| m3 | “河岸画室允许自己选题材，也能尝试不同画法。” | 当前条件 |
| m4 | “河岸画室的这种安排正合我意，所以我想重新报名学画。” | 用户明确说出的当前动机；“这种安排”需由 m3 绑定 |
| m5 | “以前我嫌照着画太受限；如今河岸画室能自由创作，我因此愿意重新学。” | 可独立支持完整解释的替代总结 |

对“前后对比”的一种支持是 m1 AND m3 AND m4；若 m5 已发现且语义足够，可选择 m5 替代这组来源。m2 可以帮助找到 m3，却在实体和动机已被其他原文明示时不必进入最终上下文。

若后来同一当前画室的明确原文 m6 说“自由创作班已取消，我暂时不报名”，需要重评当前意愿；旧动机仍是历史事实。若 m6 只是助手猜测，就不能覆盖用户意愿。若 m5 指的是另一时间/画室，也不是有效替代。

## 8. 文件级实施映射

| 文件 | 动作 | 验收重点 |
|---|---|---|
| `src/bridgetree/fusion_types.py` | 新增类型、schema、状态机 | 四类边/对象互不混用，时间缺失可表达 |
| `src/bridgetree/fusion_config.py` | 新增显式融合配置 | 配置校验、全局预算、版本与 hash |
| `src/bridgetree/fusion_planner.py` | 复用需求协议，增加执行 DAG | q-only、无环、顶层需求冻结、细化可追踪 |
| `src/bridgetree/fusion_resolver.py` | 原文到支持替代项及条件审计 | 不以非空答案判成功，AND/OR、speaker/time 检查 |
| `src/bridgetree/fusion_search.py` | 对接现有多根搜索 | 节点缺口、公平调度、全局缓存和 ledger |
| `src/bridgetree/fusion_selection.py` | 依赖闭包枚举、局部失效传播 | 有界精确选择、真实 token 可行性、替代项保留 |
| `src/bridgetree/evidence_selection.py` | 提取可复用 mapper/引用校验接口 | 旧 evidence_bridge 行为和默认配置不变 |
| `src/bridgetree/dependency_retrieval.py` | 增加显式子问题/缺口入参或薄 adapter | 保持 ProposalEdge 非语义依赖语义 |
| `src/bridgetree/dependency_experiment.py` | 新增 `fusion_bridge_dag` 分支 | 统一 reader、options 隔离、结果 schema 与 resume |
| `src/bridgetree/fusion_diagnostics.py` | 记录闭包、替代项、预算、失效原因 | 不以模型标签冒充人工真值 |
| `configs/fusion_bridge_dag.yaml` | 新实验配置 | 不覆盖现有 Evidence BridgeTree 配置 |
| `tests/test_fusion_*.py` | 新增机制测试与 stub 集成测试 | 覆盖后文清单，保持现有测试通过 |

具体复用点：现有 `evidence_selection.py` 的 PLAN/MAP 协议、引用校验与 `_validate_selection`；`dependency_retrieval.py` 的 `set_information_needs`、`conditional_probe_text`、`propose`、`retrieve_missing`；`dependency_experiment.py` 中 `evidence_bridge` 的执行与 ContextPlan 接口。函数签名实施时复核，不按本文行号硬编码。

不要直接 monkeypatch 解包的 DAG 包。移植的是依赖语义和闭包思想，新模块应由当前项目的模型客户端、预算器、缓存和审计统一管理。

## 9. 初始配置与预算账本

以下是待验证初值，不是实验选出的最优参数。

```yaml
method: fusion_bridge_dag
planner:
  max_requirements: 6
  max_initial_nodes: 6
  max_refinement_nodes: 2
support:
  max_alternatives_per_node: 2
  max_enumeration_states: 10000
  preserve_unknown_time: true
search:
  ann_calls_total: 36
  reserved_gap_ann_calls: 2
  proxy_mode: none
  set_score_calls_if_proxy_enabled: 512
  coverage_roots: 6
  exploration_roots: 4
  quantum_new_sets: 24
  quantum_measurements: 8
  max_consecutive_quanta: 2
  max_pivots: 4
  max_pivot_depth: 2
  max_speculative_depth: 2
  max_speculative_states: 8
  max_pairs_per_state: 6
reasoning:
  total_llm_calls: 24
  reserved_final_audit_calls: 1
  json_repair_calls_total: 2
  input_token_budget: 16384
  map_batch_token_budget: 6144
  output_max_tokens: 4096
reader:
  raw_only: true
  calls: 1
  context_budget: inherit_benchmark
```

`proxy_mode=none` 下不使用 set-score gain 排序，用必要需求缺口轮转及 ANN 排名安排提案；依赖 set-score 的量子/测试参数此时禁用。启用 proxy 的消融复用原搜索完整策略，并计所有 set-score 调用。不得在 none 模式内部悄悄运行免费 reranker。

24 次是全部 planning/mapping/resolving/audit/repair 的共享 LLM 上限，不是每节点 24 次。一个可能分配是 plan 1、map 最多 12、resolve 最多 8、repair 最多 2、final audit 1。实际调用由池分配，批处理 resolver 可处理多个就绪节点；出现大候选池或额外细化时，应减少/合批而不是突破总量。最终审计预留不可借空；重试和 JSON 修复计入实际成本。

统一 ledger 至少记录 ANN、embedding query、pointwise/set rerank、各阶段 LLM 输入/输出 token、reader token、缓存命中、耗时、失败、离线索引成本与每用户摊销。ANN 数相同不等于成本相同；应同时报告固定 reader 预算与完整成本-性能曲线。

预算耗尽后输出最佳已校验可行集合和显式 partial 状态，不把未处理候选当成无用。执行节点、候选上限和替代项截断均需记录。

## 10. 缓存、恢复与身份冻结

- 运行身份包括代码 hash、完整配置、prompt 版本、模型与 tokenizer 标识、语料/可见历史 hash、query_time 和 query ID。
- cache key 包含原始 q、节点版本、父支持版本、来源 hash、情境/时间约束；不同用户空间严格隔离。
- 原始 memory 的角色和顺序有固定 canonicalization，顺序敏感模型不能共享不等价输入的分数。
- 首版采用任务边界恢复；不要宣称跨进程可恢复搜索 frontier，除非完整序列化队列、图、ledger、cache 和 RNG 状态并测试。
- 中断任务保留 incomplete，不计入完成结果；同一 query 最终结果按权威 ledger 去重。
- 新方案需要一个独立 method ID；禁止让旧结果文件在同名 method 下混入新策略。

## 11. 必须通过的机制测试

1. AND：删除必要前提后，不得仍将完整需求标记 covered。
2. OR：移除一个替代项后，另一完整项仍可支持结论。
3. 导航分离：m2 仅为发现路径时，不强制进入支持闭包。
4. 计划分离：未实际使用的 planned parent 不自动成为证据。
5. 闭包 token：共享 memory 只计一次；真实 tokenizer 加系统提示后超预算要拒绝。
6. 无环：新支持边形成环时返回明确 schema error，不能随意删边装作成功。
7. 角色：助手建议不能单独支撑用户偏好；引用和 offset 必须一致。
8. 时间：未知事件时间不补造；不同情境的偏好并存；旧聊天提及未来事件不能仅按聊天时间覆盖。
9. 更新：只使匹配的支持替代项失效；后代重评可切换另一替代项。
10. unknown：父节点未解决时禁止作为已知事实填入子问题。
11. 预算：36 次 ANN 与 24 次全阶段 LLM 硬上限，修复/重试/回查不得绕过。
12. 截断：枚举上限、候选映射未完成、没有完整闭包均显式记录 partial。
13. 泄漏：options/gold 不进入规划/映射/求解；raw-only reader 看不到节点答案。
14. 细化：不超过 2 个、有引用、有 refinement_of，顶层需求 hash 不变。
15. 失败：API/解析故障与语义 unknown 分开；不产生静默 fallback。
16. 小图选择器：用手工穷举验证同一有界图上的目标最优及稳定 tie-break。
17. stub 端到端：固定返回值复现 AND、OR、更新、预算停止，追踪 reader 的确切输入。
18. 回归：现有 evidence_bridge 的行为和输出字段兼容；运行项目要求的现有测试。

这些测试只验证实现与协议，不证明自然语言依赖判断正确；需另做人工/独立评审。

## 12. 实验设计：先证明依赖有用，再证明搜索有用

### 12.1 固定候选池：隔离选择贡献

用同一冻结候选池比较 flat Evidence Bridge selection 与 dependency closure selection。共同 reader、上下文预算、候选原文、模型版本、输出长度。先观察条件证据保留、前提缺失率、错误解释率与准确率，而不是让新方案因多检索而自然占优。

可用全部方法的候选并集构造池，但应明确它是分析用 oracle-free 共池，不是某个在线方法的成本结果。不得以 gold 支持记忆构造主实验候选池。

### 12.2 端到端 2×2 消融

| | 平面证据选择 | 依赖闭包选择 |
|---|---|---|
| dense 候选发现 | dense + flat | dense + dependency |
| Bridge 候选发现 | 当前 Evidence BridgeTree（校准成本） | 完整融合 |

同一候选预算策略和 reader；若 resolver 比 flat selector 花更多 LLM，报告真实开销，并另做等总 token/成本约束对照。主张互补性时计算交互差值 `(bridge_dep - bridge_flat) - (dense_dep - dense_flat)`，报告区间；主效应也应独立分析。

必要消融：去掉条件/时间审计；去掉替代支持；导航路径强制并入闭包；去掉局部失效；开启/关闭 activation；raw-only 与带完整来源的 chain reader；固定 DAG 与允许 2 次细化。每项回答一个具体机制问题，避免只展示全量模型胜出。

### 12.3 外部基线与数据

- 方法近邻优先：RF-Mem、Associa、MRAgent，以及 SETR 思路的当前 Evidence BridgeTree；时间子集补 REMem，层次/隐式偏好补 TaciTree 或 HyperMem。
- 官方代码可复现就用官方实现；只能重实现时明确 adapter、不同 embedding/reader/索引成本。不可把本地 prompt 简化版标成论文原系统。
- PersonaMem [13] 为当前项目主要任务；按官方问题清单核验当前 32k 设置的 589 问，不假定所有问题都需要桥接。
- PrefEval [12] 测偏好遵循/隐式推断；S2Pref [9] 测情境变化与冲突；LongMemEval [14] 可作多会话时间/更新/拒答的外部检验。
- 已分析的 261 问属于诊断/开发证据，不能再宣称独立确认集。按 persona 划分并冻结开发与确认集合；若无未见 persona，应新增外部数据或明确其非独立性。
- 主指标：官方答题指标；辅指标：完整前提保留率、适用条件准确率、矛盾未解时的过度确定率、无支持解释率、预算与延迟。
- PersonaMem 的标准答案不等于必要前提标注。人工构造独立、盲评的证据支持集与桥接必要性标签；至少双标并报告分歧与裁决。
- 针对桥接必要性设计受控诊断：移除单个前提、改变情境、替换同实体更新时间。答案变化本身不证明因果依赖；要同时判断证据是否仍支持答案。
- 统计以 persona 聚类 bootstrap/配对分析，报告置信区间和失败率；9 个 persona 的结果不宜装成几百独立样本的高置信结论。

### 12.4 支持与否证条件

支持方向的证据：同池同预算下闭包方法提高必要前提完整性且减少错误解释；端到端 Bridge 在需要隐含前提的子集有额外收益；更新实验能正确保留替代项并重评受影响结论；增益在独立集合和合理成本区间保留。

应降级或否定的情况：只在 chain 注入后变好；收益完全来自更多调用；同池下依赖约束没有收益；导航记忆占满预算导致性能下降；人工发现依赖边错误频繁；仅在已查看的失败案例有效。

若只剩准确率提升而解释依赖不可信，应定位为工程改进；若依赖更正确却不提高答题，报告机制收益与适用任务边界，不能宣称普遍更强。

## 13. 分阶段交付与验收

### P0：冻结基线与设计

- 保存当前文件/配置 hash、已有结果来源、独立实验清单。
- 固定 Requirement/EvidenceSpan/SupportAlternative schema 与提示词。
- 验收：每个新增机制都能对照当前已有能力；无代码改动前已有明确实验矩阵。

### P1：仅做离线依赖选择器

- 固定候选池和人工小图，完成类型、闭包枚举、token 检查与失效传播。
- 验收：AND/OR/预算/替代项测试通过；不接远程模型也能完整追踪结果。

### P2：接入模型解析与原始 reader

- 接 planner/mapper/resolver，保留现有基线和新 method ID。
- 验收：角色/引用/泄漏/失败分类/共享预算测试；小规模人工审查支持依赖质量。

### P3：接入 Bridge 缺口调度

- 增加绑定子问题、需求公平调度、受限细化和反证回查。
- 验收：共池消融已可运行；没有重复预算器或隐性额外调用。

### P4：独立实验与写作

- 完成 2×2、关键近邻、成本曲线和预先冻结的反例诊断。
- 验收：公开/归档配置与 run identity、失败任务、不确定性、审计例子；结论与证据强度匹配。

最终 checklist：两类图严格区分；原文闭包可核验；条件/替代项能表达；统一预算；reader 输入公平；新旧方法隔离；独立确认结果；未将模型判断包装成数学证明。

## 14. 阅读依据与参考文献

以下链接是论文/会议原始来源；会议名由官方论文页核对。详细方法对比集中在 [1]–[10]，另读 [7] §3–4、[6] §3 和 [1] §3.2–3.3。基准论文用于任务/评估定位。

1. **SETR**. Shifting from Ranking to Set Selection for Retrieval Augmented Generation. ACL 2025 主会. https://aclanthology.org/2025.acl-long.861/ 。借鉴需求-证据-选集；原方法有蒸馏训练。
2. **Associa**. Bridging Intuitive Associations and Deliberate Recall: Empowering LLM Personal Assistant with Graph-Structured Long-term Memory. Findings ACL 2025. https://aclanthology.org/2025.findings-acl.901/ 。重点 §4 的图、PCST 和 missing-clue 递归补全。
3. **TaciTree / ImplexConv**. Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning. EMNLP 2025 主会. https://aclanthology.org/2025.emnlp-main.580/ 。重点层次记忆、隐式支持与反对情境。
4. **PREMem**. Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue. Findings EMNLP 2025. https://aclanthology.org/2025.findings-emnlp.1204/ 。重点 §3 的原子记忆及跨会话演化关系。
5. **RF-Mem**. Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval. ICLR 2026. https://iclr.cc/virtual/2026/poster/10008269 ；全文 https://arxiv.org/html/2603.09250 。重点 §2.3 聚类、混合查询与受限扩展。
6. **REMem**. Reasoning with Episodic Memory in Language Agents. ICLR 2026. https://iclr.cc/virtual/2026/poster/10008195 ；全文 https://arxiv.org/html/2602.13530v3 。§3 保留潜在矛盾历史、gist/fact 混合图及时间检索工具。
7. **MRAgent**. Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents. ICML 2026. https://icml.cc/virtual/2026/poster/60697 ；全文 https://arxiv.org/html/2606.06036v1 。§3 Cue–Tag–Content；§4.1–4.2 中间证据驱动路由及上下文累积。
8. **HyperMem**. HyperMem: Hypergraph Memory for Long-Term Conversations. ACL 2026 主会. https://aclanthology.org/2026.acl-long.1627/ 。重点多粒度超图、混合召回/重排及来源追踪。
9. **S2Pref**. Beyond Static Profiles: Capturing the Fluidity of User Preferences in Diverse Scenarios. Findings ACL 2026. https://aclanthology.org/2026.findings-acl.1033/ 。约 1 万用户条目与 15 万对话，不能混称为 1 万对话。
10. **TSM**. Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents. Findings ACL 2026. https://aclanthology.org/2026.findings-acl.1496/ 。重点事件时间与记忆层次/更新。
11. **PURPLE**. ACL 2026 主会. https://aclanthology.org/2026.acl-long.1467/ 。用户记录集合、顺序、生成效用与 contextual bandit；作为集合相互作用的补充近邻。
12. **PrefEval**. ICLR 2025 oral. https://arxiv.org/abs/2502.09597 。约 3000 偏好/查询对、20 个主题，显式与隐式偏好。
13. **PersonaMem**. COLM 2025. https://arxiv.org/abs/2504.14225 。动态画像与上下文适配的个性化响应选择。
14. **LongMemEval**. ICLR 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html 。500 问，时间、多会话、知识更新与拒答。

内部来源：`docs/evidence_bridge_implementation.md`、`docs/evidence_bridge_review.md`、`docs/evidence_bridge_revision_report.md`、`docs/conditional_activation_goal.md`、`outputs/partial_audit_20260923/REPORT.md`、`reports/2026-09-23-mechanism-research/REPORT.md`；以及当前 `src/bridgetree/` 和原 DAG 包。内部实验与公开论文指标不可直接横向比较。
