# DAG v2 × BridgeTree
## 面向条件偏好的证据依赖检索
2026-09-23 · 方法调研与融合设计 · 尚未实施或验证性能
> 建议融合：由 BridgeTree 发现隐含前提，由查询时 DAG 明确支持关系，再在 token 预算内选择完整、当前适用的原文证据。研究价值在三者的联动，不在简单叠加树与图。
### 两种方法各自解决什么
| 方法 | 已有机制 | 能带来的能力 / 需要补足 |
| DAG v2 | 问题拆成有输入绑定的节点；父答案填入子问题；拓扑求解；选择来源闭包 | 适合组织多步依赖；尚缺偏好情境、替代支持和严格原文预算 |
| 当前 Evidence BridgeTree | 多根桥接搜索；需求规划；精确引文映射；整组证据选取；缺口回查 | 适合发现间接记忆；已有联合推断，但缺显式依赖绑定和闭包约束 |
### 融合后要回答的研究问题
在相同上下文预算下，系统能否找齐“这项偏好为什么在当前条件下成立”的必要证据，并在条件变化时只重评受影响的结论？
这里的 preference RAG 指基于用户历史理解偏好并回答问题，不是用 DPO/RLHF 训练通用 RAG。
### 创新性判断
直接把两套流水线接起来，贡献偏弱。若能证明“依赖指导搜索 + 预算保持完整支持 + 条件更新重评”优于现有集合选择和主动记忆检索，则具有形成方法论文的潜力。不能预先宣称首创或性能提升。
本报告以 2025–2026 已核实的相关会议论文为主；主会与 Findings 分列。文献细节与可执行修改清单见配套 implementation_plan.md。
<!-- page -->
# 01 / 文献留下的空间
## 已有工作覆盖了哪些部分
| 工作 / 发表信息 | 关键做法 | 对融合创新的约束 |
| SETR [1] · ACL 2025 | 需求、证据映射、整组选取；蒸馏训练 | 整组选择已有；我们的冻结提示改造不等于原论文方法 |
| Associa [2] · Findings ACL 2025 | 事件图、PCST 子图、缺失线索递归补全 | 图检索和缺口补全已有很近对手 |
| TaciTree [3] · EMNLP 2025 | 层次事实/persona 树、隐式偏好 | 树结构与隐式偏好本身不新 |
| PREMem [4] · Findings EMNLP 2025 | 存储前推理、跨会话记忆演化 | 需比较查询时依赖与离线推理 |
| RF-Mem [5] · ICLR 2026 | 聚类、自适应回忆、受限分支扩展 | 仅桥接搜索不足以支撑创新 |
| REMem [6] · ICLR 2026 | 时间化 gist/fact 图、工具式多步检索 | 多步与时间条件已有成熟做法 |
| MRAgent [7] · ICML 2026 | 中间证据驱动 Cue–Tag–Content 路由 | 主动重构与动态路由不能称首创 |
| HyperMem [8] · ACL 2026 | 多层超图与来源追踪 | 应区分主题关联与结论的必要支持 |
S2Pref [9] 强调稳定偏好与情境偏好的差别；TSM [10] 区分事件时间和聊天时间；PURPLE [11] 已研究用户记录集合的相互作用与生成效用。
> 可争取的贡献：让同一份查询时支持依赖，同时约束“继续搜什么”“最终必须留什么”和“新条件使哪项支持失效”。这仍是待实验验证的差异化主张。
<!-- page -->
# 02 / 融合后的工作流
## 一条流水线，两种不同的图
[[workflow]]
### 发现路径不等于支持关系
发现图记录“经由哪条线索找到了记忆”；支持 DAG 记录“这项结论实际依赖哪些前提”。只有后者决定最终必须保留哪些原文。路径上的每一条记忆不必都进入 reader。
### 闭环如何停止
必要需求已有可行的完整支持，或剩余预算不足，或回查不再带来新证据时停止。未知条件保持 unknown；新反证先使具体支持替代项失效，有其他有效支持时不应删除整个结论。
reader 默认只看选中的原文、原始问题及任务允许的选项。内部节点答案用于检索绑定与选集，暂不注入 reader，以便判断收益是否来自更好的证据。
<!-- page -->
# 03 / 一个具体例子
## 为什么我现在愿意重新学画画？
以下记忆为合成示例，用来说明机制，不是实验结果。
| 记忆 | 用户原文（简化） | 在本题中的角色 |
| m1 | “以前每一步都要照着画，我很受限制，后来不去了。” | 旧经历及不喜欢的原因 |
| m2 | “小林向我推荐了河岸画室。” | 导航线索，帮助找到画室 |
| m3 | “河岸画室能自己选题材，也能尝试不同画法。” | 当前条件 |
| m4 | “河岸画室的这种安排正合我意，所以想重新报名。” | 明确动机；“这种安排”由 m3 绑定 |
| m5 | “以前嫌照着画受限；现在能自由创作，我因此愿意重学。” | 若语境明确，可独立构成替代支持 |
[[example]]
### 该保留什么
若选择支持 A，m1、m3、m4 共同支持前后对比，不能只留最像当前问题的 m4。若 m5 已完整表达同一实体/时间下的理由，可选更短的支持 B，无须强制合并 A 与 B。
m2 帮助发现 m3；当实体关联已有直接证据时，m2 可不占最终上下文预算。若指代消解仍依赖 m2，它就应作为真实前提保留。
### 条件改变时
新增“自由创作班取消了，我暂时不报名”应触发当前意愿的重评。旧动机仍可作为历史事实保留。若新文本只是助手猜测，不能据此覆盖用户偏好。
<!-- page -->
# 04 / 核心机制与边界
## 从“选相关记忆”到“选完整支持”
### 1. 用 AND/OR 表达证据互补与替代
同一个支持替代项内，实际使用的父结论和原文条件都是 AND；同一结论有多个可用替代项时，它们是 OR。条件和冲突单独审计；正向支持保持无环，不能为了 DAG 结构而删除冲突事实。
### 2. 按原文闭包计算真实成本
C(a) = 本项来源与条件的原文记忆 ∪ 实际使用的父项闭包。
共享记忆只计一次 token。预算检查使用最终 reader 的真实模板与 tokenizer，不能只数文档个数，也不能切掉必要前提后仍称完整支持。
### 3. 在有界候选图内选择
初始最多 6 个执行节点，最多 2 个细化；每节点最多 2 个支持替代项。加“不选”共至多 3^8 = 6561 种原始赋值，可拓扑剪枝枚举。先最大化必要顶层需求覆盖，再看可选需求覆盖，再优先更短原文；不奖励中间节点数量。
### 4. 保留未知并正确处理失效
状态区分 unknown、partial、supported、ambiguous、invalidated。引用准确、角色正确、图无环是程序可核验的；“这些原文是否足以支持解释”仍是模型判断，需要人工检查。一个支持失败时先找其他支持，再重评真正受影响的后代。
> “精确”仅指固定、已编译、有界图上的组合选择，不保证语义正确或召回完整。旧 logdet 证书不能迁移为 AND 前提互补的理论保证；activation 分数也只适合作提案启发式。
事件时间未知时保留未知；更晚的会话不自动覆盖旧偏好。有限反证回查无法证明不存在例外。无明确动机证据时，不把先后发生写成因果解释。
<!-- page -->
# 05 / 对现有工程怎么改
## 在 Evidence BridgeTree 上增加薄层
| 部位 | 复用什么 | 新增什么 |
| 规划 | q-only 顶层信息需求 | 子问题、输入槽、计划父节点；细化有版本与来源 |
| 搜索 | ANN + bridge、多根调度、缓存、真实记忆提案 | 面向未解前提的队列与有条件查询；统一预算 |
| 证据 | exact quote、speaker、来源 ID、支持/反对映射 | 时间/情境、实际使用父项、AND/OR 替代支持 |
| 选择 | ContextPlan 可行性、原文集合、审计输出 | 闭包枚举、共享成本、支持项失效与后代重评 |
| 执行 | 模型客户端、冻结 run identity、原始 reader | 独立 fusion_bridge_dag 方法分支与诊断 |
### 首版预算建议（待验证）
每题 ANN 共 36 次，其中保留 2 次缺口回查；所有规划、映射、求解、审计与修复共用 24 次 LLM 调用上限，reader 另计 1 次。不是每节点重新领取预算。
默认不使用 activation 分数判定依赖。可选 proxy 消融最多 512 次集合评分并记录实际成本；无 proxy 时采用缺口轮转及 ANN 排序。所有方法同时报告总 token、模型调用、索引摊销、延迟与失败。
### 分四步实施
P1：先做小图与固定候选池上的闭包选择器。P2：接入模型解析，仍用 raw-only reader。P3：将未解前提反馈给 Bridge 搜索。P4：冻结配置，跑独立实验与强基线。
新增模块包括 fusion_types、planner、resolver、search、selection、config、diagnostics；现有方法继续作为对照。详细接口、配置、错误处理和测试在配套 MD 中。
<!-- page -->
# 06 / 怎样证明融合值得做
## 分离搜索、选择与答案注入的影响
| 实验 | 平面证据选择 | 依赖闭包选择 |
| dense 发现 | dense + flat | dense + dependency |
| Bridge 发现 | 当前 Evidence BridgeTree | 完整融合方法 |
先在完全相同候选池上比较两种选择器；再跑上面的端到端 2×2。固定 reader、上下文与输出预算；额外 LLM 成本单独报告，并补成本约束下的比较。
### 指标与强对照
主指标采用 PersonaMem / PrefEval 的官方任务指标；补充必要前提保留率、条件适用性、无支持解释率、冲突下过度确定率和成本。S2Pref 测情境变化；LongMemEval 测时间、更新与拒答 [9,12–14]。
近邻优先比较 RF-Mem、Associa、MRAgent 和当前 Evidence BridgeTree。论文原实现、自己重实现和提示词改造须分开标注。
### 关键消融
分别关闭条件审计、替代支持、失效传播；把导航路径强制并入闭包；开关 activation；最后单独比较 raw-only 与来源完整且同预算的 chain reader。
### 已有证据与尚未证明之处
历史 257 个共同成功任务中，旧 activation 答对 159、dense 答对 170；19 次救回、30 次损害。它说明相关性增益不可靠地等同于证据效用，不代表当前 Evidence BridgeTree 或融合方案的性能。
已查看的 261 问用于开发诊断，不再当独立确认集。必要证据需单独盲评标注；按 persona 聚类报告不确定性，不能把高度相关的问题当作独立样本。
> 若收益只来自更多调用或中间答案注入，或同池下依赖选择没有收益，就不能支持核心创新主张。当前结论是“值得做可证伪的小规模验证”，还不是“融合已经更好”。
<!-- page -->
# 07 / 参考文献与阅读定位
## 方法近邻 · 2025–2026
[1] SETR. Shifting from Ranking to Set Selection for Retrieval Augmented Generation. ACL 2025 主会。重点 §3.2–3.3：需求、映射、选择及蒸馏；不能将其整体称为免训练方法。
https://aclanthology.org/2025.acl-long.861/
[2] Associa. Bridging Intuitive Associations and Deliberate Recall: Empowering LLM Personal Assistant with Graph-Structured Long-term Memory. Findings ACL 2025。重点 §4：事件图、PCST、missing-clue 递归补全。
https://aclanthology.org/2025.findings-acl.901/
[3] TaciTree / ImplexConv. Toward Multi-Session Personalized Conversation: A Large-Scale Dataset and Hierarchical Tree Framework for Implicit Reasoning. EMNLP 2025 主会。重点层次树与隐式偏好情境。
https://aclanthology.org/2025.emnlp-main.580/
[4] PREMem. Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue. Findings EMNLP 2025。重点 §3：原子记忆与跨会话推理。
https://aclanthology.org/2025.findings-emnlp.1204/
[5] RF-Mem. Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval. ICLR 2026。重点 §2.3：聚类、查询-质心混合及分支扩展。
https://iclr.cc/virtual/2026/poster/10008269
全文：https://arxiv.org/html/2603.09250
[6] REMem. Reasoning with Episodic Memory in Language Agents. ICLR 2026。重点 §3：时间化 gist/fact 图、保留潜在矛盾历史、多步工具推理。
https://iclr.cc/virtual/2026/poster/10008195
全文：https://arxiv.org/html/2602.13530v3
[7] MRAgent. Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents. ICML 2026。重点 §3–4：Cue–Tag–Content 与中间证据驱动路由。
https://icml.cc/virtual/2026/poster/60697
全文：https://arxiv.org/html/2606.06036v1
<!-- page -->
# 08 / 补充来源与可复核范围
## 条件偏好、时间与评估
[8] HyperMem: Hypergraph Memory for Long-Term Conversations. ACL 2026 主会。重点 topic/episode/fact 超图、分层检索与溯源。
https://aclanthology.org/2026.acl-long.1627/
[9] S2Pref. Beyond Static Profiles: Capturing the Fluidity of User Preferences in Diverse Scenarios. Findings ACL 2026。稳定/情境偏好、冲突澄清及序列识别。
https://aclanthology.org/2026.findings-acl.1033/
[10] TSM. Beyond Dialogue Time: Temporal Semantic Memory for Personalized LLM Agents. Findings ACL 2026。重点事件时间、分层记忆及时间过滤/重排。
https://aclanthology.org/2026.findings-acl.1496/
[11] PURPLE. ACL 2026 主会。补充近邻：用户记录集合的顺序与交互、生成效用及 contextual bandit。
https://aclanthology.org/2026.acl-long.1467/
[12] PrefEval. ICLR 2025 oral。显式/隐式偏好理解与遵循，约 3000 偏好/查询对。
https://arxiv.org/abs/2502.09597
[13] PersonaMem. COLM 2025。动态用户画像与个性化响应选择；本文工程沿用其当前项目设置。
https://arxiv.org/abs/2504.14225
[14] LongMemEval. ICLR 2025。500 问，覆盖多会话、时间、知识更新与拒答。
https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html
### 内部材料与范围
DAG v2：2026-09-21 压缩包及源码。BridgeTree：当前 evidence_selection / evidence_search / dependency_retrieval / dependency_experiment，以及 evidence_bridge 文档与 partial_audit 报告。
本次完成论文阅读、源码核对和设计文档；未修改算法代码，未运行远程模型实验。day2/source_manifest.json 记录本地来源摘要。所有实验收益、参数最优性和投稿价值仍需验证。
