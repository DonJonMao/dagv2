# DAG-Resolve：分歧触发的绑定条件检索

**研究路线决议 | 2026-10-02**

建议以 dagv2-origin 为底座，敲定这一条改进方向：**出现竞争中间答案时，用有限的新增检索核实“候选是否满足原问题的绑定关系与限定条件”，再提交一个答案继续原 DAG。**

论文要回答的核心问题是：相同候选、模型、语料、判读器和额外预算下，检索尚未得到证实的绑定条件，是否比检索两条下游属性、通用核验或普通补缺更有效地纠正错误绑定？

:::flow
原 DAG 桥节点：提出最多两个有出处的候选，含 primary
发现原问绑定条件缺证／候选支持冲突：触发一次补查
候选名 + 未核实关系 + 原问题范围：各检索一次
联合判读原文：supported / contradicted / unknown
提交一个答案及其来源，继续原 DAG；未决按固定策略兜底
:::

## 首版边界

每题最多一个候选提议节点、一次补查动作；最多增加两次 embedding 检索和一次 LLM 判读。候选提议替代普通节点调用，输出容量相应增加。首版沿用原 selector 和 Reader，不引入深链 beam、推测执行、回滚或训练型策略。

多候选、延迟提交、原文核验、原问锚定和局部回滚均有先例。可检验的增量是**竞争绑定出现之后的查询目标选择**。本报告确认它值得开展机制实验，尚未证明首创，也没有新性能结果。

> 调研依据：2024-2026 年正式顶会论文，重点核对 BeamAggR、ReAgent、CIRAG、DIVA 和 Beam Retrieval 的算法及相关附录；源码审查与仅论文审查分别标明。执行方案已通过文献与预算两项独立复核。

<!-- page -->

# 1. 最接近的先例决定了贡献边界

| 先例 | 已有机制 | 本路线必须验证的增量 |
|---|---|---|
| [1] BeamAggR，ACL 2024 | 节点 top-k 答案；候选组合填槽；多源检索；概率边际聚合。默认 k=2，已有纠正早期错误的案例。 | 把新增查询用于尚缺证据的原问绑定条件，而非继续展开各候选的下游问题。 |
| [2] ReAgent，EMNLP 2025 | 局部／全局回滚；关键断言 challenge；追加证据。附录已有四假设预枚举，也有下游之前的实体条件核验。 | 根据已经观察到的候选分歧选择绑定补查目标，并在固定预算下优于通用 challenge。 |
| [3] CIRAG，ACL 2026 | 根据原问、历史查询和核心 triples 找信息缺口，生成 targeted query；保留多条 plausible chains。 | 显式针对当前竞争绑定的未核实条件，比普通 history-conditioned gap query 找到更有用的证据。 |
| [4] DIVA，NAACL 2025 | 时间、主体等歧义维度；条件化伪问；检索／重排与质量验证；覆盖多个合法解读。 | 处理有明确条件的问题中的中间错绑定；缺证后再次检索，而非将真歧义强压为单解。 |
| [5] Beam 检索，NAACL 2024 | 多条 passage 序列假设；端到端训练 encoder 与分类头；跨跳 beam 扩展。 | 在冻结模型和全语料检索下选择候选绑定查询，研究对象与 passage 路径排序不同。 |

## 对 BeamAggR 的准确定位

BeamAggR 的 Algorithm 1 已对依赖候选做 Cartesian product，将答案填入问题，再执行多源 reasoning；§3.3 对最终答案做边际聚合。Figure 8 的 Cologne／Darmstadt 例子已说明多答案和后续检索可以纠正早期错误。

本路线比较的是“下一次查什么”。BeamAggR 所核对的算法未显式定义：先识别某个原问绑定谓词仍缺证，再为现存竞争实体补查该谓词。此处是方法接口差异，是否带来收益必须实验验证。

> [1] §3.1-3.3、Algorithm 1、§5.3、Appendix A.2/A.4-A.6；[2] §4、Appendix C 及多跳案例；[3] §3.3-3.4、Appendix A.2；[4] §3.2-3.3、Algorithm 1；[5] §3 与公开 retriever 模块。完整书目信息见第 9-10 页。

<!-- page -->

# 2. 收窄到查询动作，而不是机制拼盘

## ReAgent 的重合必须正面承认

ReAgent 的 Appendix C Algorithm 2 在开始时枚举四个假设并逐一检查；Appendix D.4 Figure 7（PDF 第 23 页，印刷 4089）还在 fight-song 子问前核验大学是否同时满足 Lawrence 与 Kansas City 校区条件。因此，“先保留候选”“先核验实体条件再查下游”都不是本方案可独占的贡献。后者是论文图示的预先分解策略，不代表已审计到源码实际检索行为。

本路线将问题进一步限定为：**已观察到两个有出处的竞争绑定，且 primary 的必要条件缺证时，应该把下一笔检索预算投向哪一类查询？** 固定候选前端和判读器后，与下游查询、通用 verification、CIRAG 风格 gap query 正面比较。

## 其他相关工作也压缩了可声称的新颖性

| 已有研究线 | 正式先例 | 对本路线的约束 |
|---|---|---|
| 支持／反驳与对比解释 | [6] RAFTS；[7] C-RAG | 三值标签和比较证据属于工具，不能单列为贡献。C-RAG 与 Corrective RAG 是不同论文。 |
| 图驱动的对比问题 | [8] KG-CRAFT | 对比问题和竞争论证已有先例；仍需检验开放语料上的候选绑定补查。 |
| 信息增益与证据选择 | [9] InfoGain-RAG | “信息增益”名称不构成创新；首版采用固定查询规则，不增加效用学习。 |
| proposition 路径与动态规划 | [10] PropRAG；[11] D²Plan | 图搜索、更多路径、重规划已有工作；暂不扩展为全面图框架。 |

## 可以形成论文的最窄主张

从自然错误中识别一种失效：下游属性证据很多，却不能证明上游实体满足原问条件。提出一个低预算的绑定补查动作，并证明该动作在相同候选和裁决规则下，具有更高的绑定证据命中率、净修复率及全任务答案收益。

若普通 gap query 已能取得相同证据并达到相同效果，路线应降为工程改进；若收益只来自候选 prompt 或 Reader 改动，也不能归因于查询策略。本次定向检索与深读不构成穷尽查新结论。

> 研究问题覆盖年份／版本、实体身份、关系对象与角色条件；不局限于年份，也不把模型中间错绑定等同于用户问题真实歧义。

<!-- page -->

# 3. 一个例子：该查导演资格，还是出生地？

**虚构问题：1998 年电影《归途》的导演出生在哪座城市？**

原计划有两个节点：director = 谁执导该电影；birthplace = {director} 出生在哪座城市。初始原文只说“陈海执导《归途》”和“林舟执导《归途》”，均未说明版本年份。

| 路线 | 下一步检索 | 能否核实原问绑定 |
|---|---|---|
| origin 单值路径 | 若选陈海，查询“陈海出生在哪里” | 出生地证据不能证明陈海是 1998 版导演。 |
| 多候选下游展开 | 分别查陈海、林舟的出生地 | 两条链都可能完整；链更长或来源更多不保证导演绑定正确。 |
| 本路线 Binding 动作 | “陈海 1998 《归途》 导演”；“林舟 1998 《归途》 导演” | 直接补查两候选是否满足尚未证实的导演／版本关系。 |

新增原文若明确写“1998 版《归途》由林舟独立执导”，判读采用林舟，再沿用原 DAG 检索其出生地；若原文说明他出生苏州，最终回答苏州。

## 证据状态针对完整命题

命题 P(h)：候选 h 满足原问要求的“执导 1998 版《归途》”绑定。三值判断必须有原文依据。

| 原文 | 对 P(陈海) 的状态 |
|---|---|
| 陈海出生于武汉。 | unknown：只证明出生地。 |
| 陈海执导了 1986 版《归途》。 | unknown：不能仅因另一个版本就排除他也执导 1998 版。 |
| 1998 版由林舟独立执导，并非陈海。 | contradicted：排除或唯一性内容必须确实出现在原文。 |

检索不到资料、低相似度、另一候选来源更多，都不是反证。若原文支持共同导演，应识别并记录真多值，不解释为互斥；首版沿 primary 路径，不解决多答案覆盖。origin 提示已包含原问题并提醒父答案可能错；它也可能自行纠正，本例仅说明结构性风险。

<!-- page -->

# 4. 首版协议：一次局部补查，固定兜底

## 触发与查询

每题按 plan 选择第一个有后代的桥节点做一次 proposal，替代普通调用，不用 gold 选节点。输出 ordinary 含义的 primary 与完整来源，以及最多两个候选（含 primary）的逐字引用；不能回连合法原问条件或没有第二候选则继续 primary，不在后续节点重试。

只有当两个不同候选有出处，primary 的必要条件缺证或完整支持冲突，且有新查询与足够预算时，才触发补查。条件必须引用原问题的精确文本；字符串存在不等于语义关联正确，后者仍须审计。仅替代项缺证、primary 已充分支持时，不自动触发。

固定模板：**候选原文名称 + 未核实绑定关系 + 原问实体／作品／机构 + 原问范围条件**。条件按年份／版本、身份、关系的固定优先级及原问顺序选取。不另用模型写 query，不猜别名。两候选各 dense top50；判读面板最多 20 篇，先保留至多 4 篇锚定及 primary 完整引用文档，剩余位置对称分配新 hits，不足用初始 hits 补齐。锚定／primary 来源超过保留额度则不触发。去重和完整文档容量规则共享。

## 判读后如何提交

| 结果 | 固定 MVP 行为 |
|---|---|
| 两项均 supported，或原文明示合法多值 | 优先处理：退出修正动作，保留 primary 路径并记录事件，不强迫唯一。 |
| 仅 primary 获充分支持；或双方 unknown／冲突未清 | 保留 primary；分别记录 supported 或 unknown/conflict，未决不称已核实。 |
| 替代项 supported，primary unknown／contradicted | 采用替代项；primary unknown 时标 unexcluded_competitor，不声称证明唯一。 |
| primary contradicted，没有 supported 替代 | 清空答案和来源，成为 origin 式 unresolved；后续沿用原 ground 的未解父问题接地。 |

unknown 不删除候选。这里的“延迟”只到本次有限判读结束，预算用尽后按上表提交。保留 primary 是操作兜底，primary 来自共享 proposal，并非 exact origin 的原始输出。

逐条原文三值判断；同 scope 的支持／反驳并存记 conflict，按未决兜底，不数来源投票。无效引用或标签降为 unknown。候选及裁决先写 trace，不改变 Reader 模板；unknown 仍可能被原 Reader 强先验放大，这是保留原接口的已知限制。

<!-- page -->

# 5. 与 dagv2-origin 的修改对照和预算

另建研究入口和包装模块，保留冻结原文件。最小插入点在当前节点 proposal 之后、node_state 之前；判读结束只提交一个 answer/source 集合，后代尚未执行，首版无须版本失效或回滚。

| 原代码入口 | origin 行为 | 研究入口的最小变化 |
|---|---|---|
| package/core.py:61 | answer + sources 布尔数组 | 仅一个合格桥节点增加候选／条件字段；保留 primary 的完整普通来源语义。 |
| frozen_flow_v6.py:34 | 检索、单次作答、直接 node_state | 在当前节点提交前加入两次补查与一次联合判读；统一执行采用规则。 |
| package/core.py:30 | 父直接来源优先，再取新 hits，最多 20 篇 | clarify 面板保护锚定文档并对称放入新证据；普通节点保持原策略。 |
| core.py:17/80/90；v6.py:87 | ground、node_state、selector、Reader | 首版继续原接口。新增证据按共同规则加入被采用项来源；拒绝候选不混入其 proof。 |

原 planner、archive、NV-Embed-v2、全语料检索和 Reader 可保留。不会把所有节点全面改成多分支状态，也不把同祖先一致性工程包装为首版贡献。

## 端到端逻辑上限：n ≤ 6

| 资源 | origin | MVP |
|---|---|---|
| LLM 请求，含 planner 与 Reader | n+2 ≤ 8 | n+3 ≤ 9 |
| embedding 查询，含 archive 预处理 | 2n+1 ≤ 13 | 2n+3 ≤ 15 |
| 最大输出 token 总额 | 2048+6×512+1024 = 6144 | 6144+512+1024 = 7680 |

proposal 的 1024 上限替代 ordinary 的 512；judge 上限 1024。未触发仍计 proposal 成本；先预留后续节点和 Reader。字段限制长度并检查截断。proposal 整体无效按 node_failed；primary 合法但附加字段无效则继续 primary；judge 失败按 unknown。不免费增加修复调用。

新研究请求统一检查 prompt_tokens + output_reserve + 8 ≤ 16384。exact origin 另列，不能将其原容量行为与新臂混同。报告实际输入／输出 tokens、延迟、缓存和每个失败尝试；重试上限设为 3 时，24 种逻辑服务请求的保守物理尝试上限为 72，并非实际成本。预计算共享费用与求解费用分别记账。

<!-- page -->

# 6. 先固定候选，检验下一次查询的价值

先缓存共享 proposal：相同 primary、候选、初始原文和条件。各臂从不可变初始 archive 复制自己的证据池，避免原 controller 原地追加文档造成跨臂泄漏。共用模型、判读器、fallback、面板分配、来源规则、Reader 及新增额度；gold 只用于离线评分。

| 对照臂 | 两次额外查询的用途 | 实验作用 |
|---|---|---|
| Binding | 各候选名 + 尚缺证据的绑定关系／范围 | 主机制。 |
| Downstream | 分别接地原 plan 的首个依赖子问 | 隔离“核实上游”与“继续查下游”的差异；本臂不冒称完整 BeamAggR。 |
| Generic verify | 候选名 + 完整 parent／原问题 | 检验精确绑定条件是否优于通用核验查询。 |
| Gap query | 同候选、同 history 的普通信息缺口查询 | 对 CIRAG 风格策略的最近控制；query 生成调用若有，完整计费。 |
| Extra ordinary | 从已有可执行 plan 随机选新的有用查询，固定 seed | 排除只是增加检索机会；不使用无关噪声作弱基线。 |
| No extra query | 同 proposal，以完整初始面板作相同判读 | 测量新证据价值；不只给四篇 anchor，未花额度如实报告。 |

首轮 Gap 查询在共享 proposal 中同时产生，至多两条；所有臂缓存同一份提议及建议，统一 1024 输出上限并计费，无额外写 query 的 LLM。它获得相同原问、候选和历史，允许自然提出与 Binding 相同的查询；记录语义重合率。若另加 query-generation LLM，放入更高预算档。原句重复检索仅作辅助下限，缓存命中不算同成本新检索。

## 系统基线与必要消融

另列 exact origin 与采用共同容量预留的 origin 控制。全任务系统比较之外，主归因以共享前端各臂为准。固定候选和检索文档后，再比较相同原文判读与真实 BeamAggR 式聚合；一次自报 confidence 不算 BeamAggR。

先消融查询中的候选名称、未核实条件；再检查同查询下裁决方式的影响。BeamAggR、ReAgent-paper、CIRAG 的完整同模型适配放入合适的更高预算档，保留其真实能力并披露适配范围。unknown 后代阻塞／raw Reader、局部回滚分别作为后续消融，避免混入查询效果。

> 等硬额度不等于等实际成本。除共同预算设置外，还要画 EM/F1 对实际 tokens、物理请求与延迟的成本前沿；未用额度、重复查询和缓存复用均如实报告。

<!-- page -->

# 7. 研究执行顺序和停止条件

## 阶段 A：确认自然错误是否足够支撑路线

在独立开发数据上运行共享候选前端，人工核对约 100 个自然触发实例：条件是否来自原问、候选是否可定位、正确候选是否在池、是否真有待核实绑定。另抽查不触发样本，估计漏判。人工／gold 标签用于评估，不能进入线上 gate 或 query。

现有 3000 题问句的显式年份词法检查为 311 题（10.37%）；这只描述问句形式，既不是绑定歧义频率，也不是适用率。研究应覆盖身份、版本、关系与角色条件，真实触发率需由运行结果确定。

## 阶段 B：冻结候选的机制实验

优先比较 Binding、Downstream、Generic 和 Gap。分别统计新检索的条件证据命中率、实际进入判读面板的比例、判断错误及救回／损伤。若同一证据已在初始原文里，却因 prompt 才被采用，应归入判读变化，不归为新检索发现。

## 阶段 C：接回完整 origin 流程

只有机制结果有可信收益，才运行端到端改进入口。在 HotpotQA、2WikiMultiHopQA、MuSiQue 的预先固定集合上报告全题 EM/F1、成本和触发率；未知、失败和候选均错误的题不得从主表移除。按 qid 配对、按数据集分层 bootstrap 给出置信区间，开发与测试用途严格分开。

## 阶段 D：论文级适配与扩展

实现并核对 BeamAggR／ReAgent／CIRAG 的同模型算法适配；只有观察到“补查结束仍不够，但后续证据可以反向修复”的自然案例，才评估推测分支与局部回滚。训练型 gate、VOI、复杂 selector 暂不进入首版。

| 决策门 | 继续条件 | 停止或重新定位条件 |
|---|---|---|
| 可达性 | 自然触发与正确候选覆盖足以影响全任务 | 只在注入错误的合成题上有效。 |
| 机制 | 同预算下优于 Generic／Gap 和 Downstream；净救回为正 | 仅优于单候选，或只来自更多检索。 |
| 归因 | 查询改变证据命中，且带来可重复答案收益 | 收益仅来自 proposal、Reader 或门控。 |

> 核心指标：全题 EM/F1；candidate coverage；trigger precision/coverage；binding repair precision；rescue/harm；新条件证据命中／可见率；unknown 误判 contradicted 率；judge 错误率；全部实际成本。Oracle 上限只放离线附表。

<!-- page -->

# 8. 核心参考文献与复现边界

## [1] BeamAggR | ACL 2024

Zheng Chu et al. **BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge for Multi-hop Question Answering.**

https://aclanthology.org/2024.acl-long.67/

核对正文、Algorithm 1 与相关附录。未从正式页面或 PDF 链接取得作者仓库；同模型实验应称算法适配，不称官方精确复现。

## [2] ReAgent | EMNLP 2025

Xinjie Zhao et al. **ReAgent: Reversible Multi-Agent Reasoning for Knowledge-Enhanced Multi-Hop QA.**

https://aclanthology.org/2025.emnlp-main.202/

核对 §4、Appendix C 与多跳案例。官方仓库 astridesa/ReAgent，提交 31c3d9f4ad72846a6cb8c2cc1a4e34dba91cea84。静态代码有分解／检索占位、重复字符串 verifier 与 checkpoint 容器；不能用演示实现当弱基线代表论文。

## [3] CIRAG | ACL 2026

Zili Wei et al. **CIRAG: Construction-Integration Retrieval and Adaptive Generation for Multi-hop Question Answering.**

https://aclanthology.org/2026.acl-long.1203/

核对 §3.3-3.4 与 Appendix A.2。官方仓库 52566rz/CIRAG，提交 d78caa0d1e958814212fc59bfae777b1aac22718。已读历史查询／事实整合、来源映射和 cascade；未执行训练或论文实验。冻结模型适配与 distilled student 分列。

## [4] DIVA | NAACL 2025

Yeonjun In et al. **Diversify-verify-adapt: Efficient and Robust Retrieval-Augmented Ambiguous Question Answering.**

https://aclanthology.org/2025.naacl-long.56/

核对 §3、Algorithm 1 与相关 prompts；未审查作者源码。名称为 DIVA。对其“无 verifier-failure 新检索”的观察限于论文算法定义。

## [5] End-to-End Beam Retrieval | NAACL 2024

Jiahao Zhang et al. **End-to-End Beam Retrieval for Multi-Hop Question Answering.**

https://aclanthology.org/2024.naacl-long.96/

官方仓库 canghongjian/beam_retriever，提交 902980bfb5bd47a963569ab9c3621fd5ae81204a。已读 README 与 retriever_model；未执行训练、权重或 Reader。passage beam 与实体答案候选应区分。

<!-- page -->

# 9. 相关参考文献与可追溯材料

## [6] RAFTS | ACL 2024

Zhenrui Yue et al. **Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments.**

https://aclanthology.org/2024.acl-long.556/

## [7] C-RAG | NAACL 2025

Leonardo Ranaldi et al. **Eliciting Critical Reasoning in Retrieval-Augmented Generation via Contrastive Explanations.**

https://aclanthology.org/2025.naacl-long.557/

## [8] KG-CRAFT | EACL 2026

Vítor Lourenço et al. **KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking.**

https://aclanthology.org/2026.eacl-long.302/

## [9] InfoGain-RAG | EMNLP 2025

Zihan Wang et al. **InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering.**

https://aclanthology.org/2025.emnlp-main.365/

## [10] PropRAG | EMNLP 2025

Jingjin Wang et al. **PropRAG: Guiding Retrieval with Beam Search over Proposition Paths.**

https://aclanthology.org/2025.emnlp-main.317/

## [11] D²Plan | ACL 2026

Kangcheng Luo et al. **D²Plan: Dual-Agent Dynamic Global Planning for Complex Retrieval-Augmented Reasoning.**

https://aclanthology.org/2026.acl-long.216/

> [6]-[11] 已核对正式论文与相关方法；未执行或审计其作者实现。不用摘要筛选项支撑精确否定性差异。论文报道的成绩与本仓库同模型实验不可直接混比。

## 工作区记录

> 本报告：docs/research/RESEARCH_ROUTE_20261002.md。编号文献／审查范围：docs/research/route_references_20261002.json。来源哈希：docs/research/route_source_ledger_20261002.json。详细备忘录：tmp/research/route_20261002_*.md。
