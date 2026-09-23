# Day2 无相关性代理的桥接消融

默认 `fusion` 仍使用当前快照中的完整 `EvidenceBridgeSearcher` 与真实
`SetReranker`。新增 `fusion_proxy_free` 对应 day2 实施计划第 313 行的
`proxy_mode=none`。它是明确的多机制关闭对照，**不是声称原始搜索器拥有
activation 开关，也不是只改变公式 A 的单因素干预**。

| 配置/方法 | 候选发现与调度 | 集合相关性评分 |
|---|---|---|
| `fusion` / `proxy_mode=activation` | 原版多根预算搜索、四项测量、pair、pivot、有界试探 | 实际 pointwise reranker |
| `fusion_proxy_free` / `proxy_mode=none` | 必要信息需求轮转、局部 ANN 排名、原文条件桥接 | 不调用、不生成占位分数 |

无代理模式的流程：

1. 调用相同 vendored `DependencyRetriever.build_initial_pool(expand=True)`，
   执行真实 dense 与初始桥接；初池仍共享该节点的 ANN 配额。
2. 对未完成的必要需求轮流安排一次条件提案。检索阶段不把“发现了文档”
   当作语义上已经满足需求；完成状态仍由后续原文映射与 resolver 判断。
3. 每个需求交替使用按 ANN 发现顺序排列的新根，以及前次提案返回的真实
   文档组成的后续路径。局部候选按实际 ANN 排名处理。
4. 延续路径调用原版 `propose(target, premise_ids)`；target 与所有前提均为
   已发现的原始文档。新问题包含这些完整原文；不是仅反复检索原问题。
5. 单条路径不重复文档；相同需求、目标和前提状态不重复发出提案。全题
   ANN 预算、节点公平份额、最多两次预留缺口回查和所有候选并集保持不变。
6. `retrieve_missing`、原文映射、依赖支持、失效审计和最终 reader 沿用融合
   公共流程；关闭相关性代理不关闭证据检查。

此模式禁用所有依赖 set-score 的量子测量、四集合测试、pair 测试、A 保留门槛、
基于分数的试探门槛和 pivot。它按 ANN 顺序而非 source 的几何多样性顺序选择根。
因此比较回答的是“带完整相关性代理搜索策略是否比明确的无代理条件桥接策略
更有用”，不能把差值全部归因于交互公式 A。ANN 上限相同也不代表总费用相同；
应同时报告模型调用、token、时延及依赖完整性。

运行时使用 `--arms original fusion_proxy_free`。配置可完全省略 `reranker`；
此时方法和预检均无 reranker 请求。若比较列表中还有默认 `fusion`，则仍需要
真实 reranker。如果配置保留了 reranker，现有运行器会执行独立的部署协议预检；
预检费用单独记录，不属于无代理方法的题内调用。

日志中的 `proxy_mode` 明确为 `none`，`search_archive.method_version` 为
`day2_proxy_free_requirements_ann_v1`，包含需求轮转、根/延续选择、真实 proposal
来源、原文导航路径、禁用模块、剩余状态和停止原因；`measured_sets` 与
`activations` 为空。不会用常数分数或免费隐藏 reranker 模拟关闭状态。

机制测试验证零 reranker、真实多跳原文提案、必要需求轮转、共享预算、完整
候选并集，以及相关性量子参数不影响 none 模式。完整 HTTP 协议测试还使用
真实包内 HotpotQA 第一题、原 tokenizer/向量、原版与该消融的实际 spawned worker，
仅将模型端点替换为明确的脚本服务。这些验证不证明真实问答准确率。
