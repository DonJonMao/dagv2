# DAG＋BT：接地子问题评分与终端直接回答

`dagbt_local_terminal_v1` 在同一个融合引擎中实现三个算法修改：取消游离于 DAG 的原问题 dense 基线；BT 按当前接地子问题评分；指定终端节点在求解调用中直接生成最终答案。`fusion` 及其消融方法保留为 legacy，供历史结果读取和复现。冻结的 original、dagresolve 和 vendor 搜索实现没有改动。

## 从任务 DAG 到最终回答

原问题只做一次规划。计划声明各任务的前置依赖、`retrieval` 或 `compose` 属性，以及 `final_node_id`。最多六个初始节点，包含确有需要的最终比较或综合任务；原文触发的细化最多两个节点。自然已有最终任务时直接使用它，例如导演→毕业院校→城市中的城市节点。节点编号和存储顺序不决定终端身份。

只有必要父任务都受支持时，当前任务才执行。程序将已确认父答案代入占位符；未写进问题的声明输入以明确绑定附加，实体、年份、关系和范围限制继续保留。由此得到固定、自足的接地子问题 q̃ᵢ。取证节点围绕它做 dense、BT 初始桥接和条件扩展。原问题仍可作为检索提示的总任务背景；评分服务的 query 始终是 q̃ᵢ。

所有发现的原文进入共享候选池。导航边只是发现路径，不能直接当作结论支持。原文映射为有来源位置、角色和时间范围的证据判断，再由节点求解核查其组合是否足以回答子问题。未完成映射是未评估；不把它当作无关证据。跨节点原文复用不转移语义支持，也不突破用户和时间可见范围。

合成节点直接结合受支持父结论和必要来源求解，父证据齐备时不重新检索 Q。缺口由有关父任务的既有补查或有限细化处理。每个支持路线内部是共同必要前提的 AND，同一答案的独立支持路线是 OR；两者不改变问题层 DAG 的规划。

终端使用同一节点求解协议，同时返回语义 `answer` 和必要的 `final_prediction`。普通事实题可以直接发布语义答案；单选题在这个调用内依据公开选项输出一个严格合法的选项标签。选项和输出协议只进入终端生成，不进入规划、检索、映射或评分。评测金标在生成阶段保持隔离。

必要审计之后，程序重新编译支持关系，确认终端采用的实际依赖版本和来源仍然有效。上游答案或范围变化使实际支持失效时，沿原机制重评受影响任务；独立替代路线仍可保留。无冲突的正常路径只生成一次终端答案，审计后直接发布，不再调用模型改写它。

## BT 如何寻找互补证据

BT 保留原有多起点、候选对、retain / speculate / pivot 和有限调度机制。锚点 m 是需要补充材料才能发挥作用的原文；P 是已经积累的前提集合；H 是当前测试的新候选组，可以是候选对。检索探针可以包含锚点、P、父来源和缺口；这些材料不会动态进入评分 query。

固定当前 q̃ᵢ，分别测量：

```
r0 = R(q̃ᵢ, P)
r1 = R(q̃ᵢ, P ∪ {m})
r2 = R(q̃ᵢ, P ∪ H)
r3 = R(q̃ᵢ, P ∪ H ∪ {m})
M  = r3 - r2
A  = (r3 - r2) - (r1 - r0)
```

M 表示补充候选之后锚点对当前子问题的边际贡献；A 表示这个贡献比补充之前提高了多少。四项共享相同的查询、上下文、模型和可见语料。空集合照常评分。阈值和搜索动作沿用 frozen vendor，令 ε 为已有 marginal_epsilon（默认 0）：

- A > ε 且 M > ε 时 retain，保留锚点，并将 H 加入前提后继续展开。
- A > ε 但尚未满足 retain 条件时 speculate，在已有深度和状态数限制内探索这个补充后的状态。
- M < −ε 时允许 pivot：从 H 中选一个未出现于锚点路径的原文作为新锚点，以 P ∪ H 的其余原文为前提，移除旧锚点。此动作仍受原有次数和深度限制。

其余测量不因证据数量多或最终答案标签吻合而被接受；搜索中未继续展开的原文仍留在共享候选池中供核查。

本地评分上下文包含完整查询、语义父绑定与版本/范围、可见原文内容、序列化版本、模型和评分协议。相同上下文、相同规范集合复用分数；跨子问题或绑定版本不复用。模型客户端共享，请求序号全题唯一。服务侧精确前缀缓存和原文集合分数缓存仍是不同机制。

## 可解释的完整离线例子

`scripts/demo_local_terminal.py` 使用真实融合引擎和 frozen BT，模型/向量/分数由明确的离线服务替身提供，全部人物和原文是虚构的，没有评测标签。

1. 影片资料说明 2012 年《Ash Wind》的导演是 Lin Zhou。
2. 接地任务查询 Lin Zhou 的毕业院校。访谈只说明他毕业于 Screen Arts 学院 K7 班；班级档案将 K7 接到 School of Imaging；学院说明将该项目的学位接到 North University。三篇共同成立才足以推出院校。
3. 城市任务接地为 North University 所在城市，用城市资料直接生成 River City。该节点就是规划指定的终端。

院校 BT 中，锚点是访谈，H 是“班级档案＋学院说明”。离线评分为 r0=.10、r1=.20、r2=.10、r3=.95，因此 M=.85、A=.75，候选对被保留。三篇原文随后也由节点求解组成必要支持；BT 的保留动作自身不是答案正确性的证明。

运行这个例子：

```bash
.venv/bin/python scripts/demo_local_terminal.py --output outputs/local_terminal_v1/demo
```

## 预算、结果与成本

全题 ANN 上限 36、集合评分上限 512、生成上限 24；独立 Reader 为 0。不同评分上下文的同一集合各消耗一个逻辑评分单位，同上下文重复不重复扣费。四项预检同时遵守局部搜索份额与全题 Ledger。mapping 保留必要审计和剩余节点求解额度；已移除全局选择及其 repair 预留。HTTP 重试仍按真实物理请求计费。

结果保留 `answer.status` 和 `answer.prediction`，另记录 `answer_source=dag_terminal`、终端 ID、输出格式身份、实际支持版本与来源。未支持、歧义、格式失败、审计未完成或预算耗尽均不发布成功预测。没有历史答案、最后成功节点或 Reader 兜底。

`terminal_input_doc_ids` 记录真正进入终端生成的证据视图中的原文；`sources.doc_ids` 是所选有效支持路线的可追溯来源闭包。输入可以包含核查材料，闭包可以沿父结论延伸，两者不能互换。闭包来自实际支持，排除无关搜索历史和已失败的旧路线。

新结果的 `budgets` 为空，未伪造 k=5/10/20 文档选择，也未伪造 Reader response。旧 Reader / FinalSelector 指标为 `not_applicable` 或数值 null；终端来源召回以独立名称报告。旧格式仍可读取和评分。

成本分别记录逻辑生成、节点求解、ANN、逻辑集合评分、rerank HTTP 请求，以及 runner 的物理请求日志和服务响应中可用的上游用量。各 scorer 的累计成本按上下文汇总一次，不重复累加历史快照。上游前向次数未由服务提供时保持不可观测，不能拿集合数推测。合成示例的完整计数见验证报告，不代表真实任务的成本或准确率。

不再执行的 legacy 步骤是：额外 `__baseline__` dense、全局 raw memory review、FinalSelector、独立 Reader，以及终端形成后的 read-all 重答。真实单节点问题恰好等于 Q 时，其自身取证正常允许。

## 实际入口和恢复

新配置是 `configs/local-terminal.example.json`。凭据继续使用未跟踪的本地凭据文件或环境变量。推荐包装脚本已实际指向新方法和独立输出目录：

```bash
bash scripts/run_v3.sh preflight --datasets hotpotqa personamem
bash scripts/run_v3.sh start --datasets hotpotqa personamem
bash scripts/run_v3.sh resume --datasets hotpotqa personamem
bash scripts/run_v3.sh status
```

start / resume 使用同一协调器和 manifest 校验，原始对照仍是 original。以上全量入口仅供之后明确启动；本次验证只运行固定小规模 smoke。配置、源码、方法或问题范围变化必须使用新目录；不得恢复 f6a1721 的旧实验 manifest。resume 默认跳过已终态失败题，不自动增加题目尝试。

固定 3＋3 smoke 的运行和中断后恢复命令相同，不加载评分标签：

```bash
.venv/bin/python scripts/smoke_local_terminal.py \
  --config configs/local-terminal.example.json \
  --output outputs/local_terminal_v1/smoke_3plus3_v2
.venv/bin/python -m dagbt.runner status --output outputs/local_terminal_v1/smoke_3plus3_v2
```

离线回归和差异检查：

```bash
.venv/bin/python -m pytest tests -q
git diff --check
```

## APC 等价性边界

旧方法和新方法有意改变评分目标与最终生成路径，不要求分数、路径或答案相同。前缀缓存工具现在识别算法版本和评分上下文、按上下文汇总成本，并支持无 Reader 的终端结果。旧 trace 保持可读，但不能作为新方法的分数缓存；跨算法冻结混合、搜索回放和等价性比较会拒绝。

等价性只能对照“新方法 APC off ↔ 新方法 APC on”，使用相同的上下文、完整 token 输入和物理评分协议。没有实际双臂验证时，状态继续是 **IMPLEMENTED_NOT_HARDWARE_VERIFIED**。本次算法修改不证明准确率提高、总体降本或 APC 提速。
