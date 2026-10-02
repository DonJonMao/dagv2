# DAG-Resolve：分歧触发的绑定条件检索

这是基于 dagv2-origin 的独立改进入口。复用原问题规划、archive、全语料 NV-Embed-v2 检索、单值 DAG、来源闭包、selector 和 v6 Reader。冻结的 `package/`、`dagv2/` 与融合版 `dagbt/` 文件均不改写。

## 算法边界

每题按原 plan 的依赖顺序，在**第一个有后代的桥节点**使用一次候选输出协议，替代该节点原有回答调用。其余节点保留 ordinary `answer/sources` 协议。

这里先固定干预节点，再检查是否有合格竞争；没有在全部节点之间按歧义或终态影响选择干预位置。因此当前实验可评价该干预范围内的绑定查询，不能评价终态敏感调度。自然错误审计应分别报告：错误绑定位于首个桥节点的比例、这些实例中正确替代项已在初始面板／候选中的比例，以及实际触发覆盖率。分母和不触发实例必须保留；触发子集收益与全任务收益分别报告。

候选协议保留 primary 的完整普通来源，允许 **0、1 或 2 个候选，两个候选包含 primary**。第二候选必须确实出现在当前原文中，并有逐字出处；不能靠模型猜名字、猜别名或凑够两个。只识别到一个候选时，继续原单值路径，没有额外检索或判读。

补查同时要求：

1. 两个不同候选均有有效原文锚定；primary 非空且有普通来源。
2. primary 满足原问必要绑定条件的证据未决、冲突或被反驳；primary 已充分支持时不自动补查。
3. 原问中能逐字定位当前绑定的关系、对象及限定条件；条件不得指向出生地等待查下游属性。
4. 有两个不同且尚未请求的查询；完整保护证据、后续节点和 Reader 能满足容量与请求预算。

两项完整绑定均获支持时，记录合法多值事件并沿用 primary 路径，不按来源数强迫二选一。首版不实现真实多答案覆盖。未产生合格竞争不会在其他节点再次尝试候选提议。

## 补查与提交

从至多两个原问绑定条件中，按年份／版本、身份、关系及原问顺序选择 primary 尚未获得支持的一项。确定性查询为：

```text
候选在原文中的名称 + 所选条件的完整原问片段
```

片段保留关系、对象、年份、版本、first/former 等限定词，不另调用模型写 query。例如虚构问题“1998 年《归途》的导演出生在哪里”，候选为陈海和林舟时，查询两人的“1998 年《归途》的导演”绑定，而不是展开两条出生地分支。

每个候选最多一次全语料 dense top50，结果并入该题独立证据池。联合判读最多 20 篇完整原文；primary 普通来源、候选锚定和初始绑定证据最多保护 4 篇，超过则跳过补查。两侧新 hits 使用相同 rank 配额、去重后用初始原文补齐。容量不足先移除中性填充，再同步缩短两侧 rank 窗口；不截断文档或保护引用。

一次联合 LLM 判读检查所有候选、所有必要条件，返回 supported、contradicted、unknown 或 conflict。程序检查文档索引与逐字 quote，并保存 `doc_id/start/end/quote`。语义蕴涵由模型判读，引用存在本身不等于事实充分。缺资料、另一年份或较低检索分数都不是反证。

**协议无效与语义 unknown 分开。** 每个候选须明确判读所有必要条件；空列表、漏判条件、坏引用、无证据的支持／反驳、无依据的多值标志或响应结构错误，均属于内部 `invalid`，并将本次结果标记为 `protocol_valid=false`。unknown 所附引用也必须合法。任一部分无效即退出整次可选修正，原样保留 primary 普通答案和来源，不采纳部分有效判读，不追加其支持文档；trace 记录 `failed / judge_protocol_invalid` 与具体诊断。提议的附加字段无效时同样不触发补查。模型不输出 invalid 标签，它是程序校验结果。

| 判读结果 | 提交规则 |
|---|---|
| primary supported | 保留 primary，加入其新支持证据。 |
| 判读协议合法，替代项 supported，primary unknown | 改选替代项，标记 `unexcluded_competitor=true`；不声称证明唯一。 |
| 替代项 supported，primary contradicted | 改选替代项。 |
| 双 unknown、冲突 | 保留 primary，记录相应语义状态。 |
| 判读协议 invalid 或可选 HTTP 失败 | 原样保留 primary 普通答案与来源，记录 invalid 和失败原因；不计作证据纠正。 |
| primary contradicted，无 supported 替代 | 清空答案和来源，成为 origin 式 unresolved；下一跳沿用原未解父问题接地。 |
| 双 supported／合法多值 | 记录事件，保留 primary；只加入 primary 支持，不混入替代项来源。 |

判读之前不执行依赖子节点；提交之后每个子节点只执行一次。首版没有全面 beam、推测子分支、回滚、训练、额外 planner 或新 selector。未决 primary 仍会进入原 Reader 的答案链，这是沿用原接口的已知限制，trace 中不把它记成已经核实。

## 容量和成本

所有新 LLM 请求均按实际 tokenizer 校验 `输入 + 输出预留 + 8 <= 16384`，使用完整原文；普通节点、候选提议、判读和 Reader 的预留分别为 512、1024、1024、1024，planner 为 2048。无法容纳保护证据时跳过可选补查；必需阶段失败保存失败状态，不免费追加修复调用。

对于最多六节点的 DAG，端到端逻辑上限为：

- LLM：planner 1 + 节点 6 + judge 1 + Reader 1，最多 9 次。
- Embedding：archive 最多 7 + 节点 6 + 补查 2，最多 15 次。
- 最大输出 token 总额 7680；没有触发时，候选提议替代普通调用的实际成本仍计入。

每种相同请求最多持久化重试 3 次。逻辑调用、物理尝试、缓存命中、重试、失败／未知用量分别记账；72 次只是最保守物理尝试上限。`runner_cost` 与 `cost.json` 包含真实缓存记录，部分或无效 usage 不冒充完整成本。

## 使用

使用仓库原 Qwen/NV 配置和本地 tokenizer。新入口仅接受 legacy 模型 profile；embedding model 必须与冻结 NV 索引一致，不能将不同模型的同维向量混用。

```bash
# 无网络检查：数据、索引、tokenizer、非标签原文件哈希
.venv/bin/python -m dagresolve.runner check --dataset hotpotqa --config config.example.json

# 服务可用后运行；每种配置／范围用独立输出目录
.venv/bin/python -m dagresolve.runner run --dataset hotpotqa --config config.example.json --output outputs/resolve-hotpot-pilot --limit 10

# 完整生成本次指定范围后，单独评分
.venv/bin/python -m dagresolve.runner evaluate --dataset hotpotqa --config config.example.json --output outputs/resolve-hotpot-pilot
```

同样支持 `2wikimultihopqa` 和 `musique`。无 `--limit` 时使用该数据集固定 1000 题。生成阶段不读取标签；显式评分仅在 complete、精确行范围与代码身份检查后打开 labels，失败题按零分计入全任务指标。

输出包含 `manifest.json`、`progress.json`、不可变 `inputs/`、`rows/`、逐题 `calls/` 和 `cost.json`。manifest 绑定方法、代码、配置、数据、问题范围和预算。相同输入可恢复、缓存重放；已经完成的失败行不会自动重跑，改变代码或范围必须使用新目录。CLI 使用单数据集、单 worker；原模块包含全局配置，不应在同一进程交错执行多个不同 runtime。

## 证据召回的范围

原有 R@5/10/20 和 all@5/10/20 保持原定义，计算 `budgets[k].selected_doc_ids` 的 selector 输出召回。Reader 为满足容量可能继续裁剪，因此 selector 的 R@20 不能解释为 Reader 已看到全部支持证据。

评分同时报告以下百分比，均使用原来的 gold 支持组规则，MuSiQue ID 使用相同归一化：

- `reader_support_recall`：根据 `ranking.reader.panel_doc_ids`，逐题计算支持组召回，再对完整任务范围宏平均。
- `reader_all_support`：记录的 Reader 面板覆盖全部支持组的题目比例。
- `reader_context_trim_rate`：记录已移除文档，或 selector@20 文档未进入记录的 Reader 面板的题目比例，分母为完整任务题数。

缺失／畸形 Reader 面板的题仍计入全题分母，Reader 召回为零；缺失面板本身不推断发生裁剪。另在 `reader_evidence_scope` 报告已记录面板数、不可用数、裁剪题数及以已记录面板题为分母的裁剪比例；无面板时该条件比例为 null。答案生成失败不自动抹掉已记录输入面板的召回。该面板衡量记录的 Reader 输入范围，并不保证远端服务已完成推理。

代码更新后仍须通过生成 manifest 的代码身份校验，旧版本生成目录不会被当前 evaluate 自动重解释；新实验使用独立目录。

## 自检与研究范围

```bash
.venv/bin/python -m pytest -q tests
PYTHON=.venv/bin/python bash scripts/smoke.sh
```

新增测试使用真实 tokenizer 与 loopback HTTP 服务验证 planner、archive、候选触发、来源判读、单值下游、Reader、缓存及失败成本；这是协议与实现检查，不是模型性能实验。原始文件和 vendor 哈希须全部保持一致。

论文机制实验仍需固定候选、判读器、Reader 和预算，比较绑定查询、下游查询、通用 verification 和 gap query。当前入口实现主方法，不包含完整 BeamAggR/ReAgent/CIRAG 适配，也不声称已取得新 EM/F1 收益。

## 论文贡献口径

可以围绕以下三项设计组织方法说明；它们构成同一个局部纠错闭环，不能分别声称首创。

| 设计 | 相对 origin 的变化 | 需要验证的效果 |
|---|---|---|
| 原问约束驱动的竞争绑定诊断 | 识别有出处的竞争候选，并检查 primary 是否缺少必要绑定条件的证据；只有多个答案并不足以触发。 | 与“有两个候选就补查”及总是补查比较，统计触发准确率、漏判和实际成本。 |
| 候选条件化的绑定优先检索（核心） | 新查询用于核实候选是否满足原问关系和限定条件，然后再查下游属性。 | 固定候选、判读器、Reader 和预算，对比下游查询、通用核验及普通 gap query；消融候选名称与完整条件。 |
| 预算约束的提交前局部修正 | 在 node_state 前进行至多两次补查和一次联合判读，按证据状态提交单值及其来源，再执行下游。 | 固定查询比较提交规则，统计净救回、误修正、unknown 误删和合法多值误处理；报告 EM/F1 与真实成本。 |

BeamAggR 已有多候选展开；ReAgent 已有多假设与下游之前的条件核验；CIRAG 已有原问驱动的补缺查询。最应验证的增量是：观察到竞争绑定后，检索其尚未核实的必要条件，是否比这些强对照更有效。完整文献依据与对照方案见 `docs/research/RESEARCH_ROUTE_20261002.md`。

论文的三项贡献宜写为：形式化并诊断中间绑定错误；提出分歧触发的绑定补查方法；通过控制实验分析修复收益、成本与适用边界。第三项必须由真实实验完成，当前协议测试不能代替该实证贡献。
