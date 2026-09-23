# Day2 设计对照与研究边界

日期：2026-09-23。已重新逐段核对 `docs/design_reference/implementation_plan.md` 和 `fusion_report.md`，并结合最新 BT 工作区及原 DAG 源码做实现检查。设计原文与来源哈希同目录保存。

## 对用户要求的映射

| 要求 | 当前实现与可核验证据 |
|---|---|
| DAG 放入 projects | `/Users/mao/projects/dagv2`；完整原包 57 文件 checksum 清单 |
| 保留原代码、单独支线 | `original-dagv2` 原版提交；`fusion/latest-bt` 增量；原 package/dagv2/data/脚本不改 |
| 使用最新 BT、原目录不动 | 当前工作区 57 个模块快照；vendor manifest；只在 DAG 侧适配 |
| 融合完整、能研究模块效用 | 原生 BT 搜索、依赖求解、AND/OR、反证、细化、闭包；2×2 和额外消融，完整事件日志 |
| 同题老/新版各跑一次 | paired runner 冻结同一问题列表，逐题轮换各臂；独立任务权威记录 |
| 一键后台、故障不拖垮全局 | detached 进程、持久隔离 worker、任务超时、失败落盘继续、文件锁、显式重试、任务边界恢复 |
| 日志和结果齐全 | 请求缓存、attempt 历史、节点/证据/选择/reader 快照、比较 JSONL/CSV、汇总和模块指标 |

## 忠实迁移而不是照搬偏好任务措辞

1. 用户现指令要求 DAG 为宿主，覆盖 day2 原文的 BT 为宿主路径。新增逻辑放在 `dagbt/`，不修改 BT 原目录。依赖语义仍按设计执行。
2. 原文记忆改为语料 passage；source role 为 document，保持 title + newline + text 的精确字节/字符内容。不会把百科文档伪装成 user/assistant 对话。
3. 事件时间未知保留 null；只有映射输入中存在时间引文才记录时间。没有时间语境的题不虚构“当前偏好”或 latest-wins。
4. 顶层需求采用原问题和 q-only 计划的全部终端节点联合完成判据；初始最多 6 节点，最多 2 个有证据出处的细化节点。细化不能改写原问题或终端 hash。
5. 保留原 DAG 的 q-only planner 提示、schema 和尾注验证；融合模型调用预算/输出预留独立明确。原版臂仍走未改动的原生成函数。
6. day2 建议默认不用 activation proxy，但最新用户要求使用最新版 BT。本实现默认执行最新完整 BT 搜索，保留四集合相关性代理及真实 reranker；代理从不成为依赖合法性的依据。Dense 控制组没有隐性 reranker。这个选择是保留最新算法的适配，不是声称四分数证明因果效用。

## 方法约束核对

| 约束 | 实现位置 / 行为 |
|---|---|
| 发现边不冒充支持边 | vendor ProposalEdge 原定义；bridge 只输出候选与导航 trace；engine resolver 另报 actual used parents |
| 已支持父结论才能接地 | engine.grounded；unknown/ambiguous 不绑定；缺口/父原文传入真实 BT |
| 所有发现候选保留 | bridge 合并 initial + 全部 proposal batches；engine 累计候选与全量分块映射 |
| 长文不只看头部 | engine.add_candidates 覆盖全文并对相邻块重叠；绝对 offset/hash；预算未映射部分显式保留 |
| 引用与语义判断分开 | support.make_span/compile_graph 验证原文、offset、ID/hash；supported 仍是模型判断 |
| AND/OR 完整支持 | source/guard/actual parents 同时满足；同结论同 scope 可替代；计划父项不自动并入 |
| 无环与版本 | compile_graph 拒绝向前/未知/环；answer 或 scope 变化更新版本并使旧父绑定不可用 |
| 冲突局部失效 | invalidate_support 按具体替代项；未受影响 OR 仍可用；历史反证与支持原文作为受保护组 |
| 重求解不清洗冲突 | 未显式解决的 conflict ID 继承；新替代项接受再次审计；仅更换 ID 不能恢复覆盖 |
| 解决冲突有来源 | resolve_conflict 要求逐项回应原反证，resolution 引文进入必要 guard 闭包 |
| 完整闭包与实际预算 | select_support 在最多 3^8 状态内选 necessary/optional 终端覆盖，再最少 token；原文去重 |
| 无证据/预算停止 | unknown/partial/ambiguous 与基础设施错误分开；不静默切换 dense 或丢弃必要前提伪称覆盖 |
| reader 公平与链消融 | 四个新臂同 raw-only 提示/模型/预算；chain 仅注入来源齐全的结论和精确引用，并计实际 token |
| 每题每次尝试共享预算 | ANN36、unique set512、LLM24、reader1、gap2计入36；修复/重试计费；缓存重放保留原重试配额占用；显式重试新尝试的累计成本另外完整报告 |
| 恢复身份 | config/source/vendor/语料/tokenizer 哈希，任务边界重放；不假装持久化 Python frontier |
| 标签隔离 | 生成参数仅 id/question + corpus；全部方法任务终态后评分读取 evaluation_only；失败率单独报告 |

## 完整性与有限预算

“全部候选保留”是保留发现记录，并在预算内映射全部原文；预算不足时记录具体未映射块，不能声称所有候选均经语义审查。映射/求解输入超窗显式记录，reader 仍只使用可行证据。闭包枚举的精确性只针对已发现、已编译图，不是召回或自然语言蕴涵的全局保证。

平面选择控制组需要额外一次 LLM，依赖选择使用确定性枚举。两者共享同一总预算，且真实开销分别报告；等 ANN 次数不等于等成本。固定候选池用于进一步隔离选择本身。

`fusion_no_conditions` 只关闭独立条件审计阶段；mapper/resolver 仍检查基本的实体、时间、条件和引用有效性，不能将该消融描述成“完全取消条件推理”。平面选择组的最后一次选择调用有独立保留，审计的 JSON 修复和 HTTP 重试也不能占用它。

另外保留两项设计要求的控制：`fusion_navigation_closure` 强制保留首次发现的导航路径原文，并计入最终上下文；`fusion_proxy_free` 实现设计的无评分代理调度，复用原文条件提案和缺口检索但不运行 scorer。后一项不是最新版 BT 的原生开关，因为原实现的保留、试探和优先级均依赖 activation；因此它同时移除依赖评分的调度机制，不能把其差值归因为 activation 单个公式。

四臂齐全时报告同题交互差值及 95% 配对 percentile bootstrap 区间（固定 seed、1,000 次抽样）；全任务与四臂共同成功子集分别计算。区间假定问题独立可交换，不能覆盖所有共享语料相关性或模型采样不确定性，也不是因果证明。

## 原版保持原样的已知差异

原版 resolver 用非空 answer/sources 标记 resolved，自动并入所有已解析 planned parents；reader 注入全部 resolved 节点链，可能包含最终 top20 未保留的来源，并偏向链尾。原版 source_panel 实际优先父节点直接来源而非全递归闭包；其本地上下文检查未预留 512/1024 输出 token。

以上均未偷偷修复到 `original` 臂。融合版改进必须由 matched raw controls 支撑归因，不能仅用 original 与 fusion 的差值证明 BT 的作用。

## 实验可观察项与不可推断项

- 基准：三套包内各 1000 问，固定 corpus 为 HotpotQA 9811 / 2Wiki 6119 / MuSiQue 11656 篇；没有改成每题 distractor 范围。
- 任务：原 EM/F1；原 title-group macro recall 与 ALL 保留同口径；MuSiQue ID 前缀保持。
- 模块：候选 title-group recall、选择后 recall 和二者损失；supported/unknown/ambiguous；完整需求覆盖（模型+结构标签）；预算与 reader tokens。
- 失败：所有任务/共同成功子集分别统计；失败不悄悄删除，不拿 API 故障冒充语义错误原因。
- 不能推断：程序可核验引用不代表解释正确；模拟 HTTP 无模型，不构成准确率证据；当前没有真实模型改进结果；对时间更新的泛化仍需适当诊断数据。
