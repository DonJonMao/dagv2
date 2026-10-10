# local_terminal_v1 验证记录（2026-10-10）

基准为 `f6a172123ca05ec4fdc0463b94cd3a91253bfbbb`，实现提交为 `b0e86fb`，离线验证提交为 `4220a68`，分支为 `feat/local-scoring-terminal-answer`。这两个提交已按用户后续明确要求推送到 GitHub；本文记录最终真实 smoke 结果。既有未跟踪的研究资料、压缩包和 tmp 内容未纳入提交。

三项算法修改已完成实现、兼容适配和离线行为验证。固定真实 3＋3 smoke 已完成，六题均为单次尝试，运行状态为 complete_with_failures；没有充分支持的成功预测。真实成功输出与任务质量尚未得到本次 smoke 证明，不能据此宣称准确率提升、降本或 APC 提速。

## 行为证据

新增测试位于 `tests/test_local_scoring_terminal.py`、`tests/test_local_terminal_integration.py`，以及 `tests/test_bridge_http.py` 的本地评分 HTTP 检查。以下是要求与实际覆盖的对应关系。

| 要求 | 证据 |
| --- | --- |
| 不执行游离 Q 基线 | 链式集成测试只有 director / university / city 三次取证；新路径没有 baseline attr、FinalSelector 或 Reader 调用 |
| 单节点问题等于 Q 可以取证 | `test_single_real_node_equal_to_Q_is_allowed` |
| 实际 payload 使用本地查询 | 模拟后端对 q1 / q2 / Q 有不同分数；`test_payload_queries_and_cache_namespaces_are_local` 检查实际分数和 payload；loopback HTTP 测试确认两个任务的真实请求 query |
| 四项一致性和 A/M 原义 | 同一模型、query、namespace 的完整四项测试；合成例子执行 frozen 搜索，检查 M=.85、A=.75 和候选对接受动作 |
| 缓存作用域 | 同任务命中；不同 query、父答案、父版本、范围、可见内容或模型 namespace 不同；Transport 也区分算法与上下文 |
| 全题预算 | 八个 scorer 共同消耗 512 后，第九个完整四项预检被拒绝且不发请求；重复上下文不重复扣额度 |
| trace 和成本 | stage 全题不重复；archive 测量含节点、查询和上下文；各 scorer 汇总一次；APC 累计快照按 namespace 去重 |
| 三节点链直接发布 | 三次 resolve、一次 plan、必要 audit，无 q4 / select / reader；来源版本和闭包可检查 |
| AND compose | 两父均受支持才执行；缺父时阻塞；终端不取证 Q |
| 终端身份 | final slot 显式声明；改变 ID 和存储顺序后仍选择 metadata 指定节点；缺失/循环/超额计划拒绝 |
| 失效与替代支持 | 实际父支持被反证时不发布旧终端；直接独立路线仍有效时只发布存活路线的来源；现有 support 版本回归保留 |
| 未完成状态 | unknown / partial / ambiguous、格式失败、audit 失败、mapping 未完成和额度耗尽分别保留非成功结果；mapping 保留真实终端和 audit 的调用额度 |
| 单选与标签隔离 | 选项只出现在终端生成；严格 choice parser；无额外选项匹配；注入 correct_answer 被拒绝；既有公共加载与评分前标签闸门回归 |
| 兼容与范围 | 全 tests 回归包含 original、dagresolve、旧融合模式、旧结果/导出、PersonaMem 用户与时间 cutoff、来源角色隔离 |
| APC 边界 | 新上下文 score bank 身份；跨算法轨迹对照拒绝；旧 trace 读取回归；无硬件等价性/提速声明 |

最终完整离线回归为 **730 passed, 69 warnings**，48.89 秒，包含 38 项新增行为检查。该运行发生在 planner 提示补强及 mapping 预留、未评估映射和金标字段隔离检查完成之后。警告来自已有冻结矩阵运算和测试环境，不曾改动冻结实现以消除警告。

曾直接从仓库根执行不限定目录的 pytest，误收集了 outputs 中的历史解包副本和 tmp 研究代码，产生重复模块/缺依赖的收集错误。正式完整回归范围是本仓库的 `tests`，未删除旧测试或改评测口径。

`git diff --check` 通过。相对 f6a1721 逐文件比较 `package/`、`dagv2/`、`vendor/bridgetree/` 的 73 个 tracked 文件，变更为 0。原始清单 57 个文件哈希通过（仅读取标签字节计算哈希，不解析标签）；vendor 清单 57 个文件哈希通过。根 README 同样属于原始清单，保持其原有内容。

## 合成结果和成本

可运行例子使用真实 Engine 和 frozen BT，以原文驱动的离线服务替身代替模型。导演从影片资料得到，院校需要访谈、班级档案和学院说明共同成立，城市由已接地院校及城市资料得到。没有独立 gold 答案输入。

| 结果/成本层 | 观测 |
| --- | --- |
| 终端 | city，直接发布 River City，status=ok |
| 有效依赖版本 | director=1、university=1、city=1 |
| 生成逻辑调用 | 7：planner 1、mapping 2、节点 resolve 3、audit 1 |
| ANN 逻辑调用 | 18 / 36 |
| 集合评分逻辑单位 | 40 / 512 |
| reranker adapter 调用 | 19（离线替身计数，实际网络 HTTP 为 0） |
| 独立 Reader / 全局 FinalSelector | 0 / 0 |
| 上游模型请求 / 前向 | 不适用：离线服务替身不执行模型 |
| APC 硬件观测 | 未执行 |

这些数字只说明离线完整控制流及计费机制，不代表真实准确率、真实成本下降或 APC 收益。

## 真实 smoke 结果

固定范围为 canonical HotpotQA 前三题＋PersonaMem 前三题，完整语料，每题一次尝试，不加载评测标签。使用私有目录 `outputs/local_terminal_v1/smoke_3plus3_v2`；原始请求、用户原文、向量和缓存均不纳入提交。

| 题目（固定顺序） | 终态 | prediction | 审计完成 | 可靠性 | 单次尝试墙钟秒 |
| --- | --- | --- | --- | --- | ---: |
| HotpotQA 1 | audit_incomplete | null | 否 | truncated | 890.88 |
| HotpotQA 2 | unknown | null | 是 | truncated | 369.60 |
| HotpotQA 3 | unknown | null | 是 | truncated | 696.99 |
| PersonaMem 1 | ambiguous | null | 是 | truncated_and_partially_mapped | 938.14 |
| PersonaMem 2 | partial | null | 是 | truncated_and_partially_mapped | 1,341.97 |
| PersonaMem 3 | audit_incomplete | null | 否 | truncated_and_partially_mapped | 1,436.71 |

HotpotQA 1 的初次终端求解 wire 输入估计为 11,935 tokens，符合 4,096 输出＋256 余量的预留；模型返回的支持记录未通过验证，加入修复信息后输入估计增至 12,258，预检拒绝发送。既有补查后的第二次求解和审计冲突记录也未建立完整有效支持，最终没有发布成功答案。这是修复请求容量和模型支持记录的问题，初次视图估计与 wire 估计一致；没有放宽引用校验或扩大输入/生成预算来掩盖失败。

HotpotQA 2 的上游节点 supported，但指定终端 unknown；没有使用上游答案兜底。HotpotQA 3 在上游受支持时实际建立两个本地评分上下文；审计后上游 ambiguous、终端 unknown。PersonaMem 1 终端 ambiguous，未猜选项；PersonaMem 2 终端 partial，未完成映射仍为未评估，节点响应中的无效来源引用保留为失败记录。PersonaMem 3 的生成包含截断和无效协议响应，审计中的 14 条记录未通过完整验证，局部修复额度已经耗尽，最终保留 audit_incomplete。六题 prediction 均为空，未通过 gold 比较评估准确率。

六题完整审计已通过：每题一个 attempt，算法与源码身份一致；未发现预算、请求身份、wire query/documents、上下文累计成本或四项算术不一致。PersonaMem 出现公开选项的请求均为指定终端 resolve，候选、评分集合和终端输入均在题目可见范围内。没有可检查的成功预测来源闭包，成功发布及替代支持的来源有效性由离线行为测试覆盖。原始请求另作独立扫描，物理 HTTP、集合样本和上游字段计数与审计报告一致。

## 真实题目成本

生成包括 planner、mapping/repair、节点 resolve 和必要 audit，resolve 已包含在生成总数中。独立 Reader 和全局 FinalSelector 每题均为 0。逻辑生成包含缓存回放的预算扣费，与物理 LLM 请求分开。

| 题目 | 逻辑生成 | 其中 resolve | ANN | 集合评分单位／已完成集合 | 评分上下文 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA 1 | 10 | 2 | 35 | 505 / 505 | 1 |
| HotpotQA 2 | 11 | 5 | 19 | 182 / 182 | 1 |
| HotpotQA 3 | 15 | 7 | 36 | 395 / 395 | 2 |
| PersonaMem 1 | 22 | 3 | 36 | 231 / 231 | 1 |
| PersonaMem 2 | 23 | 5 | 24 | 328 / 328 | 2 |
| PersonaMem 3 | 21 | 6 | 24 | 310 / 310 | 2 |
| 合计 | 102 | 28 | 174 | 1,951 / 1,951 | 9 |

| 题目 | 物理 LLM | embedding HTTP | rerank HTTP | HTTP 200 集合样本 | 观测 reranker 上游请求 | transport 缓存命中 | scorer 内存命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HotpotQA 1 | 10 | 32 | 335 | 505 | 670 | 3 | 1,119 |
| HotpotQA 2 | 10 | 16 | 124 | 182 | 248 | 4 | 438 |
| HotpotQA 3 | 14 | 33 | 248 | 395 | 496 | 4 | 821 |
| PersonaMem 1 | 22 | 30 | 167 | 231 | 334 | 6 | 833 |
| PersonaMem 2 | 23 | 22 | 192 | 328 | 384 | 2 | 528 |
| PersonaMem 3 | 20 | 22 | 185 | 310 | 370 | 3 | 506 |
| 合计 | 99 | 155 | 1,251 | 1,951 | 2,502 | 22 | 4,245 |

生成按操作拆分如下，修复已计入全题生成预算，不是额外赠送调用。

| 题目 | planner | map | map_repair | resolve | audit |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA 1 | 1 | 6 | 0 | 2 | 1 |
| HotpotQA 2 | 1 | 4 | 0 | 5 | 1 |
| HotpotQA 3 | 1 | 5 | 0 | 7 | 2 |
| PersonaMem 1 | 1 | 11 | 5 | 3 | 2 |
| PersonaMem 2 | 1 | 9 | 6 | 5 | 2 |
| PersonaMem 3 | 1 | 9 | 4 | 6 | 1 |
| 合计 | 6 | 44 | 15 | 28 | 9 |

各 scorer 的最后累计快照按上下文汇总一次，再与 Ledger 对照。六题没有失败/未知 rerank HTTP 集合样本，persistent scorer cache 命中均为 0。服务成功响应在 `usage.upstream_requests` 与 `scoring_metadata.upstream_requests` 中都报告 2，两处一致；报告逐响应相加，不乘集合样本数，不重复计入缓存回放或 seeded 历史响应，也不根据 yes/no 协议推断。此字段只描述成功响应观测，未报告的服务内部工作不能虚构为零。1,251 个成功响应均暴露上游字段，无缺失或两处数值不一致；没有 seeded 历史响应。模型前向次数没有显式暴露，记 null / unavailable；它与 HTTP 次数、上游请求和集合样本数是不同口径。

## 验证准备成本与边界

2026-10-10 直接探针确认 LLM、embedding、reranker 可达。第一次 preflight 的 planner 输出 `final_node_id=step2`，实际 slots 为 museum / city，校验器拒绝，正式题目尝试为 0。失败保留在 `outputs/local_terminal_v1/smoke_3plus3`。只补强 planner 提示要求 exact output_slot，未放宽验证或增加题目额度；第二次 preflight 通过。

| 准备阶段 | LLM HTTP | embedding HTTP | rerank HTTP | 观测 reranker 上游请求 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 首次 preflight（失败） | 1 | 0 | 5 | 10 | 无正式题目尝试 |
| 第二次 preflight（通过） | 1 | 0 | 5 | 10 | 与正式六题分开 |
| HotpotQA 派生索引 | 0 | 307 | 0 | 不适用 | 完整 9,811 文档，1,671.75 秒 |
| PersonaMem 派生索引 | 0 | 0 | 0 | 不适用 | 复用已有索引，历史建库成本不计入本轮 |

HotpotQA 建库 transport 缓存命中为 0；原始索引未改动。建库成本独立于每题 ANN 限额。`/models` 元数据请求不属于生成调用。首轮 preflight 失败成本保留，不伪装成零成本或正式题目重试。

未执行全量实验、生产重启、生产缓存清除或 APC off/on 硬件双臂；APC 状态保持 **IMPLEMENTED_NOT_HARDWARE_VERIFIED**。没有准确率提升、总体降本或 APC 提速结论。

## 实际命令

```bash
.venv/bin/python -m pytest tests -q
git diff --check
.venv/bin/python scripts/demo_local_terminal.py --output outputs/local_terminal_v1/demo
.venv/bin/python scripts/smoke_local_terminal.py \
  --config configs/local-terminal.example.json \
  --output outputs/local_terminal_v1/smoke_3plus3_v2
.venv/bin/python -m dagbt.runner status --output outputs/local_terminal_v1/smoke_3plus3_v2
```

smoke 恢复使用同一命令，要求源码、配置和 scope manifest 一致，跳过所有已终态题目；不要在当前 writer 活着时另起一轮。源码或协议改变后必须保留旧失败记录并使用新目录。实际全量启动/恢复入口及新版流程见 `docs/DAGBT_LOCAL_TERMINAL_V1.md`。
