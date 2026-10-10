# local_terminal_v1 验证记录（2026-10-10）

基准为 `f6a172123ca05ec4fdc0463b94cd3a91253bfbbb`，实现提交为 `b0e86fb`，分支为 `feat/local-scoring-terminal-answer`。本轮新增源码提交尚未推送。既有未跟踪的研究资料、压缩包和 tmp 内容未纳入提交。

三项算法修改已有实现和离线行为证据。真实 3＋3 smoke 尚未完成，目标保持进行中；本文的真实运行状态需在全部六题终态后更新。

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

## 真实 smoke 状态

固定范围为 canonical HotpotQA 前三题＋PersonaMem 前三题，完整语料，每题最多一次尝试，不加载评测标签。2026-10-10 直接服务探针确认 LLM、embedding、reranker 均可达。

第一次 preflight 输出 `final_node_id=step2`，而实际 slots 是 museum / city，校验器拒绝。此轮没有正式题目尝试，失败保留在私有 `outputs/local_terminal_v1/smoke_3plus3`。只补强 planner 提示要求 exact output_slot，未放宽验证或增加题目额度。

第二次 preflight 已通过，私有目录为 `outputs/local_terminal_v1/smoke_3plus3_v2`。正在补齐完整 HotpotQA 的 9,811 文档 Qwen embedding 派生索引；PersonaMem 的派生索引已存在。源索引保持不变，索引预备成本独立于每题的 36 ANN 限额。尚无六题最终结果或真实题目成本可报告。

真实成本最终应分别报告生成逻辑调用、节点 resolve、ANN、集合评分、rerank HTTP 尝试、成功完整集合样本、缓存命中，以及服务明确暴露的上游请求/前向。服务未暴露上游次数时记 null / unavailable，不根据 yes/no 协议猜测前向次数；不将重试、集合评分单位和物理请求混为一个计数。

未执行：全量实验、生产重启、生产缓存清除、APC off/on 硬件双臂。APC 状态保持 **IMPLEMENTED_NOT_HARDWARE_VERIFIED**。

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
