# DAG v2 方法卡（2026-09-21）

## 方法定义

**DAG v2 = no-thinking + 全语料节点检索 + Reader 节点链注入。**
规划器把多跳问题拆成 1–6 个检索任务的依赖 DAG（只用原问题、不答题、不猜实体）；
档案池为原问题与各 DAG 步提问的稠密检索 top50 稳定并集（NV-Embed-v2，全语料）；
每个已接地节点用父节点答案替换占位符后**在全语料上**检索 top50 并入池、取 top20 面板，
单次 guided-JSON 调用作答（answer + sources 布尔数组，强制勾选来源，512 tokens，无 thinking）；
按 k=5/10/20 预算枚举节点证明闭包并选择来源文档；Reader 携带 top20 原文**以及全部已解析
节点答案链**（依赖序：slot/接地问题/答案/来源文档）出最终答案（1024 tokens，无 thinking）。
运行命名约定见源仓库 `local_runs/METHOD_NAMING.md`（DAG v2 曾称 v6）。

## 全量成绩（各数据集本地 1000 题子集，Qwen3.8-27B-AWQ-INT4 + NV-Embed-v2）

| 数据集 | F1 | EM | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|
| HotpotQA | 79.83 | 67.00 | 98.00 | 99.00 | 99.25 |
| 2WikiMultihopQA | 80.34 | 72.90 | 97.03 | 97.70 | 97.90 |
| MuSiQue | 64.66 | 54.50 | 81.91 | 87.48 | 89.76 |

完整逐指标表见 `results/RESULTS_{hotpotqa,2wikimultihopqa,musique}.md`。

## 与 DAG v1 对照（F1）

| 数据集 | DAG v1（池内检索 + thinking） | DAG v2（本方法） | Δ |
|---|---:|---:|---:|
| HotpotQA | 82.91 | 79.83 | −3.08 |
| 2WikiMultihopQA | 75.94 | 80.34 | +4.40 |
| MuSiQue | 66.86 | 64.66 | −2.20 |

DAG v1：节点检索限制在档案池内、节点 thinking 2048 + 结构输出 512、Reader 不给中间答案。
DAG v2 在 2Wiki 上反超 DAG v1；HotpotQA/MuSiQue 略低但换来全程无 thinking
（节点单调用、Reader 非思考），推理成本显著下降。

## 消融结论摘要

1. **池化修复（全语料节点检索）**：DAG v1 的节点只能在档案池内检索，正确文档未入池时
   节点必然失败；改为全语料 top50 并入池后召回显著改善（R@5 98/97/82），是 DAG v2 的
   收益底座。
2. **answer_type 约束（判负）**：节点级（v4）、节点+Reader 级（v5）answer_type 结构化
   约束两条路线均判负停止，未纳入本方法。
3. **Reader 节点链注入（科学消融 2）**：动机是 no-think Reader 会"背叛"正确的节点链
   （2Wiki 312 共同题上 no-think Reader 正确率 61.5% vs think 70.8%，背叛 39 次 vs 3 次）。
   注入后 2Wiki 对 no-think 基线 +10 F1。但机制披露：终末节点已解析时 **87–99% 的最终
   答案是链尾答案的逐字照抄**（2Wiki 99.4%）；**终末直出消融**（跳过 Reader 直接用链尾）
   为 HotpotQA 71.82 F1（Reader 贡献 **+8.0**）、2Wiki 78.70（**+1.6**）、MuSiQue 57.97
   （**+6.7**）。即最终答案主要由 DAG 链终末节点决定，Reader 贡献格式归一与兜底。
   引用本成绩时必须同时引用终末直出消融。

## 诚实披露

- 生成链路（work/planner/archive/nodes/reader）零标签接触；evaluation_only 标签只在
  1000 行全部生成完毕后、在评分阶段读取（前置断言）；无训练、温度 0；注入链 100%
  模型自生成；索引由纯语料构建（哈希断言）；中间评分只读、无反馈回流。
- 泄露审计结论：无标签泄露。
