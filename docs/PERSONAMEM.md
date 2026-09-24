# PersonaMem 数据与成对实验协议

当前 `scripts/run_paired.sh` 已将 `personamem` 加入默认数据集，`original` 和 `fusion` 两臂都会运行。这里新增的是 RAG 推理与评测任务，不训练模型参数。历史 `original-dagv2` 分支、原始 57 个文件和冻结 BT vendor 源码均不修改。

## 数据来源和范围

使用与本地 BridgeTree 项目相同的固定官方数据源：

- Hugging Face 数据集：[`bowen-upenn/PersonaMem-v1`](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1)。
- Revision：`fd7c30f071d5c2ee2a211506783be222d7b6002e`。
- 文件：`questions_32k.csv`、`shared_contexts_32k.jsonl`。
- 范围：官方 32k 问题文件全部 **589 题**，包含 **20 个 persona、37 个 shared context**。
- 处理后：**222 个按题可见范围、3,187 条去重记忆**。

这是固定 32k 文件的完整评测，不是 PersonaMem 所有上下文长度版本，也没有复现 BT 仓库已保存的 train/validation/test 划分或声明其 held-out test 成绩。此处的 `32k` 是源数据设置名称，不能据此假设 reader 会塞入 32k token；每个方法仍遵守本实验的检索、上下文和调用预算。

原始下载 URL、源文件 SHA256、处理后公共文件 SHA256 和独立评测标签 SHA256 均记录在 [`data/personamem/manifest.json`](../data/personamem/manifest.json)。部署包已包含处理后的文件，无需运行时下载原 CSV。

| 文件 | 用途 |
|---|---|
| `questions.jsonl`、`questions.json` | 原始用户请求、所有公共答案选项、题目/persona/context 标识、独占截止位置和 scope ID，不含正确选项 |
| `corpus.jsonl`、`corpus.json` | 去重记忆的物理存储池，不代表全库均可检索 |
| `scopes.json` | 每个 scope 严格允许的文档 ID 列表 |
| `evaluation_only.json` | 独立正确选项，仅评分阶段打开 |
| `manifest.json` | 固定版本、统计、哈希和处理协议 |

## 记忆切分与用户、时间隔离

每道题先取其 `shared_context_id` 对应的消息，并执行 **`messages[:end_index]`**。`end_index` 是独占截止位置，不包含该位置及以后的消息。然后调用冻结 BT 的 `messages_to_memories`，使用 `memory_granularity="user_assistant_pair"` 和 `include_system_persona=True`，保留 system persona。

必须先截断再切分。例如，一道题只看到了某个 user 消息、尚未看到后续 assistant 回复，它只能使用这个不完整的最终 pair；不能先对完整对话配对再把未来回复带进来。不完整 pair 与后来完整 pair 具有不同记忆身份。记忆 ID 绑定 context、源消息位置与原文，避免不同 persona 或不同截止位置的内容发生混淆。

记忆正文与 BT 对截断后消息的转换结果逐字一致。为让 DAG 的文档接口保留偏好变化的先后关系，`Document.title` 另外暴露源消息序号范围，例如 `Conversation message indices 2–3 (zero-based chronological observation order; not calendar time)`。这只说明原对话中的观察顺序，不制造日期，也不将序号解释为事件发生时间；它不同于 BT 原生的 source/time 头序列化。Embedding 和 reader 使用同样的 title + newline + text。

Embedding 可以一次编码全部 3,187 条去重记忆，两方法共享同一物理索引。但运行每道题时，提供给原版 dense/archive 和融合 BT 的文档及向量都限制在 `scopes.json[question.scope_id]` 内。检索、reranker、桥接导航、证据映射、引用校验和最终 reader 只能使用当前范围：

- 不能检索另一 persona 或其他 shared context 的记忆。
- 不能检索该题截止位置及以后的消息。
- 不能因全库向量已存在，就对整个物理存储池做在线检索。

每次切题都会重新应用对应范围；物理向量共享只用于减少重复编码，不扩大题目可见信息。

## 公共选项和 reader 协议

两臂接收完全相同的用户请求与四个公共选项 `(a)` 至 `(d)`，均要求根据该用户的可见记忆选择一个答案。选项是输入的一部分，可以进入规划、检索和 reader；正确选项标签只能由评分阶段读取。

`original` 保留原 DAG 规划、dense 检索、求解和 chain reader 流程，在成对运行适配层增加单选输出要求。`fusion` 保留 BT 桥接和依赖选择，默认 reader 仍只读原文。两臂共用相同单选要求，但 reader 的证据组织方式依然不同，因此最终分数是系统级对比，不能单独归因于 BT。

最终回答必须是一个选项标签。严格解析器接受如 `(a)`、`A`、`Answer: (b)` 这样的单个标签表示；多选、歧义、超出选项范围或附带解释的回答记录为 **`invalid_choice`**，按失败和 0 分处理。这比 BT 原生的 first-match 标签提取更严格，结果不能直接等同于不同解析协议下的分数。

所有配置的数据集、方法、问题先达到成功或记录失败的终态，再加载 `evaluation_only.json`。离线预检和生成阶段只校验公共文件，不打开或哈希实际标签文件；标签文件本身的校验延迟到评分阶段。

## 结果指标

PersonaMem 没有 gold supporting-document 标签，因此不计算 gold recall、all-support 或文本 F1。模型和结构诊断仍可保留，但它们不是证据正确性标签。

主要结果位于 `outputs/<run>/personamem/summary.json`：

- `arms.<arm>.all_task_metrics_percent.accuracy`：全部题目的严格单选准确率，失败计 0 分。
- `arms.<arm>.persona_macro_accuracy_percent`：先在各 persona 内计算准确率，再对出现的 persona 等权平均；失败仍计 0 分。
- `arms.<arm>.failure_rate`、`answer_status_counts`：包括 `invalid_choice` 在内的失败统计。
- `common_success_metrics_percent` 和 `paired`：共同成功子集与同题配对准确率变化。
- `cost_all_attempts`：所有尝试的累计调用成本；向量构建成本另存。

逐题结果见 `personamem/comparisons.csv`、`comparisons.jsonl`。原三套问答数据继续使用原有 EM/F1 和检索指标，不把四套不同任务强行平均成一个分数。使用 `--limit` 时，persona 宏平均只针对所选题目出现的 persona，不代表完整 20-persona 结果。

## 运行

默认配置继续使用 BT 部署的 DeepSeek-V4-Flash、Qwen3-Embedding-8B 和 Qwen3-Reranker-8B；原版算法不额外调用 reranker。PersonaMem 仅支持当前 `bridgetree` 模型 profile，历史 `paired.legacy.json` 不支持该数据集。

```bash
# 不访问模型服务的文件与配置预检。
bash scripts/run_paired.sh preflight --offline-preflight

# 新数据先试跑两题；首次会编码全部 3,187 条去重记忆。
bash scripts/run_paired.sh --datasets personamem --limit 2 --output outputs/smoke_personamem
bash scripts/run_paired.sh status --output outputs/smoke_personamem

# 只运行 PersonaMem 全部 589 题，每题 original/fusion 各一次。
bash scripts/run_paired.sh --datasets personamem --output outputs/personamem_full

# 运行默认四套数据：原三套各 1,000 题，加 PersonaMem 589 题。
bash scripts/run_paired.sh --output outputs/paired_full_personamem
```

完整默认运行共 3,589 题、7,178 个方法任务，使用新输出目录，不能恢复到旧三数据集实验的 manifest 中。新部署包为 `dagv2_bt_deploy_20260924_personamem.tar.gz`，旧包保留；上传、安装和后台启动见 [SERVER_RUN.md](SERVER_RUN.md)。

## 本次验证

2026-09-24 全量回归 **297 passed, 6 warnings in 35.98s**，默认四数据集离线预检通过。6 条警告来自原版与适配版 archive 的既有 NumPy matmul 差分测试，有限分数和完全等价断言仍通过。原始 57 文件和冻结 BT 57 文件的 SHA256 均未改变。

真实 spawned original/fusion workers 在本机模拟模型接口下，共享 3,187 文档索引，连续处理同一对话的 cutoff 182 与 169 两题。测试故意提高未来记忆的向量匹配分数，确认缩小可见范围后两臂仍无越界候选或选集；同时检查单选解析、标签读取闸门、persona 宏平均和无支持标签时的指标行为。

这些测试只验证数据范围、隔离、协议和运行机制，不构成真实模型准确率或收益结论，也不表示目标 Linux 服务器已经实测。实际服务是否可达、模型输出是否符合单选协议，需在目标机器试跑确认。
