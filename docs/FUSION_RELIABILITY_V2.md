# DAG + BridgeTree：证据协议可靠性 v2

日期：2026-09-27。融合方法诊断版本为 `dagbt_fusion_reliability_v2`。本轮将 BridgeTree `16809bd64b6b1edc93e719489cec45ebd19db6f9` 中的可靠性修订适配到 DAG 的 planner、证据映射、节点求解和融合选择流程。它是冻结模型的推理评测，没有参数训练。

默认仍运行 HotpotQA、2WikiMultihopQA、MuSiQue 各 1,000 题和 PersonaMem-v1 32k 全部 589 题，`original`、`fusion` 各一次，共 7,178 个方法任务。PersonaMem 按题的 persona 与时间边界、公共选项及评分协议保持现有约定。

## 为什么修改，以及本次改动的边界

上游 Evidence BridgeTree v1 的服务器错误摘要记载：283 个任务中 53 成功、230 失败；引用不匹配 102、JSON 开头解析失败 49、输入超预算 28、引用超过 400 字符 15、跨需求引用 13。该轮完整服务器原始响应尚未进入本地审计材料，这些计数用于定位错误类别，不能当成本仓库 DAG 融合实验的失败统计，也不能宣称本轮已经逐例重放并消除了这些真实失败。

本次适配区分接口可靠性和证据可见范围的改变。用 ID 指向原文、严格解析和局部修复主要减少接口错误；允许部分有效映射继续、按预算筛选完整证据记录，会改变模型看到的内容。因此 v2 是需要独立运行身份和实验结果的新版本，不能声称只修工程问题、完全不影响方法行为。

原始 57 个文件、`original-dagv2` 历史代码和 `vendor/bridgetree/` 快照不以此次修订覆盖。默认 `original` 臂保留现有算法和 reader 协议；模型部署与向量仍同融合臂共享。本轮没有把上游整个 Preference-RAG executor 替换进来，DAG 的依赖图、支持方案、约束和题内预算仍由本仓库实现。

## 来源引用与局部恢复

- 代码从可见原文构造至多 400 字符的 source spans，保留文档、字符区间、来源角色和文本身份；PersonaMem 的权威说话人边界保持可追溯。
- 映射请求展示短 source ID 和映射 unit；模型返回 `span_ids` 与所属 `node_id`，代码回取精确原文。模型不再负责逐字复写 quote 或计算字符偏移。
- 同一原文支持多个 DAG 节点时分别建立对应 assessment；不存在、不可见、跨文档或错误关联的 ID 仍拒绝。来源准确只证明出处，不能证明 claim 或 support 的语义正确。
- 有效 assessment 保留，失败单元单独修复；失败 assessment 保留待修复位置，重复已有合法记录不能掩盖失败内容。明确输出截断可以拆小映射批次。未完成部分记为 unavailable/未完整映射，不当作无关，也不补写虚构证据。
- 保留独立判断与实际来源的关系；一个判断引用多段原文、同一句被强制切片或重叠出处，不自动产生多个独立推理前提。
- 没有合法的最终输出仍记录失败，不以无效 JSON、伪造支持边或静默 dense 回退冒充成功。

## 响应协议、预算和费用

`fusion.response_format` 默认为 `plain`，可固定为 `json_object` 或 `json_schema`。plain/object 模式中的 schema 指令不是服务端 constrained decoding 保证；json_schema 按配置发送结构化字段，服务拒绝时不会静默换协议。此设置只作用于融合推理模块，不向 `original` 或最终 reader 传播证据格式约束。

严格解析器支持完整 BOM、完整代码围栏及唯一完整顶层对象的确定包装；拒绝重复键、非有限数值、竞争对象和不完整 JSON，不自动补括号或缺失字段。保留原始输出、`finish_reason`、refusal、usage、response ID 与实际协议；只有明确的 `finish_reason=length` 记作已确认输出截断，缺失元数据不冒充正常结束。

默认 planner/resolve/audit/select 的一次操作最多局部修复 2 次；映射按每个 unit/node 最多初始 1 次加修复 2 次约束，拆批不会重置该计数。整题共用 6 次修复额度（`max_repairs_per_request=2`、`json_repairs=6`）；planning、map、resolve、audit、selection、repair 和 rebatch 共用每题每方法每 attempt 的 `llm_calls=24` 额度，最终 reader 按自身额度计费。这里保留 DAG 的保守 LLM attempt 计费：每次物理 HTTP 尝试（含重试）占用一次 LLM 余额，成功缓存重放也按原已记尝试数扣额度；因此不同于上游单纯的 24 次逻辑调用预算，实际可用逻辑调用数可能更少。局部修复和重分批同样占用余额，不会重置题内总额度。HTTP 另受 `max_identical_attempts` 限制；`call_events.jsonl` 与成本报告单独记录逻辑调用、缓存和物理 HTTP，不能混为一个成本数字。

融合证据输入采用 `regex_or_utf8_bytes_div3_v2` 估算，取正则分词与 UTF-8 字节数除以 3 向上取整的较大值，并计入 schema、输出预留和默认 `input_margin=256`。默认上下文 16,384、映射输入批额度 6,144、单次证据输出预留 4,096。该估算缓解连续中文和长 ID 的低估，**不是实际服务 tokenizer 的上界保证**；原 reader/embedding/reranker 的计量不会因此自动变成服务端 tokenizer 真值。

长输入按完整证据/候选记录组成预算内窗口，固定 query、需求、必要结构不能通过切断 JSON 字符串来规避预算。实际可见 ID 限制模型可引用的范围，被省略记录保留在诊断中。父结论只有在当前窗口中存在完整、合格的前提闭包时才可作为已知输入；审计中的替代支持与其全部来源及父路线一起保留。发生预算省略、尚未映射和没有检索到证据是不同状态；最终 reader 仍使用所选原文并执行既有可行性检查，不把证据窗口裁剪等同于任意截断 reader 原文。

## 启动前的真实协议检查

默认 BT profile 的线上启动除 `/models` 和 reranker 探测外，再发送 **一次真实融合 planner 请求**：固定虚构问题，不读取数据集标签或选项；使用正式 Reasoner、planner schema 和 validator，只允许一次逻辑请求、不做格式修复、不切换协议，物理 HTTP 仍按有限重试策略处理。解析、schema、输入预算或服务失败都会阻止后台全量启动。

探测只证明这一次 planner 请求可用，不证明完整 map/resolve/select/reader 或所有样本兼容。原始请求与响应、校验结果、metadata、ledger 和成本在 `preflight_calls/planner-*/`，其中 `report.json` 的 `experiment_task=false`，不进入任何题目的预算或方法成本。后台只复用 5 分钟内配置、源码、数据范围和 arms 一致、且 planner 探测通过的 launch 报告。

显式 `--offline-preflight` 跳过所有端点与协议预检。它仅用于离线检查或明确跳过启动探测，**不会令 run/launch 离线，也不代表模型服务已验证**。正常服务器启动不要加入这个参数。

## 结果分组与错误分析

每题 `diagnostics.reliability.version=dagbt_fusion_reliability_v2`，`cohort` 为：

| cohort | 含义 |
|---|---|
| `normal` | 没有记录输入裁剪或部分映射 |
| `truncated` | 证据输入发生预算筛选 |
| `partially_mapped` | 存在未完整映射的候选/单元 |
| `truncated_and_partially_mapped` | 同时发生以上两种情况 |
| `unknown` | 旧结果或未提供相应诊断；不冒充 normal |

这些是运行可靠性状态，不是证据真实充分性。失败题可从最新 `fusion_partial.json` 恢复已落盘的 reliability；错误日志和旧 attempt 仍保留。

`summary.json` 每个数据集、方法的 `reliability.completion_cohorts` 只统计当前权威结果中成功的任务，给出 `tasks`、`correct`、PersonaMem 的 `accuracy` 或其他数据集的 `em`，以及 `cost_latest_attempt`。`failed_tasks` 和 `failure_cost_latest_attempt` 单列失败及其当前尝试成本。成本字段分别记录 observed/missing；缺失 usage 仍保留未知/下界语义。

成功重试不会把旧失败追加成另一道题，也不会把旧 attempt 成本塞入当前成功 cohort；原 `cost_all_attempts` 继续统计全部真实尝试。各 cohort 的问题集合不同，不能根据它们之间的准确率差直接归因“截断更好”或“部分映射无损”。总体结果仍须结合全任务准确率、失败率、共同成功比较和总成本。

## 部署与验证

本次新包为 `dagv2_bt_deploy_20260927_reliability_v2.tar.gz`。解压到新目录，使用新输出目录，例如 `outputs/paired_full_reliability_v2`；不能 resume 旧 PersonaMem 部署包的运行身份。数据集与模型配置不变时，派生索引仍按既有身份校验后复用；不得混写不同版本的任务结果。

上传、启动、停止和重试命令见 [SERVER_RUN.md](SERVER_RUN.md)，字段与实验对照见 [PAIRED_RUNNER.md](PAIRED_RUNNER.md)。本地测试使用受控模型/网络替身以及实际执行器、文件与进程路径；来源校验、局部恢复、预算窗口、cohort 聚合和启动闸门的测试不代表真实模型准确率提高。发布验证应同时保留完整测试输出、离线预检、原始/vendor 完整性检查与压缩包成员校验；真实模型结果由服务器的新运行产生。

本次最终统一回归：**437 passed，6 条既有 NumPy matmul warnings，39.51 秒**。命令：

```bash
.venv/bin/python -m pytest -q tests --basetemp outputs/verification/reliability_v2/pytest_final
bash scripts/run_paired.sh preflight --offline-preflight
```

原始包 57 个文件与冻结 vendor 57 个文件分别逐文件 SHA256 一致。默认离线预检覆盖四套数据；本地未调用真实模型服务。最终日志保存在 `outputs/verification/reliability_v2/pytest_final.txt`、`integrity.json`、`offline_preflight.json`；部署包逐成员哈希及解压后入口检查结果保存在同目录 `archive_verification.json` 与 `extracted_offline_preflight.json`。这些本机验证产物不进入部署包，包内保留完整测试代码。

