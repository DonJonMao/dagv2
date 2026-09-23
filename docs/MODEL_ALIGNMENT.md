# 与 BridgeTree 当前模型部署对齐

2026-09-23，按用户要求修改成对实验入口的共同模型配置。BT 原仓库及本仓库原始 57 个文件均不修改。`original-dagv2` 是历史源码快照；通过当前 `scripts/run_paired.sh` 运行的 `original` 与 `fusion` 使用相同的新模型。

## 模型与来源

配置来自 BT 的 `evidence_bridge.yaml → chain_service_fixed.yaml → chain_full.yaml → default.yaml`，非旧的基础模型猜测。来源文件 SHA256 和非敏感模型字段见 `configs/bridgetree_models_source.json`。

| 用途 | 当前配置 | 服务 |
|---|---|---|
| 规划、节点推理、融合证据映射/审计、最终回答 | `deepseek-v4-flash` | `http://111.19.156.30:8006/v1/chat/completions` |
| 文档和查询向量 | `qwen3-embedding-8b` | `http://111.19.156.74:8001/v1/embeddings` |
| BT 集合相关性评分 | 部署声明 `Qwen3-Reranker-8B` | `http://111.19.156.74:8002/rerank` |

reranker 的 BT 请求配置实际 `model=""`，表示请求中省略 model，使用该专用服务默认模型。不能擅自把部署声明写成服务未确认的 wire model。保留 pointwise、unit_interval、8192 输入长度声明和每个物理请求最多 4 条文档；完整模型权重哈希并未独立验证。

两个方法共享模型及服务配置。`original` 保留原检索/求解/选择流程，不额外添加 reranker；`fusion` 的 BT 搜索使用它。检索提示仍适用于事实问答，不套用 PersonaMem 的个人偏好措辞。任务预算沿用 DAG/融合实验设置，不把 BT 的偏好问答任务长度限制照搬到百科多跳问答。

## 索引和接口兼容

- 新 embedding 的维度即使也是 4096，也不能与 NV 向量混用。所有 27,586 篇原文按原顺序重新编码，输入为 reader 同样的 title + newline + text，不添加 query instruction，并做 L2 归一化。
- 独立索引位于 `data/derived/<identity>/<dataset>/`，身份包含模型、端点、语料、文档顺序及原文序列化。完成时保存实际维度、归一化状态和向量 SHA256，坏索引不会回退到 NV。
- 文档编码按批缓存，重启重放成功响应，只有完整索引才发布 manifest。两方法共用同一派生索引；运行目录的 `index_artifacts.json` 冻结具体向量身份，重跑禁止混用新旧空间。
- 原版 top-50 archive 算法仅将固定 4096 维校验改为匹配当前索引；查询构造、排序、并集及来源记录保持一致。原文件保留，资源适配位于 `dagbt/resources.py`。
- DeepSeek 使用 chat messages。原版节点的 completion 调用由适配层恢复原消息后发送 chat，不把 Qwen 模板或 stop token 发给 DeepSeek。JSON schema 转成明确的输出格式指令，原有 JSON/引用验证仍执行；这不是服务端 guided decoding，真实输出合规性需实测。
- provider 参数与 BT 一致，目前为空，温度为 0。保留原 DAG v6 的单次节点求解流程，不注入假的 thinking 内容，也不据此宣称远端模型内部思考已被强制关闭。

## Token 与凭据

BT 本身使用 `regex_word_or_punctuation_v1` 估算，而非本地 DeepSeek tokenizer。新配置复用同一估算，并记录 `token_count_is_estimate=true`；schema 指令计入估算，服务返回的实际 usage 另存。该预算不构成真实 DeepSeek token 数的精确上界。历史配置仍使用原包 tokenizer，不把它冒充 DeepSeek tokenizer。

凭据优先来自 `DAG_LLM_API_KEY`（或 BT 的 `BRIDGETREE_CHAT_API_KEY`），embedding/reranker 使用对应 DAG 环境变量。可选的 `configs/credentials.local.json` 被 Git 忽略，只在凭据绑定的完整端点与当前配置完全一致时读取；不进入配置快照或请求日志。迁移时只复制匹配 BT 生成服务的本地凭据，不修改 BT 文件。

## 启动

默认配置已经指向 BT 服务：

```bash
# 检查依赖、语料身份、模型配置及派生索引状态，不发送模型请求。
bash scripts/run_paired.sh preflight --offline-preflight

# 仅预建/恢复语料向量，不进行测试题推理。
bash scripts/run_paired.sh prepare-index

# 一键后台：先检查服务，再自动准备缺失索引，然后逐题新旧各跑一次。
bash scripts/run_paired.sh --output outputs/paired_bt_models
```

索引开销保存在该输出目录的 `index_build/<dataset>/`，与题内检索预算分开。新输出目录也可直接复用已经完成且身份一致的索引。旧配置可显式指定 `--config configs/paired.legacy.json`；现有旧输出目录不能用于新模型实验。

## 当前验证边界

本机通过现有 HTTP 代理访问模型服务返回 502，关闭代理的 embedding/reranker HTTP 探测超时。因此没有用真实服务重建语料，也没有开始真实 DeepSeek 准确率实验。配置信息来自 BT 的有效配置，不能将连通性失败解释成模型身份已获在线验证。

本地协议测试使用真实原版/融合流水线及真实包内语料，外部模型由明确的脚本服务替代；它验证新模型接口与索引迁移机制，不能作为效果或成本结论。

最终统一回归：**244 passed in 29.39s**；离线预检通过，三套真实服务索引均明确为 `needs_build`。原版 57 文件和 BT 57 源码快照校验无差异，凭据文件不被 Git 跟踪。NumPy 在原版/适配版 archive 差分测试中产生 6 条 matmul 警告，测试另行确认所有分数有限且两者请求、排名、并集、trace 完全一致。

新增完整协议测试对 HotpotQA 的 9,811 篇真实原文建立 8 维脚本向量（307 批请求），特意与旧 4096 维索引区分，原版与融合版同题均 `ok`。断言新索引只建一次、两臂身份相同、原 NV SHA 不变、全部生成走 chat、reranker 不带空 model 且每批不超过 4 条、实际 usage 与估算标志分别记录。8 维只是测试 fixture，不代表实际 Qwen3 服务维度。

验证产物：`outputs/verification/model_alignment/pytest.txt`、`integrity.json`、`system_summary.json`、`scripted_pair/`；预检为 `outputs/verification/model_alignment_preflight.json`。脚本成对产物带 `SCRIPTED_BT_PROFILE_ONLY.json` 标记。
