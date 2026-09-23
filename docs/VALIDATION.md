# 验证记录

日期：2026-09-23。本记录区分源码保留、机制测试、真实流水线协议测试与真实模型效果。

## 原始材料与运行环境

- 原包 57 个文件逐项与 `original_manifest.json`、原始 `.tar.gz` 的同路径文件比对，SHA256 全部一致。
- `original-dagv2` 分支：`87d09da`。三个大向量数组保存在本地并纳入校验，未提交到 Git；其余 54 个原始文件均已提交，包含 tokenizer 的 `config.json`。
- `vendor/bridgetree/` 的 57 个文件与冻结时最新 BT 工作区逐字节一致，再次与当前 BT 源文件比对无差异。快照包括尚未提交到 BT Git HEAD 的更新；没有修改 BT 原目录。
- 两份 day2 设计原文副本与来源文件哈希一致，见 `docs/design_reference/manifest.json`。
- 本地使用 Python 3.11.15、NumPy 2.2.6、Transformers 4.57.6、PyYAML 6.0.3、Jinja2 3.1.6。Jinja2 是实际渲染包内 chat template 所需依赖。
- 完整性明细保存在 `outputs/verification/integrity.json`。本地大文件和验证产物不提交到 Git。

## 验证命令

从仓库根目录运行：

```bash
.venv/bin/python -m pytest -q
PYTHON="$PWD/.venv/bin/python" bash scripts/smoke.sh
bash scripts/run_paired.sh preflight --config configs/paired.example.json --offline-preflight
```

原始 smoke 已通过三套数据的 prepare-only 检查：HotpotQA、2WikiMultihopQA、MuSiQue 各 1,000 题，模型调用为零。离线预检另外检查融合依赖、本地 tokenizer 模板和数据/源码身份；它不是模型服务兼容性测试。

最终统一回归：**199 passed in 17.30s**。实际命令增加了 `--basetemp outputs/verification/pytest_final` 以保留测试产物，完整输出见 `outputs/verification/pytest.txt`。原始 smoke 输出见 `outputs/verification/original_smoke.txt`；离线预检见 `outputs/verification/offline_preflight.json`，通过 54 个非标签原文件、三套各 1,000 题及 73 个新增源码/vendor 身份检查。其余 3 个标签文件不会被预检打开，完整性核验与评分阶段另行检查。

## 同题双流水线协议测试

`tests/test_system_http.py` 对同一道真实包内 HotpotQA 问题运行原版和融合版。两者均使用实际 spawned worker、原始数据准备、真实 tokenizer、9,811 篇完整语料和 4,096 维索引；融合版使用实际最新版 BT。仅外部模型 HTTP 服务由本地脚本响应替代，未 monkeypatch 检索、支持编译、选择或原 DAG 算法。

测试包含生成、落盘、标签隔离、评分及同题比较输出；脚本模型不读取也不接收答案标签。其答案是任意固定标记 `SCRIPTED_HTTP_PROTOCOL_FIXTURE_NOT_A_MODEL_PREDICTION`，不能把对应 EM/F1 当作算法表现。测试为了缩短时间，将该题融合 ANN/set-score 上限设为 12/64，正式默认值仍为 36/512。

保存了两个成对运行：

| 运行 | 方法状态 | 融合家族 ANN / set-score / reasoning LLM / reader | 实际 HTTP 尝试 |
|---|---|---|---|
| 原版 + 默认融合 | 两臂均 `ok` | 10 / 64 / 4 / 1 | 原版 4、融合 56；另有预检 5 |
| 原版 + 无评分代理控制 | 两臂均 `ok` | 10 / 0 / 4 / 1 | 原版 4、控制 15；reranker 为 0 |

两次使用同一问题 ID `5abe953b5542993f32c2a170`。产物分别保存在 `outputs/verification/scripted_pairs/fusion/` 与 `fusion_proxy_free/`，包括权威结果行、原始请求响应、逐模块事件、比较 JSONL/CSV、汇总及 `SCRIPTED_TEST_ONLY.json`。汇总入口为 `outputs/verification/scripted_pairs/verification_summary.json`。预算数字和调用数只是这两个脚本服务 fixture 的执行记录，不能估计正式全量任务成本。

## 可验证的行为及解释边界

- AND 前提缺失、OR 替代支持、局部冲突失效、父结论/适用范围版本、冲突明确解决、必要 resolution 原文闭包。
- unknown 父节点不用于事实绑定；证据触发细化不改写原问题或终端需求；审计后具体缺口驱动真实回查。
- 全文分块与精确引用、共享来源去重、实际 tokenizer 输入及输出预留、raw-only reader 与有完整来源的 chain reader。
- 共享 ANN/set-score/LLM/reader 预算、缓存重放原重试占用、审计修复不能占用 flat selector 的最终预留。
- 导航强制闭包的真实首次发现来源、额外 token/文档约束与 OR 选择变化；无评分代理控制的零 reranker、需求轮转及真实多跳条件提案。
- 配对 bootstrap 保留同题四臂关系、固定抽样可复现、少于两题明确无法估计；已测量的零调用与缺失计量信息分开。
- 原版包装器与原 `e.work` 计算路径对照；后台启动/状态/停止、超时与坏题隔离、显式重试、恢复身份、标签延后加载。
- `fusion_no_conditions` 只关闭独立条件审计阶段，仍保留基础实体、条件、时间和引用检查。程序结构约束不是自然语言蕴涵证明。

当前默认模型端口 8019/8020/8021 均未运行，见 `outputs/verification/default_services.json`。示例配置的 reranker 名是占位符；尚未运行真实模型的完整对比实验，也没有关于融合收益的结果。配置实际 Qwen、NV-Embed-v2 和 pointwise reranker 服务后，应先用独立输出目录运行 `--limit 2`，确认真实服务协议，再启动全量成对实验。
