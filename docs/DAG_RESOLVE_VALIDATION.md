# DAG-Resolve 实现验证

验证日期：2026-10-02。方法：`dag_resolve_binding_v1`。

## 初版验证（0e63c76）

- 新增协议、控制器和 runner 测试：80 项通过。使用仓库实际 tokenizer、脚本化回答和 loopback HTTP，覆盖候选触发、完整绑定查询、原文引用、改选／兜底、下游仅执行一次、来源隔离、容量、预算、缓存、重试和评分标签隔离。
- 全套复跑：`.venv/bin/python -m pytest -q tests`，609 passed，48.55 秒。
- 首次全套执行为 608 passed、1 failed；旧用例 `test_stop_foreground_relative_output_path` 超时。日志与代码显示旧 `dagbt.runner` 在进程启动期间收到停止信号时存在清理竞态，该子进程不导入 `dagresolve`。此用例孤立复跑通过；随后全套通过。本次未修改旧 runner。
- 测试报告含 60 条 NumPy 矩阵运算警告，涉及原检索及新入口。实际结果通过有限值与排序检查；未屏蔽警告。
- `PYTHON=.venv/bin/python bash scripts/smoke.sh` 通过，三个数据集各 1000 题的数据、索引、tokenizer 和原始文件检查成功，无模型调用。
- 新 CLI 的 `check` 对 HotpotQA、2WikiMultiHopQA、MuSiQue 均通过：文档数分别为 9811、6119、11656，索引维度均为 4096；网络调用为 0，未读取评分标签。
- 原始完整清单 57/57、BridgeTree vendor 清单 57/57 的 SHA-256 均一致。该独立完整性检查读取了清单内评分标签的文件字节用于哈希，未将标签用于生成。
- `git diff --check` 与 `compileall` 通过；独立代码复核未发现剩余必需修正。

以上验证实现与协议行为。真实 Qwen／NV 模型服务不可用，尚未运行模型性能实验，也没有新的 EM/F1、修复率或实际推理成本结论。论文级机制对照和完整相关方法适配仍按研究路线执行。

## 协议与 Reader 指标修订

用户反例揭示初版将无效引用降为 unknown，可能在另一候选 supported 时改选。这属于协议错误，初版通过的测试与复核没有覆盖该组合。

- 协议有效性与语义状态分开：引用、响应结构、必要条件覆盖或多值标志校验失败，整次可选判读退出，原样保留 primary 普通答案与来源，trace 记录 failed 与具体原因。合法 unknown 与 supported 的改选规则保持。
- 新增 19 个回归用例；新入口三个测试文件共 99 passed，5.72 秒。覆盖共同导演情况下只破坏 primary 引用、替代项无效时不清空被反驳 primary、unknown 的坏引用、漏判必要条件、提议无效不触发，以及 selector 与 Reader 支持证据范围不同。
- 全套验证：628 passed，69 条 NumPy 运算警告，51.10 秒；未屏蔽警告。
- Reader 指标保留全部评价题作为分母，独立于答案成功状态；缺失面板召回计零，单报可用数量和两种裁剪率分母。测试包含 selector R@20/all@20=100%、Reader support recall=50%/all support=0% 的反例及 MuSiQue ID 归一化。
- 原始清单 57/57 与 vendor 清单 57/57 哈希一致；compileall、git diff --check 与独立复核通过。
- 固定首个桥节点的干预范围保持原设计。文档要求自然错误位置、正确替代项可达性及触发覆盖率的审计，尚无这些实证统计，不能据此宣称终态敏感调度有效。

本轮仍无真实模型性能结果；新增证据指标并不产生新的性能结论。
