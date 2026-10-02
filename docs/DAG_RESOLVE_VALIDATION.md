# DAG-Resolve 实现验证

验证日期：2026-10-02。方法：`dag_resolve_binding_v1`。

- 新增协议、控制器和 runner 测试：80 项通过。使用仓库实际 tokenizer、脚本化回答和 loopback HTTP，覆盖候选触发、完整绑定查询、原文引用、改选／兜底、下游仅执行一次、来源隔离、容量、预算、缓存、重试和评分标签隔离。
- 全套复跑：`.venv/bin/python -m pytest -q tests`，609 passed，48.55 秒。
- 首次全套执行为 608 passed、1 failed；旧用例 `test_stop_foreground_relative_output_path` 超时。日志与代码显示旧 `dagbt.runner` 在进程启动期间收到停止信号时存在清理竞态，该子进程不导入 `dagresolve`。此用例孤立复跑通过；随后全套通过。本次未修改旧 runner。
- 测试报告含 60 条 NumPy 矩阵运算警告，涉及原检索及新入口。实际结果通过有限值与排序检查；未屏蔽警告。
- `PYTHON=.venv/bin/python bash scripts/smoke.sh` 通过，三个数据集各 1000 题的数据、索引、tokenizer 和原始文件检查成功，无模型调用。
- 新 CLI 的 `check` 对 HotpotQA、2WikiMultiHopQA、MuSiQue 均通过：文档数分别为 9811、6119、11656，索引维度均为 4096；网络调用为 0，未读取评分标签。
- 原始完整清单 57/57、BridgeTree vendor 清单 57/57 的 SHA-256 均一致。该独立完整性检查读取了清单内评分标签的文件字节用于哈希，未将标签用于生成。
- `git diff --check` 与 `compileall` 通过；独立代码复核未发现剩余必需修正。

以上验证实现与协议行为。真实 Qwen／NV 模型服务不可用，尚未运行模型性能实验，也没有新的 EM/F1、修复率或实际推理成本结论。论文级机制对照和完整相关方法适配仍按研究路线执行。
