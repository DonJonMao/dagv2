# PersonaMem a48：planner domain 协议补丁（v2）

本补丁用于已部署的 `742ab9172e5a6752691fab6ac14887a1aa65514d` 版本，继续以 a48dc8d 的 `dagbt_residual_memory_v1` 为主方法，并保留 original 对照。补丁包名为 **`dagv2_personamem_a48_v2_patch.tar.gz`**。这是小型增量补丁，不替换数据、模型或派生索引，也不切换到旧 fusion 或 v4 支持复核方法。

## 这次启动失败说明什么

用户遇到的错误为：

```text
ProtocolError: Domain must be finite values or an interval
```

错误发生在 `launch` 的线上 planner 协议预检阶段，该次后台实验尚未启动。旧 planner schema 没有充分说明变量声明和 `domain` 的结构，提示也缺少未知域的明确写法；验证器却要求 domain 是有限值数组或数值区间，因此模型输出容易与实际契约不一致。

这个报错不能单独证明真实响应写了 `domain: null`。也可能是字符串、标量或其他不合法结构，具体字段和值必须查看保留的原始响应。补丁补全契约和错误定位，不假定或改写尚未确认的线上响应。

## 补丁改变什么

- **未知或不限制的域省略 `domain`。** 不用 null、字符串、标量或空数组代替未知。开放实体示例为 `{"type":"entity","demand":"place","open":true}`。
- **已知有限域使用数组，只有 number 变量允许区间对象。** 区间写明 `lower`、`upper`，端点为数值或 null；端点 null 表示无界，不等于允许整个 domain 为 null。数组元素的类型及区间语义仍由原验证器检查。
- **合法域与验收门槛继续分开。** domain 描述可能取值，门槛留在比较表达式里；不会通过收窄 domain 排除可能不合格的观测，也不为未知规则猜测阈值。剩余求值、证书条件和终端输出语义保持不变。
- **仅 typed joint/residual planner 预检增加有界格式修复。** 默认先请求一次，非法协议最多再修复 2 次，共最多 3 次逻辑请求；实际上限受 `max_repairs_per_request` 与 `json_repairs` 共同约束。其他 planner 预检仍保留原来的单请求规则。模型没有被允许绕过来源、类型或表达式验证。

预检复用运行时的 planner schema、验证器和修复协议。原始 JSON、验证错误及后续修复响应均保留，不会把原始请求结果静默“修正”为成功记录。`report.json` 区分 `initial_response_valid`、`repair_calls` 和 `logical_calls`，修复后的通过不伪装成首轮通过。

预检使用独立 Ledger 记录预算、HTTP 尝试与费用，不占用任何正式题目的预算。HTTP 重试仍按物理请求计数，所以逻辑请求数不等于物理请求数。预检不读取评分标签，也不构成准确率实验；通过仅说明这次探针完成了 planner 协议校验，不保证后续阅读、求值或全量任务成功。

## 上传并安装

将 `dagv2_personamem_a48_v2_patch.tar.gz` 上传到服务器 `~/bt/`。补丁包含 `apply_patch.py` 和 `payload/`，按下面的独立路径解压：

```bash
mkdir -p "$HOME/dagbt_patches/personamem_a48_v2"
tar -xzf "$HOME/bt/dagv2_personamem_a48_v2_patch.tar.gz" \
  -C "$HOME/dagbt_patches/personamem_a48_v2"

export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
"$DAGBT_PYTHON" "$HOME/dagbt_patches/personamem_a48_v2/apply_patch.py" \
  --target "$HOME/dagbt_runs/20261011_personamem_a48/dagv2"
```

如果解释器位于其他位置，只修改 `DAGBT_PYTHON`。安装器要求目标是上述 742ab91 部署身份，检查原部署文件哈希和没有活跃 writer，备份被替换文件并更新部署元数据 `DEPLOYMENT.json`。如检查拒绝，按它报告的版本、文件或活跃运行问题处理，不手工覆盖 payload 来绕过检查。此次预检失败本身没有启动 writer，但目标目录中其他运行仍须停止后才能安装；安装期间不要另行启动实验。

安装后进入目标代码目录，确认加载路径：

```bash
cd "$HOME/dagbt_runs/20261011_personamem_a48/dagv2"
"$DAGBT_PYTHON" -c 'import dagbt.runner; print(dagbt.runner.__file__)'
```

该路径应位于 `20261011_personamem_a48/dagv2/`。无需再次复制派生索引，也不要把旧 rows、attempts 或运行 manifest 复制进新输出目录。

## 使用新目录重跑两题

```bash
"$DAGBT_PYTHON" -m dagbt.runner launch \
  --config configs/residual-memory.example.json --datasets personamem \
  --arms original dagbt_residual_memory_v1 \
  --limit 2 --output outputs/personamem_a48_v2_smoke

"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_a48_v2_smoke
tail -n 80 outputs/personamem_a48_v2_smoke/launcher.log
```

代码和 planner 契约已变化，必须使用新输出目录。不要对旧 `outputs/personamem_a48_smoke` 直接续跑，也不要用 `--offline-preflight` 绕过这次正在修复的线上协议检查。

若预检再次失败，查看新目录的 `preflight_calls/planner-*/report.json` 及原始请求响应。报告会记录具体变量/字段的验证错误、首轮是否有效、修复次数及成本。无需先加载巨大的正式逐题结果来诊断这一阶段。

## 烟测确认后启动全量

两题共四个方法任务完成后，检查失败状态；烟测通过仍不代表方法准确率。全量改用另一个新目录，并去掉 `--limit`：

```bash
"$DAGBT_PYTHON" -m dagbt.runner launch \
  --config configs/residual-memory.example.json --datasets personamem \
  --arms original dagbt_residual_memory_v1 \
  --output outputs/personamem_a48_v2_full

"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_a48_v2_full
tail -n 80 outputs/personamem_a48_v2_full/launcher.log
```

这仍是两臂各 589 题，共 1,178 个方法任务。residual 没有独立 Reader、FinalSelector 或 reranker；终端同次输出 `semantic_answer` 与 `final_prediction` 的协议不变，original 保留自己的最终 Reader。完整环境、停止和恢复规则见 [a48 部署说明](PERSONAMEM_A48_RESTART_20261011.md)，尤其默认 `max_question_attempts=1` 的限制。

本补丁的最终文件哈希、修复提交和离线验证结果由交付记录提供。此处不预先宣称用户服务器的线上预检、烟测或全量运行已经通过。
