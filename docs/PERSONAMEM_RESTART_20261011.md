# PersonaMem：上传文件后使用修复版重新运行（2026-10-11）

本次完整部署包为 `dagv2_bt_deploy_20261011_personamem_v4.tar.gz`，包内顶层目录是 `dagv2/`。通过服务器的文件上传功能传入即可，不需要 GitHub 或 `git pull`。以下命令假设文件上传到此前使用的 `~/bt/`，旧包位于 `~/dagbt_runs/20260930_v3/dagv2`。

包内保留 HotpotQA、2WikiMultihopQA、MuSiQue 和 PersonaMem 四套数据；下方命令显式指定 `--datasets personamem`，只运行 PersonaMem。包内不带 `.git`、`.venv`、旧运行结果或操作者本地配置。源码包含 v4 支持复核与本次 PersonaMem 问题输入协议修复。

这是推理和评测，不更新模型参数。仅跑 PersonaMem 时，两臂各执行固定官方 32k 文件的 589 题，共 1,178 个方法任务；模型配置继续使用 DeepSeek-V4-Flash、Qwen3-Embedding-8B、Qwen3-Reranker-8B。

## 本次修复的输入协议

已提供的旧服务响应显示，PersonaMem 的 original planner 返回了 `(a)`，没有返回规划 JSON。旧入口把公开选项和“只返回选项标签”的完整单选题一起送入 planner，这与规划指令冲突。本次在成对入口修正：**original 与 fusion 的规划、检索和节点推理都使用 `user_question`，只有最终 reader 接收完整问题及四个公开选项。**

original 仍使用原有求解流程和 chain reader；fusion 使用 [v4 支持复核与完整闭包选择](SUPPORT_REVIEW_V4.md)。两臂的区别没有因此全部消失。修复位于适配层，不修改冻结的 `package/`、`dagv2/` 或 BT vendor 文件。每题 `diagnostics.memory_scope.question_protocol` 记录 `user_request_reasoning_mcq_reader_v2`，供核对实际运行协议。

旧 fusion 的固定选集 header 失败属于 v3 文档复核协议。新包默认 dependency 分支已使用 v4 支持图补丁复核，不再要求模型输出 v3 固定文档 header；服务故障、拒绝或其他非法输出仍有明确失败/回退规则，不能据此保证线上所有题都会成功。此次输入修复没有修改模型权重，也没有据此宣称准确率提高。

## 1. 停止旧任务，解压到新目录

如果旧任务仍在运行，先在旧目录停止它，并查看状态：

```bash
cd "$HOME/dagbt_runs/20260930_v3/dagv2"
.venv/bin/python -m dagbt.runner stop --output outputs/personamem_restart_20261009
.venv/bin/python -m dagbt.runner status --output outputs/personamem_restart_20261009
```

`stop` 发出停止请求后，等待 `status` 显示 `live_writer: false` 再继续；若仍为 true，稍后再次执行同一 status 命令。保留旧结果目录供比较，不把旧 rows、请求缓存或运行 manifest 复制到新结果目录。

```bash
mkdir -p "$HOME/dagbt_runs/20261011_personamem_v4"
tar -xzf "$HOME/bt/dagv2_bt_deploy_20261011_personamem_v4.tar.gz" \
  -C "$HOME/dagbt_runs/20261011_personamem_v4"
cd "$HOME/dagbt_runs/20261011_personamem_v4/dagv2"
```

这是一个新运行目录，不要把新包覆盖解压到旧 v3 代码目录。如果上传位置不同，只修改 `tar -xzf` 后的 tar 文件路径。

## 2. 复用已有 Python 环境

本次没有新增依赖，可以直接使用旧环境的绝对 Python 路径。**当前工作目录必须保持在刚解压的新 `dagv2/` 下**，这样 `-m dagbt.runner` 加载的是新代码：

```bash
export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
"$DAGBT_PYTHON" -c 'import dagbt.runner; print(dagbt.runner.__file__)'
```

输出应指向 `20261011_personamem_v4/dagv2/dagbt/runner.py`。复用的是解释器与依赖，不是旧运行目录。重新连接终端后，需要重新 `cd` 到新目录并设置 `DAGBT_PYTHON`。

如果旧环境不存在或依赖不完整，可在新目录创建环境，替代上面的解释器设置：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-fusion.txt
export DAGBT_PYTHON="$PWD/.venv/bin/python"
```

## 3. 复制旧派生向量，并做离线预检

旧任务已经完成的派生向量可以复用。仅在新目录首次部署、尚未启动实验时复制：

```bash
if [ -d "$HOME/dagbt_runs/20260930_v3/dagv2/data/derived" ]; then
  mkdir -p data/derived
  cp -a "$HOME/dagbt_runs/20260930_v3/dagv2/data/derived/." data/derived/
fi

"$DAGBT_PYTHON" -m dagbt.runner preflight \
  --config configs/paired.example.json --datasets personamem --offline-preflight
```

预检会根据 embedding 模型、端点、语料及文档顺序身份查找对应索引，再核验完成 manifest、向量 SHA256、形状和归一化状态。匹配的完整索引显示 `ready`；复制目录本身不等于验证通过。缺失或未完成索引显示 `needs_build`，正式启动会继续准备索引；已有完成 manifest 与实际内容不一致则报错，不会默默改用旧 NV 向量。

示例配置沿用包内服务设置。如果旧部署使用了不同端点，在新目录准备自己的 `configs/paired.local.json`，并把后续所有命令的 `--config` 一致替换；不要直接复制旧运行 manifest。embedding 端点或模型变化会改变索引身份，旧缓存可能无法复用。离线预检不访问服务，不能证明模型接口正常。

## 4. 先跑两题烟测

```bash
"$DAGBT_PYTHON" -m dagbt.runner launch \
  --config configs/paired.example.json --datasets personamem \
  --arms original fusion --limit 2 --output outputs/personamem_v4_smoke

"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_v4_smoke
tail -n 80 outputs/personamem_v4_smoke/launcher.log
```

`launch` 会进行线上预检并启动脱离终端的后台进程，不需要 `nohup`。返回 PID 只代表启动，不代表实验通过。烟测应完成 2 题 × 2 臂，检查最后状态及失败数；如仍在 `preparing_index`，表示正在准备索引，尚未进入答题。即使只跑两题，缺缓存时也需为 PersonaMem 语料建完整索引。

烟测完成后的轻量摘要可这样读取，不需要加载巨大的逐题诊断文件：

```bash
"$DAGBT_PYTHON" - <<'PY'
import json
from pathlib import Path
path = Path('outputs/personamem_v4_smoke/personamem/summary.json')
if not path.exists():
    print('尚未产生评分摘要，请先查看 status 和 launcher.log。')
else:
    summary = json.loads(path.read_text())
    for arm in ('original', 'fusion'):
        result = summary['arms'][arm]
        print(arm, '状态=', result['answer_status_counts'],
              'accuracy(%)=', result['all_task_metrics_percent']['accuracy'])
PY
```

两题烟测用于检查实际接口和完整流程，不用于判断方法性能。若有失败，先保留日志排查，不把“终态已完成”当作所有调用成功。

## 5. 使用独立目录跑全量

烟测检查完成后，去掉 `--limit`，并使用全新的全量输出目录：

```bash
"$DAGBT_PYTHON" -m dagbt.runner launch \
  --config configs/paired.example.json --datasets personamem \
  --arms original fusion --output outputs/personamem_v4_full

"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_v4_full
tail -n 80 outputs/personamem_v4_full/launcher.log
```

不能把 smoke 目录扩成 full，也不能恢复旧的 `outputs/personamem_restart_20261009`。旧 v3、本次修复后的 v4、烟测和全量分别保留结果身份。

停止本次全量实验：

```bash
"$DAGBT_PYTHON" -m dagbt.runner stop --output outputs/personamem_v4_full
"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_v4_full
```

等待 `live_writer: false` 后，再执行上面同一条 full `launch` 命令即可继续未完成的题；代码、配置、数据范围和 arms 必须一致。已有成功和失败终态默认都跳过。如果明确要重试失败，在相同命令后加 `--retry-failed`，受配置中的每题最大尝试次数约束，会产生新的服务请求与费用。

## 结果口径与验证边界

全量结束后读取 `outputs/personamem_v4_full/personamem/summary.json`，主要指标为两臂各自的 `all_task_metrics_percent.accuracy`。只有合法单选标签且与正确选项一致才计 1；失败、歧义、多选和带解释的非法答案计 0。另有 `persona_macro_accuracy_percent` 及共同成功子集指标，分母不同，不应混成同一个准确率。

修复后的 original 也改变了 PersonaMem 的推理输入，因此新旧成绩必须标注协议版本，不能混合追加为一次实验。两臂同题比较应同时保留失败率；只比较共同成功题可能掩盖协议可靠性差异。数据按题隔离、未来消息截止和标签仅用于评分的约束继续适用。

本说明中的上线步骤需要在用户服务器执行。没有在这里声称真实服务烟测或全量实验已经通过；离线回归、脚本化 HTTP 接线验证和打包校验的实际结果由本次交付记录报告。

## 本次验证记录

- 完整回归：`python -m pytest -q`，**625 passed, 6 warnings in 44.91s**。六条警告来自既有 NumPy 原版/适配资源一致性测试。
- 冻结完整性：`verify_originals(include_labels=True)` 检查 **57** 个文件，全部通过。相对 reliability v3 基准，`package/`、`dagv2/`、`vendor/`、`original_manifest.json`、`README.md` 均未修改。
- 回归先用脚本化本地服务复现旧原版收到完整 MCQ 后返回 `(a)`、报 `planner_json`，随后验证修复后的两臂真实 worker 均能走完流程；还验证连续题目、异常退出时 Reader 选项绑定恢复及普通 QA 路径兼容。
- 这些验证覆盖程序和协议边界，不等于已在用户的真实模型服务上完成烟测，也不产生新的准确率结论。
