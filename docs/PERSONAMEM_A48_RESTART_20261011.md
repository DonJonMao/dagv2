# 基于 a48dc8d 的 PersonaMem 上传部署与重新运行

本次基准提交为 **`a48dc8da586c8f2c9858548dadcebde14aa9d9ed`**，修复工作分支为 `fix/personamem-a48dc8d`。部署包名为 `dagv2_bt_deploy_20261011_personamem_a48.tar.gz`，包内顶层目录为 `dagv2/`。通过服务器文件上传功能，把包传到 `~/bt/` 即可，不需要 GitHub。

本次仅修正 PersonaMem 的问题输入适配，主方法保持 a48dc8d 的 **`dagbt_residual_memory_v1`**。实验保留 `original` 对照，因此每条启动命令都显式指定 `--arms original dagbt_residual_memory_v1`。不要把此前的 v4 支持复核部署包、旧 `fusion` 臂或其启动命令用于这次实验。

## 方法和输入协议

主配置为 `configs/residual-memory.example.json`，其中 `fusion.algorithm_version` 是 `dagbt_residual_memory_v1`。该配置使用 DeepSeek-V4-Flash 和 Qwen3-Embedding-8B，没有 reranker 配置；主方法每题预算为 ANN 36、LLM 24、独立 Reader 0、集合评分 0。

residual 方法按预规划 DAG 进行任务内 dense/条件桥接、联合阅读和剩余求值，由指定终端直接给出最终输出；**无独立 Reader、无全局 FinalSelector，也不恢复 v4 末端支持复核或事后重答**。公开选项只在真正终端输出阶段出现。结果保留该方法的 `budgets={}`，不能用旧 k=5/10/20 Reader 选集指标解释它。完整方法说明见 [DAGBT_RESIDUAL_MEMORY_V1.md](DAGBT_RESIDUAL_MEMORY_V1.md)。

本次适配将 PersonaMem 的 `user_question` 用于两臂的规划和非终端推理，避免把“只返回一个选项标签”的完整 MCQ 指令送入 original planner。original 在最终 Reader 才使用完整问题和公开选项，保留原有 chain reader；residual 不增加 Reader。两臂仍有方法和最终生成路径差异，这是一组系统级对照。

对 residual 等终端直出方法，适配层还将 `reader_question` 设为纯 `user_question`，通过独立 `output_options` 把选项仅传给真正终端。这里的参数名不表示调用了 Reader。这样避免把完整 MCQ 末尾的 `Return exactly one option label ... no explanation` 指令带进终端的结构化 JSON 请求；终端同次返回 `semantic_answer` 和 `final_prediction` 的既有协议保持不变，没有新增事后的标签匹配或重答调用。其 `diagnostics.memory_scope.question_protocol` 为 `user_request_reasoning_structured_mcq_terminal_v3`。历史 fusion 仍在最终 Reader 接收完整 MCQ，不能把它与终端方法的修复混为一条路径。

冻结的 `package/`、`dagv2/` 和 BT vendor 不在本次修改范围。题目、选项、正确答案、记忆切分、题目截止时间和作用域不因问题适配改变。只有新的代码和运行记录才能体现本次修复，不能把旧结果复制过来继续计数。

## 1. 解压到独立目录

文件上传到 `~/bt/` 后，在终端执行：

```bash
mkdir -p "$HOME/dagbt_runs/20261011_personamem_a48"
tar -xzf "$HOME/bt/dagv2_bt_deploy_20261011_personamem_a48.tar.gz" \
  -C "$HOME/dagbt_runs/20261011_personamem_a48"
cd "$HOME/dagbt_runs/20261011_personamem_a48/dagv2"
```

如果旧实验仍占用服务，先在旧代码目录用原命令停止对应输出目录，等待 `status` 显示 `live_writer: false`。保留旧结果用于核对；新包不要覆盖解压到旧代码目录，也不要复制旧 `rows`、`attempts`、请求缓存或运行 manifest 到新输出目录。

## 2. 选择 Python 环境

如果此前的环境位于 `~/dagbt_runs/20260930_v3/dagv2/.venv`，可以复用解释器和依赖。保持当前工作目录为新包根目录：

```bash
export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
"$DAGBT_PYTHON" -c 'import dagbt.runner; print(dagbt.runner.__file__)'
```

输出应位于 `20261011_personamem_a48/dagv2/dagbt/runner.py`，表明加载的是新代码。若旧环境在别处，只修改解释器路径；新终端需要重新 `cd` 到新目录并设置变量。

没有可复用环境时，在新包根目录创建一个：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-fusion.txt
export DAGBT_PYTHON="$PWD/.venv/bin/python"
```

## 3. 可选复用派生索引，并离线检查

旧运行的 `data/derived` 可以复制到新目录，避免重新编码相同 PersonaMem 语料。仅在新实验尚未启动时执行一次：

```bash
if [ -d "$HOME/dagbt_runs/20260930_v3/dagv2/data/derived" ]; then
  mkdir -p data/derived
  cp -a "$HOME/dagbt_runs/20260930_v3/dagv2/data/derived/." data/derived/
fi

"$DAGBT_PYTHON" -m dagbt.runner preflight \
  --config configs/residual-memory.example.json \
  --datasets personamem --offline-preflight
```

预检按模型、embedding 端点、语料、文档顺序和序列化身份定位索引，核验完成 manifest、向量 SHA256、矩阵形状及归一化状态。`ready` 才表示可复用；`needs_build` 表示启动后还需准备索引，复制目录本身不保证复用成功。已宣告完成但内容损坏的索引会报错，不会默默退回旧 NV 向量。

如服务端点不同，在新目录准备自己的 residual 本地配置，并对后续所有命令一致替换 `--config`；保持 `fusion.algorithm_version=dagbt_residual_memory_v1`，不要用 `paired.example.json` 代替。embedding 身份变化时旧向量可能无法复用。离线预检不访问真实服务。

## 4. 两题烟测

```bash
"$DAGBT_PYTHON" -m dagbt.runner launch \
  --config configs/residual-memory.example.json --datasets personamem \
  --arms original dagbt_residual_memory_v1 \
  --limit 2 --output outputs/personamem_a48_smoke

"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_a48_smoke
tail -n 80 outputs/personamem_a48_smoke/launcher.log
```

这是 2 道题、两个方法，共 4 个任务。`launch` 执行线上预检后启动后台进程，不需要 `nohup`；返回 PID 不等于实验通过。`preparing_index` 表示仍在准备语料向量，尚未进入答题。没有可用索引时，即使两题烟测也会为 PersonaMem 语料建索引。

烟测结束后，读取轻量摘要：

```bash
"$DAGBT_PYTHON" - <<'PY'
import json
from pathlib import Path
path = Path('outputs/personamem_a48_smoke/personamem/summary.json')
if not path.exists():
    print('尚无最终摘要，请先查看 status 和 launcher.log。')
else:
    summary = json.loads(path.read_text())
    for arm in ('original', 'dagbt_residual_memory_v1'):
        result = summary['arms'][arm]
        print(arm, '状态=', result['answer_status_counts'],
              'accuracy(%)=', result['all_task_metrics_percent']['accuracy'])
PY
```

应核对两臂实际名称和失败数，不能把 `complete_with_failures` 当成全部成功。residual 的产物在 `personamem/dagbt_residual_memory_v1/`；如果输出目录只有旧 `fusion/`，应核查启动参数。两题结果只能检查接口和流程，不足以判断方法准确率。

## 5. 全量 589 题

烟测核对后，使用新输出目录，去掉 `--limit`：

```bash
"$DAGBT_PYTHON" -m dagbt.runner launch \
  --config configs/residual-memory.example.json --datasets personamem \
  --arms original dagbt_residual_memory_v1 \
  --output outputs/personamem_a48_full

"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_a48_full
tail -n 80 outputs/personamem_a48_full/launcher.log
```

两臂各执行固定官方 32k 文件的全部 589 题，共 1,178 个方法任务。包中保留其他数据集时，上面的 `--datasets personamem` 仍确保只跑 PersonaMem。

停止与继续：

```bash
"$DAGBT_PYTHON" -m dagbt.runner stop --output outputs/personamem_a48_full
"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_a48_full
```

等待 `live_writer: false` 后，再执行完全相同的 full `launch` 命令，继续尚未完成的运行。已有终态默认跳过。**本配置 `max_question_attempts=1`**，已开始后被中断的 attempt 也占额度，不能承诺再次启动或加 `--retry-failed` 就会重跑该题；达到上限会保留或记录失败。需要允许额外尝试时，应在首次启动前准备好相应本地配置。启动后改变代码、配置、方法或题目范围，都必须使用新的输出目录。

烟测、全量、旧 v3/旧 fusion 和之前的 v4 结果各自独立，不能混合追加。这里的 runner 恢复是任务运行恢复；不等同于方法内部观察边界快照的恢复接口。

## 结果与验证边界

全量结束后读取 `outputs/personamem_a48_full/personamem/summary.json`，两臂主指标为 `all_task_metrics_percent.accuracy`。合法单选与正确选项一致才计 1；失败、非法标签、多选或附带解释的无效输出计 0。`persona_macro_accuracy_percent` 先在 persona 内平均，和全题 accuracy 分母不同；共同成功子集也应与失败率一起报告。

residual 没有旧 Reader/FinalSelector 的选集，相关指标为不适用；不要用空 `budgets` 判定数据丢失，也不要把来源证书理解为模型语义解释必然正确。本次问题适配不改变 residual 的算法含义或停止规则，不构成准确率提升结论。

本说明的命令供用户在服务器执行，没有宣称线上烟测或全量实验已经通过。本次最终修复提交、离线测试和打包校验结果由交付记录统一给出，不沿用先前 v4 包的验证数字。
