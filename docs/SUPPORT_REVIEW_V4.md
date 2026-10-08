# 融合 v4：复核支持证明，再由程序选择来源闭包

这是一项从冻结 v3 提交 `774d285ffe91a23a99ee9afb006e85103be335b5` 分出的实验修改，工作分支为 `feat/support-proof-review-v4`。默认 dependency 方法的融合可靠性版本为 `dagbt_fusion_support_review_v4`，语义诊断版本为 `dagbt_semantic_evidence_v4`。原 DAG v2、模型配置和数据协议不因这次支持复核变更而改变。

**新版本必须使用新输出目录。** 代码身份和方法协议已改变，不能恢复或混写 v3 的结果。`scripts/run_v3.sh` 是历史包装脚本名，名称不决定当前加载的选择协议，其默认输出目录仍需特别留意。建议直接指定 runner 和新目录：

```bash
# 只检查依赖、配置、源码和数据；不访问服务。
.venv/bin/python -m dagbt.runner preflight \
  --config configs/paired.example.json --offline-preflight

# 后台启动默认 original/fusion 两臂、四套数据的完整实验。
.venv/bin/python -m dagbt.runner launch \
  --config configs/paired.example.json --output outputs/paired_support_review_v4

.venv/bin/python -m dagbt.runner status --output outputs/paired_support_review_v4
.venv/bin/python -m dagbt.runner diagnostics --output outputs/paired_support_review_v4
.venv/bin/python -m dagbt.runner stop --output outputs/paired_support_review_v4
```

如有端点覆盖，用自己的 `configs/paired.local.json` 替换示例配置。烟测另用 `--limit 2 --output outputs/paired_support_review_v4_smoke`，不能再把同一目录扩成全量实验。恢复仍使用同一 `launch` 命令和完全相同的范围参数；重跑终态失败需显式加 `--retry-failed`。离线预检通过不代表服务或完整实验通过，`launch` 仍会进行线上预检和实际推理。

## 与 v3 的区别

v3 的最终模型复核输出文档 ID 和需求覆盖标注。程序约束选集并在已有支持图上计算结构证书；模型读到但尚未映射的原文可以进入 reader，却不能仅靠选中原文补出支持路线。固定选集后的覆盖修复还要保持已接受的 header。已有失败分析暴露了这种协议的多种限制，但不能把全部失败都归因于同一个原因，也不能据此推断 v4 的准确率会提高。

v4 的默认 `FinalSelector` 把模型职责改为**提出支持图补丁**。模型可从本次完整可见原文补充精确证据片段、修改节点的支持方案、提出失效和冲突解除；程序验证补丁、更新图，再在 reader 预算内选择完整来源闭包。模型不再直接写最终选集及 `covered` 标注。

```text
当前问题 → DAG 规划 → 按节点发现候选与建立证据/支持路线
         → 支持复核：已有图 + 完整可见原文 → 模型提出补丁
         → 来源、依赖、版本、冲突验证 → 更新支持图
         → 分别为 k=5/10/20 选择完整可行闭包
         → reader 读取所选原文 → 答案；评分阶段另读标签
```

原问题 dense baseline、BT 发现方式和完整原文渠道沿用现有设计。raw 渠道可挽回映射阶段漏掉的历史，但只有建立并通过验证的支持路线才能改变结构覆盖。

## 补丁协议与来源约束

`select` 使用 `prompts.SUPPORT_REVIEW`；无修改的合法响应如下：

```json
{
  "new_spans": [],
  "invalidations": [],
  "resolutions": [],
  "node_updates": [],
  "supplemental_doc_ids": [],
  "reason": "复核后保留当前支持图"
}
```

字段的严格 schema 和验证以 `dagbt/proof_review.py` 为准。每层对象显式列出全部必填字段并拒绝额外字段；补丁不接受旧协议的 `selected_doc_ids` 或模型自行填写的覆盖结果。`new_spans` 引用当前复核输入中**完整可见**的原始记录：`doc_id`、Python 字符坐标 `start` 和 `quote` 必须精确对应原文，单片段最多 `min(400, max_quote_chars)` 字符；不能引用本轮没有看到的文档，也不能凭摘要制造来源。既有片段仍需通过实际可见 ID 引用。新片段声明节点、支持/反对立场、显式/推断种类、主张和实体范围，这些自然语言关系仍是模型判断。

来源角色从原记录的 authority/source segments 推导。跨多种角色的片段保留分段、标记歧义，不能把 assistant 建议改写成 user 偏好。观测顺序不是事件时间；新片段的 `event_time` 固定为空，不凭排序伪造时间。原始角色、作用域以及 PersonaMem 的题目截止时间约束继续生效，未暴露的未来消息或其他 persona 文档不能借复核进入证据。

`supplemental_doc_ids` 只保留模型认为仍有帮助的可见原文，服务于没有完整证明时的部分上下文。该字段不生成片段、支持边或完整覆盖声明。空数组允许保持空上下文；程序不会为了凑满预算自动宣称这些原文支持了答案。

## 节点版本、替代路线与冲突

节点更新显式列出 `retained_alternative_ids`，而不是默认用新路线替换全部旧路线。复核看不到的旧路线必须保留；看到的旧路线可以明确删除。保留路线保有原 ID、失效和争议状态，不能靠换 ID 将旧的无效路线重新标为有效。

对同一节点、答案、适用范围、父版本和原文来源区间，硬失效证明的身份记录独立于模型局部 ID。重命名来源、重新引用相同 quote 或把同一区间拆成多个 quote 不能复活该证明；即使一轮已删去旧路线，后续补丁仍检查已保存的失效身份。答案或适用范围真正改变时可以重新评估来源，但必须走新版本，不能把不同结论塞进同一组 OR 替代方案。这是来源结构约束，不是对模型换一种语义说法的完整等价性判定。

Engine 后续重编译保留 `proof_review`、补充原文 ID、导航闭包及单调 revision；最终 `diagnostics.support_graph` 带有该审计，不只在语义事件里留副本。对重编译后的图再次复核时，已经删除的失效路线身份仍可阻止同源复活。

新路线仅能绑定该节点的拓扑前驱，而且父节点必须有当前可见的完整有效证明；程序绑定其当前版本。修改节点答案或适用范围时，必须显式放弃旧路线，且不得藏起未显示的旧路线再更新。版本更新使后继对旧父版本的绑定失效，随后只有使用新版本的路线才可能构成有效闭包。

复核展示按来源完整性判断：展示子路线时，每个实际父节点只需有一种来源完整可见的路线，不要求把父节点全部 OR 方案的来源并入输入。分支汇合需要每个实际父节点的来源。可疑、失效或父版本过期的旧路线仍可展示，供模型撤销和修订；展示不会把它改成 eligible。只有给**新路线**绑定父节点时，才要求完整可见、有效的父证明并使用当前版本。看不到某条路线的完整前提时，该路线不可被本轮补丁修改，但仍留在真实图中供最终选择。

对于已有答案和范围不变的节点，新原文建立的完整独立路线可以把节点从不完整升级为受支持，而不删除本轮不可见的旧路线。旧路线若原本因节点缺项而不可用，不得随节点升级获得有效资格；程序会在必要时把这份已知的不完整性保守记到旧路线的 `semantic_status=partial`，保留其 ID、来源、父版本和冲突记录，并审计原节点缺项。真正修改不可见路线的答案、范围或来源仍被拒绝。

如果实际父节点已经没有任何支持路线，复核视图明确标记 `missing_parent_proof_ids`，允许撤销或替换依赖它的失效旧子路线；这一缺失状态不是证明，不能用于新的父绑定。父路线存在但被输入预算省略时，则仍按不可见来源保护，不能冒充“没有路线”来删除。

失效处理传播到依赖后继；冲突不能通过删旧路线、新建同义路线绕过。存在未解除节点冲突时，新路线继承相应限制。解除冲突必须引用当前可见证明，并覆盖原冲突的全部反证片段；解除所依据的证明进入新路线的 guards，防止下一次选择遗忘解除条件。

补丁在图副本上完整验证后才应用。任一字段、来源、版本或冲突操作非法，整个补丁被拒绝，原图不受半成品更新污染。局部修复重提补丁，不能把一个失败补丁的前半部分当作已经生效。

## 完整闭包与不完整回退

最终选择由程序从验证后的图中计算。一个证明使用的节点来源、父支持、guards 和适用的导航闭包必须整体进入上下文；同时满足文档数与实际 reader 上下文可行性限制。若一条完整路线超预算，尝试其他完整路线；不能先选证明，再截断来源列表来满足 k。

k=5、10、20 分别求解并认证，不从 k=20 的列表截取前 5 或前 10 个文档。默认要求仍是一个必要的 `answer` 需求，全部 DAG 终端节点同时满足；这是结构目标，不是若干中间节点各拿分。完整路线枚举使用有限图的实际状态规模作为足够的最终搜索界限，避免过小配置上限漏掉本来可行的证明；有效搜索上限进入诊断。

没有完整可行证明时，允许保留部分闭包、争议来源组和被复核认可的补充原文，reader 仍可利用这些内容作答。该结果明确保留结构不完整状态，不因 reader 答对、补充文档存在或 k 用满而自动变为 `complete_required`。

## 失败、预算和诊断

支持复核继续使用共享的每题、每方法、每次 attempt 预算，修复有次数上限；不会为每个节点或每次修复重置额度。非法补丁、输出截断、输入或调用预算不足在有界修复后可退回已有合法图，明确记录 `review_complete=false` 和失败原因。不能把降级标成完整复核。

`ServiceError` 和 `RefusalError` 仍是任务失败并向外传播。失败路径保存当时图、语义阶段、请求和实际尝试计数；不会把服务故障或模型拒绝伪装成普通“不完整支持”。

保留 `selection_pending`、`selection_progress`、`semantic_selection_complete` 事件，以及 `evidence_state`、所选 ID、逻辑调用数和预算尝试数等诊断。结构覆盖由程序从图和最终选集推导，不再来自模型的覆盖标注。应分别读取：

- `review_complete`：支持补丁复核协议是否完成。
- 图覆盖与 `complete_required`：选集是否包含当前图所要求的完整有效闭包。
- 答案正确性：评分阶段对照独立标签得到的结果。

这三者不能相互替代。精确引用、合法图结构和完整闭包都不保证自然语言推断为真，`review_complete=true` 也不表示答案正确。

runner 保留历史 `coverage_state`，其 `complete/unassessed/unknown` 描述覆盖标注或兼容验证字段，**不能解释为存在完整证明**。v4 新增的 `support_state` 才描述结构支持，取值为 `complete/incomplete/unknown`；只有选择已结束、`structural_validation_complete=true` 且 `complete_required` 是明确布尔值时，runner 才确认 complete 或 incomplete。旧记录缺这些字段、选择尚未结束，或结构验证标志与覆盖布尔值不明确时保持 unknown。

例如无证明但 raw reader 成功的题，可能同时是 `coverage_state=complete`、`review_complete=true`、`support_state=incomplete`；模型复核失败后保留已有完整路线，则可能是 `review_complete=false`、`support_state=complete`。读取准确率时必须保留这种区分。

方法摘要的 `support_completion_cohorts` 对成功产出合法答案的题按结构状态分组，并报告题数、评分阶段的答案指标和最近尝试成本；不等于只统计答对的题。`all_task_support_state_counts` 包含全部当前权威任务，`failed_support_state_counts` 单列失败任务。历史的 `coverage_completion_cohorts` 和 coverage 计数继续保留。失败重试历史不会作为额外题目重复计数；所有重试费用仍在 `cost_all_attempts`。这些分组用于描述结果，不能单凭组间准确率差推断完整证明带来的因果收益。

## 需求与回归证据

四项要求沿用已有 [支持图实现](../dagbt/support.py)，由 [补丁验证](../dagbt/proof_review.py)、[最终选择](../dagbt/final_selection.py) 和 [Engine](../dagbt/engine.py) 接线；没有新增与原图平行的证明系统。

| 要求 | 实现位置与约束 | 对应验证 |
|---|---|---|
| 1. 选择完整支持路线，保留直接来源、guards 和实际父来源，支持替换、撤销和汇合 | `select_support` 枚举节点方案并递归合并实际父闭包；`validate_selection` 在 reader 前重查选中路线；`apply_review` 原子修改方案 | `test_supplement_b_cannot_drop_its_required_parent_a`、`test_converging_branches_preserve_both_parents_and_exact_guard`、最终 parent/guard 删除拒绝测试 |
| 2. 保留 OR 替代与 raw 补救，不锁初选，也不永久保留历史来源 | `apply_review` 校验精确 quote、节点归属、范围、可见父证明和父版本，调用原 `invalidate_support`、`resolve_conflict`、`update_node_version`；raw 新证明可省略规划父依赖 | 独立 raw 路线、父证明替换、答案/scope 变更传播测试，以及 `test_proof_review.py` 中可见性、冲突、同源失效和原子性测试 |
| 3. 按完整原文 union 计费，共享来源只计一次；超预算重选而非裁断 | `FinalSelector._select` 复用 `select_support`，`Engine.feasible` 计算真实 reader 消息成本；k=5/10/20 独立求解；reader 前再次验证 | `test_shared_source_charged_once_and_callback_includes_template_overhead`、token 超预算换路线、六前提不能截成五文档、最终 reader 闸门测试 |
| 4. 有可行完整证明时保留选中证明；否则保留 raw 回答通路并标 incomplete | `_select` 先搜完整证明，仅在不完整时填充部分来源；`run` 的有界协议失败回退原图；raw 补充不创建 span/边 | 无图时的真实 raw reader 测试、预算不够时 raw 回退、极小探索枚举上限测试、runner 的 support_state 测试 |

六项核心回归以实际 Engine 路径为主，对应 [tests/test_support_review_v4.py](../tests/test_support_review_v4.py) 和 [tests/test_final_selection_v4.py](../tests/test_final_selection_v4.py)：

| 回归 | 具体测试与断言 |
|---|---|
| 必要前提不拆 | `test_supplement_b_cannot_drop_its_required_parent_a`：模型补充 B 时，采用 B 的证明仍把 A 一起送入 reader；`test_reader_gate_rejects_parent_or_guard_removed_after_real_selection`：最后删除 parent/guard 时直接拒绝 reader 请求 |
| 独立 raw 替代 | `test_empty_mapper_review_adds_exact_c_as_direct_route_without_parent`：空 mapper 后从 C 精确建证据，最终只选 C，保持 `used_parent_ids=[]`，不伪造原规划父节点支持 |
| 共享文档不强制全部 OR | `test_document_shared_by_or_routes_does_not_force_all_other_premises`：A+B 与 A+C 只取一条完整路线，未选路线的独有来源不进入 reader |
| 失效证明不永久保留 | `test_review_revokes_one_support_and_removes_its_old_document`：失效路线保留审计状态，但被有效替代后其原文退出最终上下文 |
| 预算不足不裁断 | `test_reader_budget_reselects_whole_proof_or_marks_raw_supplement_incomplete` 与 `test_five_document_budget_reselects_complete_or_route_instead_of_slicing_six_premises`：超 token/文档预算时选另一条完整路线，无替代时不伪报完整 |
| 无完整证明仍保留 raw 通路 | `test_failed_review_with_no_proof_preserves_actual_raw_reader_context`：修复耗尽且没有 span 时，实际 reader 的 Context passages 仍包含完整原文，`complete_required=false`、`review_complete=false` |

补充边界分别由 [tests/test_proof_review.py](../tests/test_proof_review.py) 验证源码引用、版本、冲突与失效身份，由 [tests/test_support.py](../tests/test_support.py) 验证 union 计费和既有求解器，由 [tests/test_runner_support_v4.py](../tests/test_runner_support_v4.py) 验证历史 coverage 与 v4 support 统计不混淆。[tests/test_review_visibility_v4.py](../tests/test_review_visibility_v4.py) 进一步验证长父 OR 放不下时短证明仍让子路线可见、来源完整的有效或失效子路线均可撤销、隐藏父路线不可删除，以及分支汇合无需把全部父 OR 强制取并集。上述是需求到测试的定位；最终通过数量以本次完整验证记录为准。

`test_engine_recompile_preserves_invalidated_proof_history_and_blocks_later_revival` 还执行真实 Engine 失效、替换、reader 和最终结果保存路径，随后再次编译并提交旧来源补丁，检查失效身份不会在 Engine 生命周期中丢失。`test_schema_is_compatible_with_strict_structured_output` 递归检查 schema 的全字段必填和额外字段拒绝；它不等同于对某个线上模型的结构输出能力认证。

## 对照方法与验证范围

`selection="dependency"` 的默认 `FinalSelector` 使用上述支持复核。`selection="flat"` 保留 `DocumentSelector` 和 v3 文档选择/覆盖标注协议；因此 `bt_flat`、`dense_flat` 是既有文档复核对照，不应解释成仅从同一 v4 协议移除一条闭包约束。旧文档中的 v3 header 固定、coverage repair 等细节描述历史协议。

`fusion_no_raw_review` 关闭独立原文渠道，仍可能复核已有可见证据；`selection_review=false` 关闭最终模型复核，并改变前期预算预留等流程，不是只关 raw 输入的单因素消融。历史命名 `fusion_strict_coverage` 在 v4 dependency 下将 `allow_unassessed_coverage` 设为 false，作用是禁止复核失败后退回原图，不再表示模型覆盖行的修复策略；复核非法或预算失败会直接失败。其他方法开关仍以 `dagbt/config.py` 为准；解释结果时应报告实际配置和输出中的版本身份。

离线协议测试使用明确脚本化的模型响应，执行真实来源验证、图更新、闭包选择和 Engine/reader 接线。它们验证程序约束与失败记账，不证明线上模型会稳定地产出这些补丁，也不测量数据集准确率。本次修改不包含真实服务实验，不继承 v3 文档中的线上探测结论。

选择器回归入口为 `python -m pytest -q tests/test_final_selection_v3.py tests/test_final_selection_v4.py`。范围包括保留的文档复核回归、完整原文与角色 metadata、不可见路线拒绝修改、协议/截断回退、坏 quote 原子拒绝、无证明时真实 reader 原文回退、严格模式、服务/拒绝失败、低枚举上限，以及最终 reader 前删除 parent 或 guard 来源时的拒绝。真实子进程加本地 HTTP 服务的接线验证另见 `tests/test_system_http.py`、`tests/test_bt_profile_system.py` 和 `tests/test_personamem_runner.py`；这些服务仅返回显式脚本响应，仍不是线上模型实验。最终通过数量和完整运行记录由本次交付统一报告。

## 本次交付验证记录（2026-10-08）

- 完整测试：`python -m pytest -q`，**620 passed, 6 warnings in 44.34s**。警告来自既有 original/resource parity 测试中的 NumPy 数值运算。
- 冻结内容完整性：`verify_originals(include_labels=True)` 检查 **57** 个文件，全部通过。相对基准 `774d285ffe91a23a99ee9afb006e85103be335b5`，`package/`、`dagv2/`、`vendor/`、`original_manifest.json` 和 `README.md` 均无修改；`git diff --check` 通过。
- 上一轮完整测试曾触发一次既有的启动/停止竞态：`test_detached_cli_stop_reaps_active_workers` 在 worker 启动期间收到停止信号，清理时报 `can only join a started process`。相关启动、运行、停止和清理函数与基准一致，本次未修改进程管理逻辑；该项单独重跑通过，随后完整重跑得到上述 620 项全部通过的结果。
- 本记录覆盖离线测试及脚本响应的本地 HTTP/子进程接线，不包含真实模型服务实验或新准确率结果。
