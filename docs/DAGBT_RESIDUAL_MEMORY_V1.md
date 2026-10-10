# 剩余求解驱动的逐节点记忆推理

方法为 `dagbt_residual_memory_v1`，固定代码基线为 `f4e38df23b5b621a4d0e759c018c60e1fe25dd9b`。它复用 Engine 的一次规划、来源注册、transport 和全题 Ledger；旧 `dagbt_local_terminal_v1` 的局部评分路径保留为独立对照。

第一次检索前，规划明确初始任务 DAG、每个任务的输入依赖和 `final_node_id`。未知实体、规则和条件保留为待绑定输入，不省略可预先规划的下游任务。F 是这张 DAG 的类型化求解关系，F_t 是当前证据下的剩余求解状态；二者不替代 DAG。BT 的 dense 和条件桥接始终归属于具体任务或它的局部缺口。

例如岚桥案例先规划 rules → D/A/O → decision。规划不猜规则门槛；取得当前规则的原文后，才在预先声明的 D/A/O 任务内实例化条件变量和比较表达式。原始问题、输出契约、初始 DAG 和 `requirements_hash` 保持固定，实例化另外增加 `program_revision`、规则引用和记录。

## 一次证据更新如何改变下一步

每轮先免费计算剩余式和活跃需求，再选择依赖已绑定的 DAG 节点。当前需求第一次执行一批 dense，立即返回阅读；存在缺口时再推进一个原文条件桥接批次。适配层显式设置未扩展初始池，避免 vendor `propose()` 隐式扩满初始池。共享语料按文档身份去重，发现路线与证明路线分开。

一次空探针不会关闭仍有根或延续分支的查询。BT 返回 `empty_probe`、`frontier_exhausted`、`budget_exhausted` 或 `service_failed` 等状态；控制器仅在前沿真正耗尽、预算耗尽或动作失败时关闭当前查询身份。前沿检查不消费状态，空探针仍按实际请求计费。

联合阅读一次返回类型化事实、来源、实际父版本、未解字段和反证。它复用原有 source alias 注册与准确坐标，不调用全量 mapping。输入包括当前节点、必要父来源、当前原文窗口与相关反证；实际 wire 模板、schema、输出预留、输入余量和局部修复都计入容量。长文显式分窗，失败槽位与未读范围留存。

同批合法事实和反证先整合，再统一求值。未知、false、0、空集合分别保存。独立来源是 OR 替代路线；证书采用实际充分路线，来源撤销后可换用仍有效的路线。冲突值暂不绑定；实际依赖旧版本的事实失效。失败协议槽位不会因后续另一次成功响应而被抹去。

阅读义务按查询身份分别记录未读原文窗口与校验失败批次。窗口成功读取后只关闭对应的未读义务；失败字段、反证和其他查询的未读义务继续阻塞相关证书。快照保存两类义务，恢复后能够关闭后续已读窗口，不清空真实失败。旧快照缺少分类信息时保守保留其 pending。

确定性 AND/OR/IF 先部分求值，再判断仍需哪些父输入。中间 compose 可用 `task_expressions` 指定白名单运算，免费形成有推导记录的绑定，然后接地下游取证；语义 compose 和实体参数仍需必要父输入。一个否决见证可以消去其他条件取证；全部失败项契约继续要求完整列表。暂停前沿仍可恢复，事实撤销后重新激活需求。

规则中的 D 使用 `IF(违规见证, false, delete_compliant)`：在删除后180秒仍返回足以否定“60秒起不得返回”，单次未返回则不能证明所有时点合规。`delete_compliant` 是预规划 D 内的独立待绑定输入，联合约束排除完整合规与违规见证同时成立。字段只引用本节点及其祖先输入时，可用带来源的局部投影证书完成该节点；未使用输入保持未知。joint 与 residual 都允许这种局部求解，joint 仍继续求解 A/O。

## 契约、表达式与证书

规划在已有 `steps` / `final_node_id` 上增加 `output_contract` 与 `program`。

| 对象 | 内容 |
| --- | --- |
| output_contract | mode、结果类型、对象/时间范围、解释要求、可选字段名 |
| program.variables | 稳定变量 ID、类型、所属预规划 demand、可选有限域/开放域/单位 |
| program.expression | 终端求解关系 |
| program.task_expressions | 可选中间 compose 变量的确定性运算，引用已声明父任务 |
| program.rule_binding | 待绑定规则变量，以及预规划的条件任务 ID |
| certificate | 契约/程序身份、版本、实际事实与规则路线、来源、等价规则、世界外包络及消去变量 |

白名单只有常量、变量、AND/OR/NOT、相等和数值比较、IF、记录/元组、失败项集合与一个原因。没有模型生成的可执行代码。时间与长度单位、区间开闭端点受到检查。有限枚举尊重重复变量和联合约束；空世界不认证，超限不使用部分枚举。开放实体保留其他可能值类。

数值合法域定义可能世界，验收门槛放在比较表达式中。观测与合法域统一单位后求交，再参与绑定、部分求值和枚举；空交集在读取层拒绝绑定，直接求值则返回 inconsistent，不能签发证书。非空交集保留正确的开闭端点。不存在静默修改声明域的路径。

约束未知的抽象世界仍纳入安全输出外包络；至少一个约束全部为真的抽象世界另外证明可行世界非空。只有整个外包络的输出都精确且一致时才认证；没有已证明可行世界时，常量输出也不认证。

`certified_symbolic` 条件于当前合法语义解释与域声明，不证明模型没有误读，也不证明已检索全部反证。规则完整性未证实时不能认证肯定结果或完整失败集合；明确必要条件的否定见证可成立。开放建议使用 `semantic_assessment`，保留实际终端响应，不伪造有限世界证书。

已有不完整规则路线时，控制器仍将输出所需的规则完整性作为局部缺口激活 rules 节点，加入其查询与联合阅读上下文。原文确认完整规则后，通过带来源的撤销和新支持更新规则版本，失效的条件绑定重新求解。已有足够的必要条件否定见证且契约仅需一个理由时，不强制补齐规则。

候选证书先检查来源、范围、版本和已知相关反证，再计费执行必要语义审计。审计改变事实后重新求值；不能发布旧证书。最后一次付费响应后即使额度用尽，免费求值和发布检查仍执行。符号终端有真实推导路径，未构造假的终端模型输入。

公开选项仅在真正终端输出阶段出现。取证节点本身就是终端时，同次联合阅读返回类型化语义答案与标签；已经生成的终端不另发选项匹配调用。精确可确定的对应关系直接转标签；纯确定性 compose 若需要语义输出，只在该节点第一次且唯一的终端生成中同时返回 `semantic_answer` 与 `final_prediction`，并核对前者等于程序结果。证书只覆盖语义结果。无独立 Reader、FinalSelector 或之后的重答。结果保留 `budgets={}`、实际来源闭包、输出身份、完成类别和程序版本。没有模型终端输入的纯符号输出保持 `terminal_input_doc_ids=[]`；发生真实终端生成时另记其输入和响应。

## 执行与恢复

```bash
python scripts/demo_residual_memory.py --output outputs/residual_memory_v1/demo
python scripts/ablate_residual_memory.py --output outputs/residual_memory_v1/ablation.json
python scripts/smoke_residual_memory.py \
  --config configs/residual-memory.example.json \
  --output outputs/residual_memory_v1/smoke_new
python -m pytest tests -q
```

使用安装了仓库依赖的 Python。本地验证使用原 worktree 的 `.venv/bin/python`。新配置不含 reranker。全题上限 ANN36 / LLM24 / Reader0 / set_score0；transport 拒绝新零评分方法的 reranker 请求。实际 HTTP 重试仍由既有 transport 计费，未把删除评分的额度转给生成。

观察边界可调用 `engine.residual_control.snapshot()`，将返回的 JSON 写入私有目录；相同资源和配置的全新 Engine 使用 `engine.resume(snapshot)`。恢复校验问题、契约、初始 DAG、程序、语料与预算身份，恢复事实版本、读取状态、查询身份、frontier、seen、消费位置与已完成请求。不会重跑规划或 dense；不支持恢复尚在进行的 HTTP 请求。快照含原文，不应提交 Git。

四种独立方法为 scored local-terminal、`dagbt_local_terminal_proxy_free_v1`、`dagbt_joint_memory_v1` 和完整 residual。前两种保留旧 mapping/resolve，后两种使用联合阅读；joint 对照继续执行预规划取证任务。旧 eager bridge 与新 pausable bridge 的在线比较同时改变检索控制，不能把全部差异归因于剩余求值；脚本另提供固定完整候选池的 joint/residual 对照。

## 边界

调度采用稳定的可执行任务顺序和根/延续轮换，不是期望成本最优策略。来源蕴涵、适用规则和域声明仍依赖模型解释。重叠但不同的数值区间目前保守地视为冲突，不自动取交集。局部协议失败可能保守地阻塞相关输出；无法容纳必要来源时明确未完成。新规则只能在预规划 condition demands 内展开；超出初始 DAG 的付费需求不隐式创建。真实模型效果和 APC 硬件性能没有由离线替身证明。
