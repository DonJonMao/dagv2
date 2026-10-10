# residual_memory_v1 验证记录（2026-10-10）

状态：`IMPLEMENTED_AND_TESTED_OFFLINE`。固定代码基线 `f4e38df23b5b621a4d0e759c018c60e1fe25dd9b`，目标分支 `feat/residual-memory-f4e38df`，从已有远端 `eaeb5b2aadc2c881a32b11b245412340bbe5f7dd` 的两份规格文档继续实现。独立 worktree 为 `/Users/mao/projects/dagv2-residual-f4`。原 `feat/local-scoring-terminal-answer` 分支及其未跟踪研究资料、用户 PDF、旧报告和 trace 均保留。

直接继承 f4 的无额外 Q dense、当前子问题局部评分基线、指定终端直接输出和无独立 Reader / FinalSelector。新增类型化剩余求值、可暂停零评分桥接、联合阅读、规则实例化、事实与实际依赖版本、审计后重新求值、证书发布及观察边界恢复。用户补充的完整初始 DAG 要求已经进入 planner 提示、验证器和两份原规格文档，撤销了案例中仅预规划规则任务的旧表述。

## 离线检查

正式范围为 `python -m pytest tests -q`；不递归收集 outputs/tmp 中的历史副本。最终测试数字见本文件末尾的交付核验。新增语义测试使用固定 seed、300 个随机小布尔表达式与独立有限世界 oracle；不是只测一个 AND。运行测试包括实际 Engine、冻结 BT、来源注册/校验、断点恢复和 loopback 真实 HTTP。

首轮为 748 passed / 7 failed：七项失败都是新 worktree 缺失原有未跟踪向量文件。复用原 worktree 的既有索引并核验身份/哈希后解决；未重建或修改原始索引。中途新增测试曾发现 fixture 的 vendor stage 名、joint 对照过早标记任务已访问、选项应使用既有 list 协议、恢复契约身份缺失，以及“相关失败槽位应保留、独立否决见证仍可完成”这几处问题；均保留测试并修正实现或错误测试预期。

| 要求 | 实际覆盖 |
| --- | --- |
| A1–A3 | 未知/false/0/空集合，AND/OR/IF，一个原因与全部失败项的不同完成条件 |
| A4–A6 | 空世界、枚举上限、共享/重复变量、x=y 关联约束、固定 seed oracle |
| A7–A8 | 开放实体其他类、单位转换、严格/非严格边界与区间端点 |
| A9–A10 | 同批冲突先整合，独立 OR 来源，父/规则/来源版本失效与恢复 |
| B11–B12 | 集中方法分流，零评分配置/transport，单节点 q=Q，真实 HTTP spy，无 Reader/selector |
| B13–B15 | dense 单步返回，vendor 未扩展池初始化，JSON frontier 恢复，撤销后重新激活 |
| B16–B18 | 跨池共享文档保留发现边，联合事实/来源，无 map 调用，read/invalid/unread/gap 独立记录 |
| B19–B20 | 中间确定性 compose 短路接地下游，无猜测推进；实际 wire 容量、窗口与 scoped repair |
| C21–C23 | 符号推导支持路径、实际闭包、真实语义响应、来源角色与单选隔离；原 local-terminal 回归 |
| C24–C26 | 最后阅读加审计恰好用尽 LLM 后免费发布；审计撤销后重算；无关失败槽位不否定独立充分见证 |
| C27–C28 | 空 budgets、新旧 runner/export，逻辑/物理计费、服务与 schema 失败保留，未完成无答案替身 |

检查 `git diff --check`；相对固定基线的 `package/`、`dagv2/`、`vendor/bridgetree/`、`dagresolve/`、data、根 README 和两份 manifest 共 **108 个 tracked 文件，变更 0**。原始 manifest **57** 项、vendor manifest **57** 项哈希均通过；标签文件仅读字节做保护哈希，没有解析或向生成传标签。未改冻结源码以消除已有矩阵运算警告。

## 完整合成轨迹

原文、模型和检索服务均为明确离线替身，使用真实 Engine 与 BT。全部资料虚构，最终运行产物位于私有 `outputs/residual_memory_v1/final_delivery_20261010/`，中间运行也保留，不纳入 Git。

首次规划已经含 rules、D、A、O、decision 和输入边。规则 dense 取到当前 V8 / R17-R2 与旧规则片段，桥接取得完整 R17-R2；这时才实例化新增≤600秒、删除后60秒起不得返回、脱网可用。D 的 dense 只给 DEL-08 异常线索，不绑定 false。桥接得到适用于 V8 的“180秒时仍返回”，程序用 `IF(违规见证, false, delete_compliant)` 得到 D=false；没有声称删除延迟恰好180秒或假设删除过程单调。独立的正面合规输入保持待绑定，单点未返回不能证明全程合规。联合约束排除完整合规与违规见证同时成立，运行测试明确检查此时为 inconsistent。

带实际来源的局部字段投影完成 D 节点，不伪造未使用的 `delete_compliant` 绑定。完整失败集合继续求解 A/O；joint 对照同样使用局部投影，但没有全局任务消去。安全外包络允许未知约束世界，同时单独检查已证明可行的世界；测试确认额外未知世界不能抹去已证明的非空性，且缺乏任何可行性证明时不认证常量输出。

| 案例 | 结果 | ANN | 逻辑生成 |
| --- | --- | ---: | ---: |
| 一个决定性原因 | decision=false，reason=D；A/O 仍未知且不再专属取证 | 4 | 7 |
| 全部失败项 | 完成 D、A、O，输出 [D,A] | 8 | 13 |
| DEL-08 改为 V7 | 不绑定 V8 的 D；后来可用独立 A 失败见证作答 | 15 | 15 |
| 实体链 | Lin Zhou → North University → River City | 3 | 7 |
| 两父数值比较 | 0<2，发布 true | 2 | 4 |
| 记录输出 | {x:0,y:2} | 2 | 4 |
| 多种输出仍可能 | {false,true}，未完成，prediction=null | 1 | 2 |
| 开放建议 | 有原文支持的语义终端，semantic_assessment | 1 | 4 |

额外运行测试在已有 D 证书之后撤销 DEL-08 来源，确认旧证书失效、D/A/O 重新活跃，暂停的桥接前沿仍在。另一测试让付费 audit 撤销实际事实，确认先重算，不发布旧 D 证书。纯符号终端没有假的 response/terminal_input；真实语义终端和终端格式任务有实际 response_ref。

局部读取也包含实际相关父绑定及其必要祖先，允许新原文撤销父规则，而不是将这条反证作为越界行丢弃。测试确认同批父规则撤销后重置内部程序、恢复规则需求，不能拿旧规则形成证书。事实终端的语义答案与标签在同次联合响应内；纯计算终端需要语义输出时才进行第一次终端生成，没有独立事后选项匹配器。

## 消融与计费口径

固定相同初始 DAG、虚构语料、源码驱动模型替身、向量回应逻辑及全题额度，四种方法均保留所有结果/未完成状态。在线对照还改变了 eager/pausable 检索控制，因此不是单因素的全部归因。固定完整候选池的最后一对单独检验任务消去。

最终消融数字从 `scripts/ablate_residual_memory.py` 的实际输出提取，记录在交付核验。计费分 planner、joint_read、repair、semantic_compose、audit；ANN 与逻辑生成、物理 LLM/embedding/rerank 尝试分开。set_score 是既有集合评分单位，不等于 rerank HTTP；不以失败很多的旧六题成本均值计算降本比例。没有准确率或硬件加速结论。

| 对照 | ANN | 生成 | set_score | rerank HTTP | 未完成 |
| --- | ---: | ---: | ---: | ---: | ---: |
| f4 scored | 22 | 11 | 28 | 12 | 0/1 |
| 匹配 unscored | 27 | 11 | 0 | 0 | 0/1 |
| joint unscored | 8 | 13 | 0 | 0 | 0/1 |
| full residual | 4 | 7 | 0 | 0 | 0/1 |
| 固定完整池 joint | 0 | 9 | 0 | 0 | 0/1 |
| 固定完整池 residual | 0 | 5 | 0 | 0 | 0/1 |

所有对照在同题都输出 decision=false、reason=D。该合成例子删除评分本身没有减少生成次数；联合阅读对照的生成次数也高于旧路径，不能笼统宣称合并协议必然降本。固定完整池对照只证明此例的残余需求控制减少了无必要读取。规划协议修复另外计作 repair，不和正常 planner 重复计数；物理重试仍保留在 Ledger。

## 固定真实 smoke

固定 HotpotQA 前三题与 PersonaMem 前三题、完整可见语料、每题单次尝试的 manifest 已写入新私有目录 `outputs/residual_memory_v1/smoke_3plus3_v1/`。兼容既有 Qwen 索引核验为 HotpotQA **9811** 文档、PersonaMem **3187** 文档，未重建全量索引。

LLM 与 embedding 的独立轻量检查均为 `RemoteDisconnected: Remote end closed connection without response`。正式 smoke 在 LLM `/models` 预检阶段同样失败，progress.json 保留具体错误。**六题均未执行**，没有真实成功预测、准确率或题内费用数据。生成 LLM、ANN、rerank 题内调用均为 **0**；端点预检 GET 另计，不冒充问题预算。reranker 未预检/调用，未使用金标，未扩大预算或反复运行六题。旧 `smoke_3plus3_v2` 的全部失败和费用保持不变。APC 历史状态仍为 `IMPLEMENTED_NOT_HARDWARE_VERIFIED`。

新方法因此只达成离线实现与验证，不能标成真实模型 smoke verified。模型蕴涵、规则完整性与域解释仍需真实服务验证。已知保守限制和实际命令见 [方法说明](DAGBT_RESIDUAL_MEMORY_V1.md)。推送只针对研究分支，不合并 main、不部署生产。

## 交付核验

最终源码执行 `python -m pytest tests -q`：**801 passed，0 failed，0 skipped，69 warnings，52.09s**。其中剩余语义与运行专项 **71 passed，3.54s**；69 条已有矩阵运算警告保留。完整日志保存在本机 `/tmp/dagbt-residual-final-tests-20261010.txt`，不提交 Git。

最终合成轨迹为 `outputs/residual_memory_v1/final_delivery_20261010/synthetic.json`；同目录 `ablation.json` 保存六个对照的全部结果。八个合成案例及六个消融对照的结果、调用量与上表一致；不确定布尔案例按设计未完成，预测为空。六个消融对照均完成，不据此推断真实模型质量。

最终 `git diff --check` 通过；108 个受保护 tracked 文件相对基线变更为 0，原始与 vendor manifest 各 57 项哈希通过。原工作区分支仍为 `feat/local-scoring-terminal-answer`，HEAD 仍为 `f4e38df23b5b621a4d0e759c018c60e1fe25dd9b`。提交范围仅为实现、配置、脚本、测试和文档；数据、凭据、向量、快照与原始轨迹不纳入提交。实际提交号及远端 HEAD 核验另在交付回复报告。
