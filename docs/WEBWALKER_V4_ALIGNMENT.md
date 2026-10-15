# 从 PersonaMem v4 到 WebWalker 的修复对应

审查基准为用户指定的 tar，SHA256
`88b72a8e64b64ba4e33e806fb8023c3d24da333a1a467eb831b2e1dd47f9e4fc`；
其源码提交为 `e6677081cdfb33c01db3aaa274a497168f314f78`。
107 个公共 Python 源文件与 tar 逐字节一致，Web 代码在独立 worktree 开发，
原 PersonaMem v4、a5 和旧 Web 包不修改。发行包含精确 v4 核心，安装到独立 Web 目录，并从已安装的 v4 来源复用本地凭据。
不能直接在 PersonaMem 目录新增模块，因为它的 frozen_sources 会将这些文件也算入实验身份。

| v4 问题/修复 | WebWalker 处理 | 验证路径 |
| --- | --- | --- |
| 检索候选全部成为 pending；支持已足够仍被待读队列阻塞 | 同一 v4 reader/controller，普通网页候选显式传 discovery_doc_ids；实际尝试/失败/必要材料保留义务 | Web 实际 HTTP 多窗口早停测试；v4 discovery/frontier 和反证专项 |
| 批次未完成便早停、改查询清掉旧义务 | 保留 v4 原子槽位、源曝光、版本及付费修复；Web 子类仅增加搜索缺口与发现窗口预留 | v4 failed_batch、obligation、mandatory、frontier 专项 |
| 合成/审计重复整个父文档，或遗漏真实共同前提 | 同一 v4 mandatory_closure、实际证明投影及审计依赖 | v4 controller_source_scope、audit_dependency、proof_projection 专项 |
| 尚未接地的未来任务容量拒绝当前合法事实 | 同一 v4 handoff executable/advisory 区分 | v4 handoff_executability / planner_contract_alignment 专项 |
| 重复长值/复核材料、仅字节共享却增加 token | 同一 history/review wire，完整请求经部署 tokenizer 对比 | v4 review_wire_selection、history_wire、review_wire 专项 |
| 修复 schema 没有完整冻结 ID，旧观察被重复提交 | 同一 e667708 schema/local frozen ID 契约 | v4 frozen_ids_schema 专项 |
| 失败/中断/改身份获得新预算 | v4 reader 内部权限保留；Web 单题 max attempts=1，durable attempt marker，失败和中断不重试 | Web runner 中断、遗留尝试、活 worker、预算/超时专项 |
| 升级核心继续写旧结果、评分错用缓存 | Web manifest 增加全部核心源码；任务结果哈希索引、评分前核验和输入绑定 judge 缓存 | Web core upgrade、missing/tampered result、judge identity 专项 |
| 完成状态早于实际结果发布、误杀其他实验 | complete 最后发布；stop 核对真实 OS 锁、输出目录和 PID 命令 | Web complete publication、真实子进程 stop / stale PID 专项 |
| v4 PersonaMem 终端偏好事实被当成响应答案 | 保留个人任务专属契约；Web 不设置个人 task kind/选项，不套用字母或历史摘要权限 | v4 personal/phase 专项；Web 符号终端和开放语义问答专项 |
| 单个网站预检失败否决整套算法实验 | 仅模型/tokenize 是必要服务门槛，单站失败诊断记录，正式逐题继续；起始页明确不可达时两组配对排除并报告覆盖率 | Web website probe gate 专项 |
| 模型/tokenize 请求被环境代理干扰 | 仅为配置中的模型和计数主机加 NO_PROXY，保留网页代理；可显式关闭 | Web 网络配置专项与真实 preflight（若服务可达） |

适配不改变研究主线：DAG 在首个网页访问前规划；F 表示求解关系，F_t 控制当前
剩余需求；BT 在当前子问题的原文和出链中桥接；共同事实和反证联合阅读；真实
终端与审计决定发布。没有恢复评分 reranker、原问题全局补查或独立最终 Reader。

PersonaMem 专属修复不宣称全部在网页问答上触发。历史摘要的受源修订、个人阶段
边界和新历史初次输出缩额不授予普通网页事实。仍保留开放世界与容量限制；已
尝试未决不会因后来找到支持就自动清空。缺少父输入的容量预测不提供出版权限。

v4 包的验证文档本身说明尚有失败、14 题运行在打包时未结束，未建立准确率优势。
本次不把该报告的 2381 项测试或任何旧真实题成功当成自己重新执行的成绩；本次
实际测试、源码哈希、安装审计和真实服务状态单独写入新包 validation.json。


## 2026-10-15 access v1 补充

公共核心 107 个文件保持一致。仅 Web 访问器、WebEngine 适配、实验 runner 与脚本升级：
HTTP 拒绝/限流与网络错误可回退一次普通浏览器；起始页不可达通过独立的题级异常
交还 WebEngine，不混入公共控制器的局部 ConnectionError 处理。中间链接失败仍是
empty_probe，不能关闭其他前沿，也不能据此从分母删题。环境排除在两组间配对，
报告覆盖率、完整原因和全计划分母下的核对指标；模型/容量/配置失败仍算失败。
初始 DAG 的规划时机、预算和完整性证据不变。新协议和输出目录避免混用旧版结果。

人工切换实验时，PersonaMem 的 `stopped` 状态也允许进入 WebWalker，但仍先检查真实 OS writer 锁；
`generating`、`failed` 或 `interrupted` 不视为正常停止，不篡改旧实验状态或将停止描述为全量完成。


## 2026-10-15 access v2 补充

107 个公共 Python 源文件仍逐字节一致。WebReasoner 仅为初始规划增加完整拒绝草稿和
累计验证错误的修复载荷，沿用原始严格校验、全局共享预算和每请求最多两次修复。
WebJointReader 为普通 Web 槽位要求明确的 reading_status：没有相关证据时只允许空证据，
有证据时提交具名带源观察（或本次明确获准的冻结观察复核引用）；具体疑点采用带源
partial/unknown 观察，不再以无来源映射的自由 unresolved 字符串表达。
原始调用记录保留。旧式未决描述在信封校验前保守保留完整原文，畸形信封的空读修复
不能清除旧疑点；必须按核心原规则显式提交带源完整复核。搜索缺口不支持事实，也不
解除已存在证据或失败义务。该结构修订来自首轮可选字段真实 smoke 失败，不能将旧
2492 项回归通过或旧真实尝试计为新协议验证。

Web 新发现阅读窗口额外预测完整修复信封及未知草稿空间，实际发送与实际修复仍使用
部署 tokenizer 检查。终端、审计、语义求值和持久恢复继续采用原严格约束，未授予个人
历史摘要、whole-value 修订或输出缩额权限。强制证据闭包超限仍如实失败。
新 protocol、源哈希、安装目录与输出目录隔离旧 smoke；原失败结果保留，不续写或重用。

## 2026-10-15 access v3 补充

access v2 的服务器试跑证明，仅对无旧失败的新窗口预留修复空间不足。Web-only 预测现覆盖
旧失败恢复与新页面混合请求，完整保留已有失败和原文，并预测已知义务的覆盖反馈及冻结
复核元数据。恢复请求自身的修复消耗计入可用额度，所有实际请求仍精确检查 16K。
未知草稿的空间仍是保守预测，不构成任意无效响应可修复的保证；不授予 Web 缩减输出或
个人历史修订权限，不剪裁已尝试证据。107 个公共核心文件继续保持逐字节一致。

空观察可以表示没有新增证据，并保留已有义务，不再暗示必须重述旧未知观察。
schema 镜像核心的 settled-only 义务释放规则，不允许 unknown/partial 同时解决证据义务；
修正失败草稿仍可保留真实未知。access_v3 使用独立源身份、安装目录与输出目录。


## 2026-10-15 access v4 补充

仅增强 Web 阅读/审计的原文关系核对与修复字段说明。空 wildcard 修复提示由结构检查
限定，不用于丢弃 named draft、counter、whole-value 或持久失败恢复要求。
未修改原始响应、事实/义务状态、公共核心或调用预算。执行验收与准确率分别报告；
研究实验中合法的失败是结果，不能为了 smoke 答对无限重试。
