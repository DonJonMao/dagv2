# PersonaMem a48 v3.1：中间证据与最终答案的范围校验

安装基线是 `da64d56250733bc7a87942f6a5389b04a00176bd`（用户已安装的 a48 v3）。
方法仍为 `original` 与 `dagbt_residual_memory_v1`，使用原
`configs/residual-memory.personamem16k.json`。预算、模型、检索索引和凭据配置不变。

## 上传日志中的实际失败

`dagv2_a48_v3_smoke_debug.tar.gz` 的 SHA-256 为
`2df03ca7dee8cde6f49eef2c450e2a9fa3459f30e3468e8d18e839fd504f5760`。
原版两题成功；residual 第一题成功，第二题 `budget_exhausted`。
第二题原始问题要求一次有创意的周末出游建议。规划器将最终答案的
`time_scope` 设为 `weekend`，同时规划了询问过去兴趣的 `creative_interests` 子任务。

v3 把最终输出范围复制到所有中间变量的 `required_scope`，并要求每条支持或反证
与最终范围完全相等。首次读取和两次修复均返回历史证据对应的 `past`，于是被
`Observation scope does not apply to this task` 拒绝。后续读取无法解除这批失败
义务，控制器继续检索首个任务，耗尽 36 次 ANN 后未进入后续偏好和建议任务。
这次共 19 次 LLM、36 次 embedding 请求，全部 HTTP 成功；627 次辅助计数
HTTP 也均成功。状态名里的预算耗尽是后果，不能据此归因为额度配置过小。

上传运行与此前本地 smoke06 的 100 项源码哈希、配置、题目，以及语料、文档顺序、
passages 和向量哈希均一致。两边的最终时间范围都是 `weekend`；本地先前的
历史证据响应也填了 `weekend`，因而通过旧校验，服务器这次返回 `past` 才暴露
错误。本地执行通过不证明范围语义正确。此次回归测试必须覆盖“中间变量为过去、
最终答案为周末”这一差异，并重放实际失败读取。
诊断与修复不使用正确答案选择配置，也不将失败任务移出评测分母。

## 修复边界

中间变量使用预先声明的实体/时间范围；最终变量仍受不可变的输出契约约束。
复核父变量时按父变量自己的范围检查。范围不匹配仍然拒绝，不把 `past` 自动
改写为 `weekend`，也不通过清空失败义务、合并不同集合或删去来源来生成答案。
旧声明不带局部范围时保留原契约继承行为。

仍使用 16384 上下文、4096 推理输出预留、24 次 LLM、36 次 ANN、6 次全局格式
修复与每请求最多 2 次修复。v3 的来源协议、导航容量和实际部署计数修复保留。

## 只读计数接口的瞬时故障

本轮本地真实验证另发现：`/tokenize` 偶发连接/读取超时，会在证据读取发出前
终止任务。两轮验证均保留为失败，不能用随后两次成功的短时健康探测替代。
这不是上传日志中那一次 scope 失败的原因；上传失败题的计数请求全部成功。

针对新发现的传输问题，计数接口复用现有 `max_identical_attempts=3`，仅对瞬时
连接/读取错误有限次重发完全相同的只读请求。格式、计数、4xx 和重定向错误
不按此规则重试；生成返回 usage 不一致仍直接失败。请求内容、上下文上限和
LLM/ANN 预算不变，计数失败不降级到估算。重试次数、失败尝试及耗时单列，
重试后成功也不抹去之前的失败尝试。
配置总尝试次数允许 1–10（包含首次请求）；当前配置保持 3。瞬时服务错误
仅重试 HTTP 500/502/503/504，每次等待 0.25 秒递增、最多 1 秒。

## 验证

完整测试套件最终复跑 1093 项通过，92 项依赖可选 JSON Schema 验证器的检查跳过；
在另一个具备该依赖的环境补验作用域与读取响应契约，共 130 项全部通过，
覆盖上述 92 项。69 条既有数值运算警告保留记录。
第一次完整测试有一项停止进程测试超时，日志显示原有的进程启动/停止竞态；
未改生产代码，单项复查通过后再跑完整套件通过。失败记录和日志哈希均保留。

上传原始响应离线重放的 18 项检查全部通过：118 个可见来源均与权威原始语料
一致，回答、理由、原文和引用未修改。仅在测试的预声明变量范围改为 `user/past`
后接纳原始响应，终端范围仍为 `user/weekend`；错误主体、错误时间、终端范围
冲突以及篡改恢复状态均被拒绝。此确定性测试不等于大模型必然生成正确规划。

最终真实服务测试采用上传运行相同的前两题、配置及预算，两个方法共 4 个任务
全部 `ok`，每题一次尝试，不读取评测标签。新版两题的审计均完成，没有未读或
失败义务；第二题实际规划并保留 `past / present / weekend` 三种局部范围，
消耗 5 次 LLM、2 次 ANN。所有生成输入的服务计数与返回 usage 一致，连同
输出预留及安全余量均在 16K 内。测试源码哈希与交付源码一致。

最终这一轮没有触发计数重试；瞬时失败后恢复及耗尽行为由离线故障注入测试
覆盖，不能声称真实网络中的重试恢复已被验证。新版两题分别产生 251、605 次
辅助计数 HTTP，累计约 21.2、54.9 秒，仍有明显计数开销。
前两轮计数接口超时导致的失败也保留在验证报告中。
真实服务测试和安装器结果见随包 `validation.json`。两题执行通过不等于全量
可靠性或准确率；模型仍可能生成错误规划。

## 安装与运行

将 `dagv2_personamem_a48_v3_1_patch.tar.gz` 上传到服务器 `~/bt/` 后执行：

```bash
mkdir -p ~/dagbt_patches/personamem_a48_v3_1
tar -xzf ~/bt/dagv2_personamem_a48_v3_1_patch.tar.gz -C ~/dagbt_patches/personamem_a48_v3_1
bash ~/dagbt_patches/personamem_a48_v3_1/install_and_smoke.sh
```

安装器要求未手动修改的 v3 部署，检查目标 `outputs/` 内的活动 writer 锁，
验证哈希并备份；安装失败回滚。补丁不包含凭据文件，沿用服务器现有凭据。

```bash
cd ~/dagbt_runs/20261011_personamem_a48/dagv2
export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_a48_v3_1_smoke
```

只有 `live_writer=false`、`progress.state=complete` 且 `failed_arm_tasks=0`
同时满足才启动全量；不要求两题答案全对。

```bash
"$DAGBT_PYTHON" -m dagbt.runner launch \
  --config configs/residual-memory.personamem16k.json \
  --datasets personamem \
  --arms original dagbt_residual_memory_v1 \
  --output outputs/personamem_a48_v3_1_full
```

不复用 v3 的输出目录，不把此前失败记录改写为成功。本包不引入外部 RF-Mem 等基线。
