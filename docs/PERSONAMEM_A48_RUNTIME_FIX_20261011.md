# PersonaMem a48 v3：导航容量与来源引用协议修复

基线为 `54c3baefca943320b7a173f13dec381344d3feea`（已安装的 a48 v2 补丁），
方法仍为 `dagbt_residual_memory_v1`，对照仍为 `original`。
不迁移 support-review/v4 分支，不修改原始算法目录或既有数据/向量。

## 日志确认的问题

上传的 `dagv2_a48_v2_smoke_debug.tar.gz` 包含两道 PersonaMem smoke 题。
旧运行已结束，状态为 `complete_with_failures`：original 两题执行成功，
residual 一题 `execution_failed`、一题 `budget_exhausted`。
执行成功不表示答案正确。本次修复与验证不使用评价标签。

1. **导航查询超过 embedding 容量。** BT 路径累计九条原文后，查询达到
   22,447 字符。一次原样重放得到明确的 `input_tokens` HTTP 400：服务上限
   4096 tokens。上一条成功请求服务计数为 3958 tokens。
   本地包内 tokenizer 对全部 163 个成功 embedding 请求、3250 条输入的
   计数，加每条输入一个附加 token，均与服务 usage 对齐；失败查询本地为
   4405 tokens。此证据不等于独立核验了服务端 tokenizer 或模型权重哈希。
2. **读取协议混用两套来源 ID。** 输入同时展示 `sN` 和 `src_<hash>`，
   校验仅接受前者。22 次读取/修复逻辑响应中，342 次引用精确匹配当前可见的
   稳定来源 ID，却被拒绝；另有 9 次引用确实不匹配，不能放行。
   该题使用 23/24 次 LLM 额度，剩下一次为审计预留。16 次读取中 15 次失败，
   并消耗全部 6 次格式修复。两次修复请求完全重复，只重放同一缓存响应。

## 修复边界

- 仅对检索导航文本生成有容量约束的视图；完整原文和后续证据引用保持不变。
  计数包含 query instruction 与最终序列化查询，保留固定问题/任务/需求，
  对原文片段记录来源、区间、哈希及省略情况。固定字段本身超限则明确失败。
- 读取请求使用统一 alias 命名空间；兼容当前可见来源的一一对应稳定 ID。
  仍拒绝虚构、不可见或歧义引用，仍校验来源内容、角色和坐标。
- 格式修复携带失败观测及反馈，保留观测 ID 和已验证内容；重复的修复内容
  不再重复发出。实际修复输入不够容量时明确记录，失败义务继续保留。
- 每次读取/修复都明确列出要求返回的变量 slot，并把该名单写入本次 schema。
  父节点复核仍须显式返回；任务背景中的终端变量不能替代请求中的父变量。
  这是 plain 模式中的协议提示与本地校验，不宣称服务端强制 JSON schema。
- 按每个变量声明补齐响应的 JSON 类型约定：集合为数组，记录为对象等；
  数值仍允许原协议中的标量、带单位数值和区间。明确本次可撤回的既有
  路线 ID；没有既有路线时不能返回无目标撤回。运行时校验继续严格执行，
  不将错误类型自动转换为事实，失败反证仍须解决。
- 失败反证不能因后续缺少 slot、格式错误或重复 ID 而消失；同轮重复 ID
  不能被“已冻结”分支跳过。修复响应只可引用该轮实际可见的来源。
- 多来源读取将完全相同的文档 ID 和时间元数据共享到一个表中，只改变
  请求中的表示方式；来源字段名只列一次，片段按固定列序完整传输。
  joint 读取的 JSON 去除分隔符空格，字符串内部的空白保持原样。
  每个片段的原文、来源 alias、坐标、角色和消息索引
  完整保留；本地权威来源与输出 provenance 不变。预算估算和实际请求使用
  同一编码，审计指令也计入分窗容量，不删除 OR 证明路线、反证或历史撤回记录。
- 集合变量的一条 support 表示完整变量值，多条 support 是替代证明。
  不把不一致集合自动取并集，不把局部证据强行转换成完整事实。

预算保持 ANN 36、LLM 24、JSON repairs 6、每次请求最多修复 2 次；
不增加独立 Reader、不弱化证据约束、不读取标签来选择答案。
模型、endpoint 与凭据解析配置保持原样。完整验证结果随交付包记录。

## 16K 容量计数修复与文献对齐

参考 [Preference RAG 基线调研](PREFERENCE_RAG_BASELINE_PROTOCOL_20261012.md)，
不将 PersonaMem 的 32K 历史版本当成请求上限，也不单独把 residual 扩容到 24K。
ANN 36、LLM 24、输出预留 4096、输入余量 256 和上下文 16384 保持不变。

第五轮开发 smoke 仍有一个终端窗口被旧本地估算拒绝；它没有发送生成请求。
已用实际部署 `/tokenize` 对照此前 10 次 residual 和 6 次 original 的完整
chat 请求，16 次计数均与已有服务 `prompt_tokens` 相同。原先失败首窗口实际
输入 8650，带 repair envelope 为 8830；计入输出和全部余量后为 13010/13190，
在 16384 之内。这只验证该窗口，不代替完整新 smoke 的结果。

新增配置 `configs/residual-memory.personamem16k.json`，与原配置相比只显式启用
同源部署计数接口。分窗、Reasoner 发出前检查和 Transport 最终检查共用同一
完整消息计数，包括 system、schema、原文及 generation prompt；不按经验比例
换算，不减输出预留。服务报告的最大容量只作为额外约束，不用于放大实验上限。

两组最终聊天请求均使用同一服务计数门禁，并要求生成返回的 `prompt_tokens`
与计数一致；不一致或缺失则记录失败，保留原响应，不重复生成来掩盖差异。
计数接口失败不降级为猜算。旧 original 算法内部原有估算预检仍保留，可能更保守；
没有改其规划、检索或选择逻辑，不能据此声称各方法的全部中间预算/算力相同。

计数仅支持已验证的纯文本、plain schema 提示和服务默认聊天模板；工具、多模态、
自定义模板及未知协议字段明确拒绝。计数缓存只在单个 Transport 生命周期内，
按实际消息和模型区分；它不缓存或改写证据。计数 HTTP、缓存命中和耗时单独
写入 `tokenization_events.jsonl` 与成本统计，不计入 LLM/ANN 调用或生成 token 总量。
模型权重及 tokenizer 文件哈希未被独立核验。

## 本次验证

- 完整套件 1020 项通过，90 项可选 JSON Schema 测试在另一个具备依赖的环境
  补验通过；原有数值运算警告单独记录在 `validation.json`。
- 上传失败日志中的同两题重新运行，单次尝试：original 2/2 `ok`，residual
  2/2 `ok`，后者两题均完成审计且没有未解决失败义务。
- 18 次最终聊天请求的实际输入计数与生成 usage 全部一致；最大请求连同输出
  和余量为 16199 / 16384。没有扩大预算，也未对答案准确率评分。
- 计数存在额外 HTTP 开销：residual 两题分别为 251 / 787 次、约
  19.56 / 83.59 秒。两组共 1044 次、约 103.59 秒，单列在辅助成本中。
  两题执行通过不代表全部 589 题已验证，也不代表答对。

## 安装与运行

补丁要求未手动修改的 a48 v2 部署。安装器验证文件哈希，检测目标目录
`outputs/` 下的活动 writer 锁并拒绝覆盖，
自动备份，失败回滚，最后更新 `DEPLOYMENT.json`。重复安装同一补丁为幂等操作。

上传 `dagv2_personamem_a48_v3_patch.tar.gz` 到服务器 `~/bt/` 后：

```bash
mkdir -p ~/dagbt_patches/personamem_a48_v3
tar -xzf ~/bt/dagv2_personamem_a48_v3_patch.tar.gz -C ~/dagbt_patches/personamem_a48_v3
bash ~/dagbt_patches/personamem_a48_v3/install_and_smoke.sh
```

查看状态：

```bash
cd ~/dagbt_runs/20261011_personamem_a48/dagv2
export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
"$DAGBT_PYTHON" -m dagbt.runner status --output outputs/personamem_a48_v3_smoke
```

两题运行结束且状态为 `complete` 后，启动完整 PersonaMem 实验：

```bash
cd ~/dagbt_runs/20261011_personamem_a48/dagv2
export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
"$DAGBT_PYTHON" -m dagbt.runner launch --config configs/residual-memory.personamem16k.json --datasets personamem --arms original dagbt_residual_memory_v1 --output outputs/personamem_a48_v3_full
```

代码改变后必须使用新输出目录，不将旧运行的成功/失败行混入新版。
当前每题最多尝试一次，不能承诺对已耗尽尝试的旧任务使用 `--retry-failed`
即可重跑。这里的运行是推理评测，不是更新模型权重的参数训练。
