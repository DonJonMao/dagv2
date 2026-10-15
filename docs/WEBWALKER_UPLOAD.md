# WebWalkerQA v4 网站容错与协议修复实验包（access v3）

本包以你指定的 PersonaMem v4 实验包为安装基线，公共求解核心保持逐字节一致：

- 源码提交：`e6677081cdfb33c01db3aaa274a497168f314f78`。
- 输入包 SHA256：`88b72a8e64b64ba4e33e806fb8023c3d24da333a1a467eb831b2e1dd47f9e4fc`。
- 方法：`dagbt_joint_memory_v1`（完整 DAG 对照）和 `dagbt_residual_memory_v2`（剩余求值）。

这是安装到独立目录的网页后端和实验 runner，不修改 PersonaMem 核心、配置、凭据、源码身份或输出。
不能把新适配运行在旧 `a5b993a` 核心上；包内提供精确的 v4 核心，安装器同时校验来源部署与包内的 107 个公共源码文件。
先安装你已有的 `dagv2_personamem_v4_experiment.tar.gz`，再安装本包以复用其本地凭据。
它不是 PersonaMem 的 `original` 对照，也不是官方 WebWalker baseline。
新输出必须使用新目录，旧 WebWalker 结果不能混入新版实验。

## 上传与安装

上传 tar 和同名 `.sha256` 到 `~/bt/`：

```bash
cd ~/bt
sha256sum -c dagbt_webwalker_v4_experiment.tar.gz.sha256
mkdir -p ~/dagbt_patches/webwalker_v4_access_v3
tar -xzf dagbt_webwalker_v4_experiment.tar.gz -C ~/dagbt_patches/webwalker_v4_access_v3
bash ~/dagbt_patches/webwalker_v4_access_v3/dagbt_webwalker_v4_experiment/install_and_smoke.sh
```

默认 Web 目标：`~/dagbt_runs/20261015_webwalker_v4_access_v3_experiment/dagv2`，与 PersonaMem 分开。
默认凭据来源：`~/dagbt_runs/20261014_personamem_v4_experiment/dagv2`。
其他 Web 目录可作为第一个参数传入；来源目录用 `DAGBT_PERSONAMEM_BASE` 指定。
不允许目标覆盖来源部署或嵌套在其目录中。Python 优先使用目标的 `.venv`，否则沿用
`~/dagbt_runs/20260930_v3/dagv2/.venv/bin/python`；可用 `DAGBT_PYTHON` 指定。
安装不替换任何已有不同内容的文件，重复安装幂等，失败回滚新添加文件。
包内不含私人凭据；安装时在独立目标内以 0600 权限复用来源的本地凭据，原文件不修改。新增 `configs/webwalker-v4.json` 沿用 v4 的模型、tokenize 与
私人凭据路径，预算仍为 24 次 LLM、6 次 JSON 修复、16384 总上下文、36 次网页访问。
需要改变模型服务配置时复制这份配置，设置 `DAGBT_WEBWALKER_CONFIG`，并使用新输出。

已按上一版安装好浏览器的服务器，可以先复制浏览器依赖与缓存，避免重复下载。
只复制以下两目录，不复制输出、页面缓存或任务缓存。下面命令适用于本次已有安装：

```bash
export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
TASK_OLD="$HOME/dagbt_runs/20261015_webwalker_v4_access_v2_experiment/dagv2"
TASK_NEW="$HOME/dagbt_runs/20261015_webwalker_v4_access_v3_experiment/dagv2"
mkdir -p "$TASK_NEW/_webwalker_browser_vendor" "$TASK_NEW/_webwalker_browsers"
cp -a "$TASK_OLD/_webwalker_browser_vendor/." "$TASK_NEW/_webwalker_browser_vendor/"
cp -a "$TASK_OLD/_webwalker_browsers/." "$TASK_NEW/_webwalker_browsers/"
bash ~/dagbt_patches/webwalker_v4_access_v3/dagbt_webwalker_v4_experiment/install_and_smoke.sh "$TASK_NEW"
```

复制缓存后安装器仍检查缓存并实际启动浏览器；本次旧部署已安装 Playwright 1.56.0，系统动态库已安装时无需再执行 sudo。
缓存不完整时按正常流程补下载。使用独立副本，避免浏览器安装时修改旧部署。

默认下载隔离的 Playwright/Chromium，需要 PyPI 和浏览器 CDN 可访问。
不修改原虚拟环境、不执行 sudo、不自动安装系统库；浏览器无法启动时明确失败。
只需 HTTP 时，安装命令可加第二个参数 `--no-browser`。PDF 依赖随包提供。
安装后的 offline smoke 使用本地网页和模型 HTTP 夹具，不访问真实模型或网站。
本地验证平台及远端环境验证范围以包内 `validation.json` 为准。

## PersonaMem 完成或正常停止后运行

进入安装目标：

```bash
cd ~/dagbt_runs/20261015_webwalker_v4_access_v3_experiment/dagv2
bash scripts/run_webwalker.sh smoke
```

检查 PersonaMem 的真实 OS writer 锁及 `complete` / `complete_with_failures` / `stopped` 状态。
若需先停止 PersonaMem，在其部署目录运行
`bash scripts/personamem_v4.sh stop --output outputs/personamem_v4_experiment_full`，
再用 status 确认 `live_writer=false` 且 `state=stopped` 后启动 WebWalker。
`stopped` 只表示手动停止，不代表 PersonaMem 全量完成；已发布结果保留。
默认检查来源部署的 `outputs/personamem_v4_experiment_full`；其他输出可作为 smoke/full 的第二个参数传入。
有失败题但已终止的 PersonaMem 允许开始，不要求 14 题全成功，也不停止旧实验。
smoke 默认官方前两题、两组共四个任务，输出 `outputs/webwalkerqa_v4_access_v3_smoke`。
完成后评分；执行完成不等于答对。算法实验不以全题成功作为发行或全量启动条件。

全量 680 题、两组共 1360 个任务：

```bash
bash scripts/run_webwalker.sh full
bash scripts/run_webwalker.sh status
tail -f outputs/webwalkerqa_v4_access_v3_full/launcher.log
```

全量预检要求模型与部署 tokenize 可用；首个网站仅作诊断，其访问失败不会否决全部题目。
生成完成后自动评分。`live_scorer=true` 表示评分仍在运行，
`evaluation_available=true` 仅表示已有报告；需同时检查报告的实验身份。
每题超时 3600 秒，沿用原模型请求、JSON 修复和容量约束。
重复启动同一身份只继续未开始的任务；已失败或中断的单题尝试不会免费重跑。
若上次协调器丢失但题目 worker 仍在，拒绝重复启动；待其结束后读取原结果。
代码（含公共核心）、配置、数据、题目或方法变化会拒绝续写；重新实验使用新目录：

```bash
export WEBWALKER_OUTPUT=outputs/webwalkerqa_v4_access_v3_second
bash scripts/run_webwalker.sh full
```

停止指定 WebWalker 运行（只对身份匹配且持有 OS 锁的进程发信号）：

```bash
bash scripts/run_webwalker.sh stop outputs/webwalkerqa_v4_access_v3_full
```

停止不会停止 PersonaMem。Ctrl-C / SIGTERM 清理自己的 worker。
重新评分：`bash scripts/run_webwalker.sh score`。独立 judge 配置可用
`DAGBT_WEBWALKER_JUDGE_CONFIG`，judge 模型和配置身份会记录。
默认仅为配置中的 LLM/tokenize 主机加入 NO_PROXY，网页仍按原代理环境访问。
模型服务需要代理时可设置 `DAGBT_WEBWALKER_DIRECT_MODELS=0`。

## 对齐 v4 修复

详细逐项对应见包内 `V4_ALIGNMENT.md`。公共核心直接采用你指定包中的版本，
并非在旧方法外附加提示词或只改方法名。

1. 首次网页访问前仍规划完整任务 DAG，未知实体与规则保留待绑定输入。
2. 普通网页候选通过 `discovery_doc_ids` 进入探索前沿；每个实际阅读批次原子处理
   事实、反证和有界修复后交还剩余控制。未尝试候选可暂停；实际失败、带源未决、
   必要共同前提和规则完整性仍阻塞相关发布。候选暂停不证明网站已穷尽。
3. 合成和审计按当前证明及真实依赖提供完整必要原文，不自动重复整个父网页。
   反证、未解义务和失败草稿不因容量不足被裁掉。闭包超限仍明确失败。
4. 缺父输入的未来目标仅预测容量，不能否决已合法的局部事实；接地后重新严格检查。
5. 输入共享、修复冻结 ID、失败范围和版本约束使用 v4 相同实现；完整请求仍走
   部署 tokenizer 计数，不扩预算、不换身份洗掉失败。
6. Web runner 冻结公共核心及适配源码；`complete` 在逐题结果、成本和哈希索引发布后
   才写出。评分先核对全部任务与结果哈希，judge 缓存绑定实际题目、预测、参考答案、
   生成结果和 judge 配置。数值 0、布尔 false 不当成空回答。

PersonaMem 的个人历史摘要修订、响应选项、阶段脱敏和历史输出缩额权限依照任务
种类控制，网页问答不伪装成个人历史，也不引入选项字母或个人摘要修订权限。
开放网页问答仍必须真实语义终端与审计，完成级别为 `semantic_assessment`；
符号任务的证书是当前声明、合法来源和约束条件下的输出证书。

## 网站访问与数据

给定 root URL，沿实际观察到的 `<a href>` 链接在线访问；没有全局搜索、oracle
URL 导航或黄金页面向量库。原 ANN 初始检索替换为根页，桥接替换为原文条件化的
超链接选择，无 reranker 或独立最终 Reader；导航边不是事实支持边。

根页、缓存和失败访问均计一次 web action；HTTP 重试、浏览器资源请求、缓存、
LLM、部署计数 HTTP 和时延分别报告。两组共享每题首次观察的页面快照，但各自
必须通过已发现链接到达，方法顺序逐题交替。访问动作与论文 A.C. 口径不相同。

官方 main 680 题固定 revision：`c7ab3329a2a4a35ca569fdca369b61a3d1787e5b`。
原始 SHA256：`cbe95d9de714ef52fe4a9ddf160dfab462407cb1ba526f1c31b4305ca32c34cb`。
数据来源（Apache-2.0）：https://huggingface.co/datasets/callanwu/WebWalkerQA 。
生成只读取 `questions.jsonl` 的 id/question/root_url；答案、`golden_path`、
`source_website` 和其他 info 仅供生成全部终止后的独立评分。
仅已确认起始页不可访问的题从主指标分母排除，两组使用同一个题目子集；失败与未回答仍计 0。
judge 服务失败记 unknown 并报告准确率上下界。起始页排除后的成绩不是官方全 680 题准确率。
默认 judge 沿用原模型；论文 GPT-4 judge 与当前口径有差异，不能直接排名。

auto 后端读取 HTML/文本/PDF，检测 JS shell / 访问验证时尝试普通隔离 Chromium。
HTTP 403/429、网络错误或 200 验证页面会在有界 HTTP 尝试后回退一次普通 Chromium；
不使用私人 profile、stealth、验证码破解或关闭 TLS。仍被阻挡的起始页标记环境跳过。
只支持显式 URL 链接，不填写表单。数据发布后网页可能变化，保存快照和哈希供核验。

主要输出：`manifest.json`、`progress.json`、`generation_records.json`、
`generation_summary.json`、`access_exclusions.json`、`evaluation.json`、`tasks/`、`pages/` 与 `judge/`。
源码一致、离线回归和安装成功只能证明对应机制与执行条件；不证明真实答案准确率
或方法优于对照。真实模型验证状态及失败必须查 `validation.json`，不将旧版失败
或其他提交的成功写成新版成绩。

## 保留：访问失败时继续实验

- 首次访问网站前仍由原规划器生成 DAG，不额外预抓取题目网页，也不读取参考答案筛题。
- 每次访问最多使用配置中的 1–3 次 HTTP 尝试；可重试错误采用最多 2 秒的有界退避。
  auto/playwright 后端至多再尝试一次普通 Chromium，沿用原超时与 36 次 web action 限制，
  不给 LLM、JSON 修复或问题尝试增加预算。
- 数据集给定的起始页在上述尝试后仍返回拒绝/限流/其他 HTTP 错误、网络失败、
  验证页面或空页面时，当前题记 `skipped_environment`。第一组被跳过后，第二组不再重复
  请求该题；两组统一排除同一题。每题单独处理，不永久屏蔽整个域名或后续题目。
- 已经打开起始页之后，某个中间链接失败只关闭这个链接，继续探索其他已观察的链接。
  这类失败不自动排除题目，避免因两种方法选了不同链接而产生筛选偏差。
- 模型/规划/协议/推理失败、浏览器未安装或启动失败、缓存损坏、磁盘或解析错误均不当作
  网站不可达排除；保留失败与成本。普通任务失败不会终止后续题目，人工停止仍生效。
- `progress.json` 单列环境跳过任务数；`access_exclusions.json` 记录题号、URL、原因及
  HTTP 状态；`evaluation.json` 报告两组相同的计划题数、排除数、覆盖率与有效题准确率。
  `all_planned_accuracy_*` 另将被排除题按失败保留在原计划分母，供核对。全部跳过时
  有效题准确率为 null，不显示为 0 或 100%。所有已发生的请求成本仍统计。
- 新协议和源码身份拒绝续写旧结果。默认安装到新的 `20261015_webwalker_v4_access_v3_experiment`
  目录，输出也改为 `webwalkerqa_v4_access_v3_*`。已安装旧包时使用这个新目录，不覆盖旧的 access_v2 或 access_stopped 安装。

本地的容错测试通过不代表远端 Linux 上能绕过网站限制，也不代表已跑完整官方评测。


## 本次新增：规划修复与网页搜索缺口（access v2）

服务器首轮 smoke 的 `wwqa-0001` 暴露两条独立失败路径：规划器反复返回 tuple 表达式但
声明 text 输出，或将终端依赖之外的变量用于输出；JOINT 将普通页面未找到答案写入
`unresolved`，使强制保留原文从 9 段增至 14 段、50 段，实际修复请求超过 16K 上限。
这些是旧版失败记录，不能在原输出目录覆盖或免费重跑。

- Web 初始规划的有界修复携带完整的上一次拒绝草稿、原始问题和累计错误。
  `tuple` 返回 set、`record` 返回 record、终端依赖和输出类型继续严格检查；不将非法计划
  强行转换为合法计划。模型连续返回非法草稿仍会按原预算失败。
- 普通网页阅读的每个槽位必须声明 `reading_status`。`no_relevant_evidence` 必须是空观察、
  空 `unresolved`，不得声称解决旧义务；可附 `search_gaps` 作为后续检索方向。
  `evidence_assessment` 必须提交带来源的具名观察，或显式引用本次获准复核的冻结观察；
  具体疑点使用带源 `partial` / `unknown` 观察表达，`unresolved` 保持空数组。
  程序只校验结构、来源、范围和版本，不按自然语言关键词将旧疑点自动改成搜索缺口。
  没有相关证据时变量仍未绑定，不建立事实、强制原文闭包或输出证书。
- 可识别槽位的旧式 `unresolved` 描述在响应信封校验前先保留完整原文范围；遗漏状态、
  重复槽位、额外字段或混写后的空读修复不能令旧疑点消失。若此前误报的缺失实际只是
  原文不相关，模型必须重新审阅所提供的全部原文，以合法带源 `irrelevant` 观察明确复核，
  并按核心原有规则显式解决获准的义务 ID。它不能据此生成答案事实或排除真实反证。
- 新页面首次普通阅读按真实 wire/tokenizer 计数，为可能的失败草稿和修复请求保守预留
  空间。实际修复仍重新精确入场检查；不扩充 16384 上下文、不缩减强制证据，不能保证
  所有真实闭包都装得下。Web 不获得 PersonaMem 历史摘要改写或输出缩额权限。
- 新增独立诊断脚本：

```bash
"$DAGBT_PYTHON" scripts/webwalker_diagnose.py \
  --output outputs/webwalkerqa_v4_access_v3_smoke --question wwqa-0001
```

它只读结果和调用记录，不读取凭据或参考答案。新版预检、回归、真实 smoke 结果以
`validation.json` 为准；通过执行检查不能替代准确率评测。

## access v3：旧失败与新页面混合请求的容量预留

服务器 access v2 smoke 仍出现 RESIDUAL `InputOverflow`：实际修复输入 12399 tokens，
加 4096 输出和 264 余量超过 16384。旧失败草稿存在时，之前跳过了 Web 额外修复预测，
即使同一请求继续加入新原文。JOINT 也曾因重述冻结观察而留下失败阅读批次。

新版为这种混合请求也精确计入已有原文、失败草稿、义务覆盖反馈和冻结复核元数据，
再预留一份既有输出额度作为未知草稿空间。只缩小尚未尝试的新窗口，旧证据不删除。
持久恢复首个请求消耗的 JSON 修复次数仍计入原预算；没有后续修复额度时不会假装
还存在一次可付费修复。所有真实修复仍按完整 wire 再计数；未知草稿可能重复出现，
预测不是任意响应都能装下的保证，强制闭包自身超限仍明确失败。

`no_relevant_evidence` 明确表示本次没有新增可用观察：即使载荷含旧事实或旧疑点，
也可返回空观察，既有状态保持未变，不必重述冻结的 unknown/partial。真正疑点仍使用
带源观察；unknown/partial 不允许同时声称解决证据义务。修正失败草稿与解决事实疑点
分别校验，不自动重写模型回答或将真实疑点标成无关。

默认安装和输出改为 access_v3。access_v2 的服务器失败、本地试跑与实际评分均保留，
不会作为新版通过证据，旧结果目录不能在新版源码下续写。
