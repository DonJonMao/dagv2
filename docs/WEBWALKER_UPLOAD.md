# WebWalkerQA v4 修复实验包（access v5）

这是独立 WebWalker 部署包。107 个公共 Python 核心文件保持与
`e6677081cdfb33c01db3aaa274a497168f314f78` 逐字节一致；两组仍为
`dagbt_joint_memory_v1` 和 `dagbt_residual_memory_v2`。

**当前先安装并运行 smoke，暂不建议启动 680 题全量。** Linux 旧版只有失败的 smoke，
没有正在等待的全量结果。执行测试通过、任务终止、生成答案和答案正确分别报告；
本版实际验证结果见 `validation.json`，不以旧版结果冒充新成绩。

## 安装

上传本包与同名校验文件到 `~/bt/`，执行：

```bash
cd ~/bt
sha256sum -c dagbt_webwalker_v4_access_v5_experiment.tar.gz.sha256
mkdir -p ~/dagbt_patches/webwalker_v4_access_v5
tar -xzf dagbt_webwalker_v4_access_v5_experiment.tar.gz -C ~/dagbt_patches/webwalker_v4_access_v5
export DAGBT_PYTHON="$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python"
bash ~/dagbt_patches/webwalker_v4_access_v5/dagbt_webwalker_v4_experiment/install_and_smoke.sh
```

默认新目标：`~/dagbt_runs/20261015_webwalker_v4_access_v5_experiment/dagv2`。
默认来源：`~/dagbt_runs/20261014_personamem_v4_experiment/dagv2`。
目标不能覆盖或嵌套在 PersonaMem 来源目录中，安装器不替换任何已有不同内容的文件。
包内不含私人凭据；从来源部署复用本地凭据到新目标，权限为 0600，原文件不修改。
可用第一个参数指定其他新目标，用 `DAGBT_PERSONAMEM_BASE` 指定来源。

若服务器已有 access_v2 的浏览器，可在安装前复制依赖与浏览器缓存，减少下载：

```bash
TASK_OLD="$HOME/dagbt_runs/20261015_webwalker_v4_access_v2_experiment/dagv2"
TASK_NEW="$HOME/dagbt_runs/20261015_webwalker_v4_access_v5_experiment/dagv2"
mkdir -p "$TASK_NEW/_webwalker_browser_vendor" "$TASK_NEW/_webwalker_browsers"
cp -a "$TASK_OLD/_webwalker_browser_vendor/." "$TASK_NEW/_webwalker_browser_vendor/"
cp -a "$TASK_OLD/_webwalker_browsers/." "$TASK_NEW/_webwalker_browsers/"
```

只复制这两个目录，不复制旧输出、页面快照或任务缓存。复制后仍执行正常安装命令，
安装器会检查依赖并实际启动 Chromium。缓存不完整时补下载。
若仍报告服务器最初的四个系统动态库缺失，执行：

```bash
sudo apt-get update
sudo apt-get install -y libatk1.0-0t64 libatspi2.0-0t64 libxdamage1 libasound2t64
```

安装器本身不运行 sudo、不更改旧虚拟环境。浏览器下载需要 PyPI 和浏览器 CDN 可访问。
仅使用 HTTP 时安装命令可加第二个参数 `--no-browser`。PDF 依赖随包提供。
安装后的 offline smoke 使用本地网页和模型 HTTP 夹具，不访问真实模型或外站。

## 运行 smoke 与核对结果

```bash
cd ~/dagbt_runs/20261015_webwalker_v4_access_v5_experiment/dagv2
bash scripts/run_webwalker.sh smoke
```

smoke 默认官方前两题、两组共四任务；结果写到
`outputs/webwalkerqa_v4_access_v5_smoke`，完成生成后独立评分。
它先核对 PersonaMem 的真实 OS writer 锁与 `complete` / `complete_with_failures` / `stopped`
状态。`stopped` 是正常手动停止，不代表全量完成；没有活跃 writer 时可以试跑。
不自动停止或修改 PersonaMem。

核对以下结果：

- `progress.json`：任务是否全部终止，失败数与环境跳过数。
- `tasks/*/*/result.json`、`fusion_events.jsonl`：完整初始输出要求是否保留，实际链接与阅读、修复、义务及终端状态。
- `evaluation.json`：共同可访子集的正确率、有效题数和覆盖率。
- `generation_summary.json`：实际 LLM 请求与 token、网页动作和耗时，含缺失 usage 数。

只读诊断命令：

```bash
"$DAGBT_PYTHON" scripts/webwalker_diagnose.py \
  --output outputs/webwalkerqa_v4_access_v5_smoke --question wwqa-0001
```

没有答案、预算用尽、规划/阅读违规与实际容量不足都保留为失败及成本；
不把它们当网站故障删题。模型持续违反协议时，有界修复仍可能失败。
不要在原失败目录换代码重跑；源码、配置、数据和方法身份冻结，变更后须用新输出。
本包的 full/status/stop/score 接口仍保留，本次交付不自动启动全量。

## 本版针对的三个缺陷

1. **规划修复丢失输出要求。** 初始 DAG 在首个网页访问前完成。Web 准入检查所有计划任务
   与真实输出依赖的连通性，并在同一次有界修复中保留先前识别的输出分支和 record 字段。
   修复不能将多属性问题缩成只输出一个属性来通过类型校验。异类属性优先声明明确字段的
   record，或由完整 compose 输出。合法条件分支仍可在后续剩余求值中消去；
   结构检查不能独立证明初次规划已理解自然语言中的每个要求。
2. **导航被界面噪声和固定前缀窗口带偏。** 完整原文、raw 快照、链接前沿均保留，另标识
   DOM 正文、导航和界面区域，为导航提供正文与链接附近的原文上下文。链接窗口公平推进，
   已接受的同作用域阅读反馈交给下一次导航；反馈不是事实证据，不生成新 URL。
3. **旧观察重复修复与长页面饿死其他子任务。** 新窗口明确标注新旧来源角色，携带持续的
   冻结观察规则。Web 调度轮转当前可执行节点；长页面没有新证据时允许让出一次导航机会，
   随后恢复尚未读完的前沿。完整旧原文、失败草稿、反证和未解义务仍保留。

预算保持 24 次 LLM、6 次 JSON 修复、每请求最多 2 次修复、16384 总上下文、4096 输出、
36 次 web action。所有真实请求按部署 tokenizer 精确计数。修复预留不足或强制闭包超限
仍明确拒绝，不裁掉旧证据、不扩预算、不靠免费重试改写结果。

## 实验边界与来源

只从题目给定 root URL 沿实际观察到的 `<a href>` 链接访问。没有全局搜索、黄金路径导航、
答案页面向量库、reranker 或独立最终 Reader；导航边不是事实支持边。
两组每题共享首次观察的页面快照，但各自必须从已发现链接到达。网页访问动作和论文 A.C.
口径不同；模型/请求预算不因网站变化而增加。

根页在有界 HTTP 尝试及普通隔离 Chromium 回退后仍不可用，才按题目配对排除并报告覆盖率。
中间链接失败只关闭该链接。浏览器依赖、缓存损坏、解析、模型和推理错误均不当作环境排除。
不使用私人浏览器 profile、stealth 或验证码破解。页面 raw、原文和哈希用于复核。

官方数据为 WebWalkerQA main 680 题，revision
`c7ab3329a2a4a35ca569fdca369b61a3d1787e5b`，原始 SHA256
`cbe95d9de714ef52fe4a9ddf160dfab462407cb1ba526f1c31b4305ca32c34cb`。
来源（Apache-2.0）：https://huggingface.co/datasets/callanwu/WebWalkerQA 。
生成只读取 id/question/root_url；参考答案、golden_path 与 source_website 只供全部生成终止后评分。
judge 模型及配置身份写入报告，当前 judge 与论文 GPT-4 judge 口径不同。
共同可访子集成绩不是官方全 680 题准确率，也不能据此宣称方法优势。

旧 access_v1/v2 的服务器失败与 access_v3/v4 的本地失败保留，不作为本版通过证据。
本包验证平台、真实 smoke 结果及仍存在的限制，以 `validation.json` 为准。
