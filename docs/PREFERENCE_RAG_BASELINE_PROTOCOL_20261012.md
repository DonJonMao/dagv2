# Preference RAG 顶会 baseline 设置与 PersonaMem 对齐协议

核验日期：2026-10-12。研究对象是当前 a48 PersonaMem 实验的可比设置。
本文记录论文、正式会议页和固定公开源码的证据及决策；不代表已运行这些外部 baseline。

## 结论

最接近当前数据和记忆粒度的外部方法是 **RF-Mem（ICLR 2026）**。
PersonaMem 原论文自己的 **Dense RAG** 是更简单的基准锚点。
PrefEval 可补充偏好遵循的研究依据；A-Mem、MemoryOS 可作为后续迁移基线，
其原论文实验与当前 PersonaMem-v1 32K 不同，不能直接引用成绩作横向排名。

文献没有共同规定“16K / 24K 请求上限、24 次 LLM 调用”。数据历史长度、
检索条目数量、单请求总容量和每题总推理成本必须分别报告。
**本次不将 residual 的 context_tokens 单独从 16384 改为 24576。**
保留 16K 上限，新增显式启用部署 tokenizer 的配置以修复计数失配（见下文）；
这不是宣称 16K 来自论文。

## 已核实的设置

| 方法 / 正式来源 | 数据与检索单位 | 检索设置 | 模型与输出 | 与本项目的关系 |
| --- | --- | --- | --- | --- |
| PersonaMem RAG，COLM 2025，§4.4 / Fig.5 | PersonaMem-v1，32K 历史；论文称 message，未明确 turn 合并细节 | BGE-M3，top-5 messages；Mem0 对照是逐 turn 抽取后 top-5 facts | GPT-4o / GPT-4o-mini；该节未给统一请求 context/output cap | 同 benchmark 的直接基准；本项目当前 pair 切分和模型不同 |
| RF-Mem，ICLR 2026，Appendix B / 官方 run.sh | PersonaMem 32K；user query + 一条 assistant response 组成一条 turn memory | MiniLM；top-k=10，B=3，F=2，缩放 λ=20，高/低阈值 0.6/0.3 | 论文 PersonaMem 生成器写“advanced LLM”；公开代码 API 分支为 gpt-4.1-mini，输出 32 tokens、温度 0 | 与当前固定数据最接近；需移植至共同模型与严格 cutoff，不能宣称精确复现论文分数 |
| PrefEval RAG，ICLR 2025，Appendix A.4/A.7 | PrefEval 显式/隐式偏好任务；论文称 exchanges，代码选择 role-tagged 消息 | sup-simcse-roberta-large，top-5 | 主方法比较含 Claude 3、Llama 3、Mistral；代码输出上限 generation=300、MCQ=5 | 不是 PersonaMem；公开代码保留已有完整对话，再在末尾追加 top-k 内容 |
| A-Mem，NeurIPS 2025，Appendix A.5 / Table 8 | LoCoMo / DialSim；LLM 创建和演化的 memory notes | k 随模型/类别变化；GPT-4o(-mini) 各类别为 40/40/50/50/40 | 包括建库、链接和演化 LLM 调用；README 的 k=10 是代码默认，不能替代论文设置 | 后续迁移对照；需要另外统计建库成本和 cutoff 快照 |
| MemoryOS，EMNLP 2025 正式版，§4.1 | GVD / LoCoMo；多层记忆 | STM 7 pages、MTM 2000 segments、User KB / Agent Traits 各 100；MTM top-k=5/10，另有 LPM top-10 等输入 | 其最终输入还包括 STM、画像等；不能写成总共只有 5 或 10 条 | 后续迁移对照；正式版 STM=7，而 SDK 默认 10；正式版 MTM=2000，早期预印本为 200 |

RF-Mem Table 1 的 32K 部分报告平均输入：Dense 3515.9、RF-Mem 3566.6、
Full Context 24657.8 tokens。**均值不是上限**，也不是当前 DeepSeek 服务的 tokenizer 计数。

## 必须保留的论文 / 实现区别

1. **32K 是历史版本。** 当前固定版本共有 589 题、20 personas。它不代表每次 RAG
   请求必须使用 32768 tokens，也不是模型训练步数或训练集大小。PersonaMem 官方
   发布推理代码采用 cutoff 前完整历史；本次检查的代码未提供论文 §4.4 的 RAG 实现。
2. **top-k 必须带单位。** 五条 message、五个抽取 fact、十条 user–assistant pair、
   四十个 note 都不是相同的信息量。不能统一改成 top-5 就称公平。
3. **PrefEval 的 top-5 不等于全部输入。** 固定源码保留原对话并追加检索提示。
   它不能支持“所有偏好 RAG 只输入五条，因此几 K 一定够”的推论。
4. **生成答案与结构化推理的输出预算不同。** RF-Mem 的 32-token 与 PrefEval MCQ
   的 5-token 上限针对最终选项。a48 的 4096-token reserve 用于 typed values、
   来源引用及结构化推理，不能直接改成 32 或 5。
5. **公开脚本不能不审计就照跑。** RF-Mem 固定 driver 读取但未使用每题 cutoff；
   run.sh 是 60 组 alpha/tau 扫描，parser 默认 F=3，而脚本/论文 F=2。
   当前源码还访问未声明的 CLI 字段。上述事实不证明论文表格一定由该 driver 产生，
   但本项目必须修正可见性边界并记录有效参数，不能在全部 589 题上选最佳配置。

## 本项目采用的对齐口径

**采用统一模型下的受控复现，不把不同模型的论文分数作为我们的实验结果。**

- 固定 PersonaMem-v1 revision `fd7c30f071d5c2ee2a211506783be222d7b6002e`、
  同一批问题、原始四个公开选项及逐题 `messages[:end_index]`。
  正确答案只用于最后评分，不用于检索、规划或选择超参数。
- 继续统一使用已有 DeepSeek 生成服务和 Qwen3 embedding 服务、温度 0，记录实际
  wire model ID、服务返回 usage 和失败。模型名并不能独立证明部署的权重与最大上下文。
  使用这些后端移植 BGE/MiniLM 方法时，明确标注 model-aligned port。
- 最优先新增对照为 **Dense 与 RF-Mem**。若实现 PersonaMem 论文 RAG 的受控移植，
  应采用明确声明切分规则的 message/top-5；若隔离 RF-Mem 的检索算法收益，Dense 与 RF-Mem 应共用同一
  turn-memory pool 和 top-10。这两种对照分别命名，不能静默改变数据切分。
- 保留方法各自的机制、提示与中间结构化输出；原生设置与统一资源消融分开报告。
  同模型、同题并不等于同计算成本；a48 的多步推理不能称为单次 RAG 的等算力比较。
- 评分采用固定分母单选 accuracy，并另外列 execution failures、invalid answers、
  按 question type / persona 分组结果。当前项目严格单标签 parser 与上游的宽松
  marker/正则判定不同；所有本地对照共用当前评分器并公开这一差异。
- 成本分别报告索引/建库、在线检索、推理、最终作答、重试；列每题 LLM 调用、
  ANN 调用、输入/输出 tokens、峰值单请求容量、实际来源数和延迟。
  成功题条件下的准确率只能作附表，不能替代固定全题分母。

本节是确定后的复现协议；外部方法尚未接入当前 runner，不能把计划的对照行标作已经实现或完成。

## 当前 a48 修复的验收状态

部署计数适配前的代码 HEAD：`4c873732989dc2365af04428a60ada8974345ff9`，基于用户已安装的
`54c3baefca943320b7a173f13dec381344d3feea`，方法仍为 `dagbt_residual_memory_v1`。

第五轮开发 smoke：original 两题 `ok`；residual 一题 `ok`，另一题在完整来源闭包
进入终端时触发本地估计容量不足。后一题的读取响应合法，不是先前的 embedding 400
或 source-ID / value 类型协议错误；这仍然是未完成任务，不能包装成通过验收。
该 smoke 未使用评分标签计算准确率；执行成功不能解释为答对。

`package/frozen_flow.py` 原算法预检硬编码 16384，residual 配置也为 16384。
二者当前使用的本地 token 估算方式还不同；相同名义数值本身不能证明服务端实际
tokens 完全匹配。单独将 residual 提高到 24K 不符合共同资源对齐，文献也不为其背书。
因此后续修复选择核对准确分词计数，而非增加容量：现有服务 `/tokenize` 与此前
10 次 residual、6 次 original 的实际生成 `prompt_tokens` 全部一致。旧失败窗口
完整输入的部署计数为 8650，含 repair envelope 为 8830；保留 4096 输出、256
输入余量、8 safety 后为 13010/13190，低于 16384。没有按观测比例修正估算。

新增 `configs/residual-memory.personamem16k.json` 仅启用同源部署计数。两组最终
请求应用相同实际 context gate，residual 分窗使用同一计数；original 原算法内部
估算仍保留，不能宣称完全等算力。计数服务失败、生成 usage 缺失或不一致均拒绝，
辅助 HTTP 成本单列。

第六轮完整 smoke 已完成：同两题、单次尝试，original 2/2 `ok`，residual 2/2
`ok`；residual 两题均完成审计，没有未解决失败义务。18 次最终聊天请求的计数
全部匹配生成服务的 `prompt_tokens`，计入输出预留和余量后的最大值为
16199 / 16384。测试套件 1020 项通过；90 项可选 JSON Schema 测试在另一环境
补验通过。该结果只验证这两题的执行路径，未计算准确率，不能外推全量可靠性。

准确计数存在额外开销：两题 residual 分别发送 251 / 787 次计数 HTTP，
累计计数耗时约 19.56 / 83.59 秒；原因是保留原分窗流程，逐个检查不同候选前缀。
本轮未为降低该开销改变分窗语义。两组共 1044 次计数 HTTP、约 103.59 秒，
已与生成/embedding 成本分开记录；不能把它省略后宣称等成本比较。
完整验收以交付包 `validation.json` 为准。外部 Dense / RF-Mem 尚未接入或运行。

## 原始来源

- PersonaMem：[论文 v2（COLM 2025）](https://arxiv.org/abs/2504.14225v2)，
  [官方代码固定 commit](https://github.com/bowen-upenn/PersonaMem/tree/d07e6ade22e85e0e5d562247323a9c3e07553226)。
  主要依据论文 §4.4 / Fig.5、Appendix C / Fig.9；代码
  [cutoff 与输入](https://github.com/bowen-upenn/PersonaMem/blob/d07e6ade22e85e0e5d562247323a9c3e07553226/inference_standalone_openai.py#L216)。
- RF-Mem：[ICLR 2026 会议页](https://iclr.cc/virtual/2026/poster/10008269)，
  [论文 Appendix B](https://arxiv.org/html/2603.09250#A2)，
  [固定 run.sh](https://github.com/Applied-Machine-Learning-Lab/ICLR2026_RF-Mem/blob/d6a2cb3416e8320bcb1439921d319729aeccec8e/RF_mem/personamem_data/run.sh#L14)，
  [模型 / 检索 / 输出实现](https://github.com/Applied-Machine-Learning-Lab/ICLR2026_RF-Mem/blob/d6a2cb3416e8320bcb1439921d319729aeccec8e/RF_mem/personamem_data/main_batch.py)。
- PrefEval：[ICLR 2025 会议页](https://iclr.cc/virtual/2025/poster/29686)，
  [论文 A.4](https://arxiv.org/html/2502.09597#A1.SS4)，
  [论文 A.7](https://arxiv.org/html/2502.09597#A1.SS7)，
  [源码保留历史并追加检索内容](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/utils/explicit_utils.py#L388)，
  [输出配置](https://github.com/amazon-science/PrefEval/blob/50795054b5ff5f418d2b768a331d71e480f93331/config.yaml)。
- A-Mem：[NeurIPS 2025 会议页](https://neurips.cc/virtual/2025/poster/119020)，
  [论文](https://arxiv.org/html/2502.12110)，Appendix A.5 / Table 8。
- MemoryOS：[EMNLP 2025 正式论文](https://aclanthology.org/2025.emnlp-main.1318/)，§4.1。

本地完整证据与原始下载位于 `dagv2_analysis/a48_v3_validation/research/`。
本次 metadata 统计解析过含 label 列的官方 CSV，仅取 metadata 字段；没有读取项目
`evaluation_only.json`，没有使用正确答案选择配置，不能将其表述为“从未读取含标签文件”。
