# DAG v2 异机执行发行包（dagv2_package_20260921）

## 一句话定位

DAG v2 是一个**冻结的 RAG 评测方法**：让 LLM 规划器把多跳问题拆成依赖 DAG，逐节点
"接地提问 → 全语料检索 → 单调用作答并强制勾选来源"，再把节点答案链注入 Reader 出
最终答案——**全程无 thinking**。本包含全部代码、三数据集（HotpotQA / 2WikiMultihopQA /
MuSiQue，各 1000 题）问题、语料、预建 NV-Embed-v2 向量与评分标签，可在其他服务器
独立复现；**不含模型权重**。

## 方法流程

```
原问题
  │  ① 规划器（chat/completions, structured JSON, 无 thinking）
  ▼
依赖 DAG（1–6 节点：question/output_slot/answer_type/inputs；父占位符 {slot}）
  │  ② 档案池：原问题 + 各 DAG 步提问（占位符替换为"the unknown entity"）
  │     各自全语料稠密检索 top50，按查询顺序稳定并集
  ▼
候选池 candidate_doc_ids（≦ 50×(n+1) 篇）
  │  ③ 按依赖序逐节点（frozen_flow_v6.controller）：
  │     父答案代入占位符 → 接地查询 → 全语料检索 top50 并入池
  │     → top20 面板（优先父引用闭包）→ 单调用 guided-JSON 作答
  │     （answer + sources 布尔数组，512 tokens，stop=<|im_end|>，无 thinking）
  ▼
节点答案链（每节点：接地问题/答案/勾选来源/解析状态）
  │  ④ 预算选择：k=5/10/20 枚举节点证明闭包组合，在篇数预算内最大化覆盖
  ▼
选中来源文档
  │  ⑤ Reader（chat/completions，1024 tokens，无 thinking）：
  │     top20 原文 + 全部已解析节点链作为 lineage_evidence 注入，
  │     指令"以链尾为出发点、与原文核对、冲突时以原文为准"
  ▼
最终答案 + 逐题 scores.json / summary.json / RESULTS.md（full1000 完成时自动生成）
```

## 目录导览

```
dagv2_package_20260921/
├── README.md                ← 本文档
├── requirements.txt         ← pipeline 本体依赖（仅 3 个包；HTTP 用标准库 urllib）
├── config.example.json      ← 端点/模型名覆盖配置（经 DAGV2_CONFIG 环境变量生效）
├── package/                 ← 冻结 DAG v1 核心（原样复制，未改一行）
│   ├── run.py core.py frozen_flow.py reader.py metrics.py
│   └── tokenizer/           ← Qwen3.8 tokenizer（5 文件，含 chat template）
├── dagv2/                   ← DAG v2 扩展层（本方法的全部增量逻辑）
│   ├── experiment.py        ← 规划器/档案池/validate_plan/CONFIG（路径已改为包内相对）
│   ├── repair_planner.py    ← 规划器尾注修复（{slot: X} 剥离）
│   ├── frozen_flow_v6.py    ← 无 thinking 控制器：节点单调用 + Reader 链注入
│   ├── experiment_v6.py     ← patch 加载器：用 v6 变体替换冻结核心 4 个入口
│   ├── pipeline_dagv2.py    ← HotpotQA/2Wiki 流水线（入口）
│   └── pipeline_dagv2_musique.py ← MuSiQue 流水线（保留 musique: id 归一化）
├── data/
│   ├── inputs_manifest.json ← 源服务器 provenance 哈希（prepare 按文件名后缀匹配校验）
│   └── {hotpotqa,2wikimultihopqa,musique}/
│       ├── questions.jsonl corpus.jsonl        ← 原始数据集（逐字节复制）
│       ├── questions.json corpus.json          ← EvLink++ 规范化副本（prepare 一致性断言用）
│       ├── evaluation_only.json                ← 评分标签（仅评分阶段读取）
│       └── index/{passage_vectors.npy,manifest.json} ← 预建向量（float32 4096 维）
├── serving/                 ← 模型服务脚本（不含权重）
│   ├── start_qwen38.sh      ← LLM 服务（生产实测命令，见下）
│   ├── start_nv.sh + serve_nv_embed.py + nv_model_code/ ← NV-Embed-v2 服务（端口 8019）
│   └── requirements-qwen.txt / requirements-nv.txt      ← 两个服务的独立环境
├── scripts/
│   ├── smoke.sh             ← 本地冒烟：模块导入 + 三数据集 prepare 干跑（无模型调用）
│   ├── run_hotpotqa.sh run_2wiki.sh run_musique.sh      ← 正式跑（支持 NOHUP=1 后台）
└── results/                 ← 源服务器全量成绩（RESULTS_*.md + METHOD_CARD.md 方法卡）
```

运行产物写入 `outputs/<dataset>/{smoke2,full1000}/`（manifest.json、progress.json、
rows/ 逐题结果、requests/ 逐请求缓存、scores.json、summary.json、RESULTS.md）。

## 快速开始

### 0. 硬件与前提

- Linux，Python 3.10+。pipeline 本体不需要 GPU，只需能访问两个 HTTP 服务。
- **LLM 服务**：Qwen3.8-27B-AWQ-INT4，vLLM 部署，**需要 2×24G 显存（TP2）**。
- **Embedding 服务**：NV-Embed-v2，**需要 1×24G 显存**。共 3 张 24G 卡。
- 模型权重自备，本包不含。客户端机器预留 ≥8GB 内存（向量矩阵约 450MB×3 数据集，
  逐数据集加载）与输出磁盘（全量含请求缓存约数 GB/数据集）。

### 1. 建 pipeline 环境

```bash
cd dagv2_package_20260921
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 起两个模型服务（两个独立 Python 环境）

LLM 服务（生产实测命令已固化在 `serving/start_qwen38.sh`）：

```bash
python3 -m venv .venv-qwen && source .venv-qwen/bin/activate
pip install -r serving/requirements-qwen.txt   # vllm
export QWEN38_MODEL_PATH=/path/to/Qwen3.8-27B-AWQ-INT4
CUDA_VISIBLE_DEVICES=0,1 bash serving/start_qwen38.sh
```

等价的显式命令：

```bash
vllm serve $QWEN38_MODEL_PATH --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8020 --tensor-parallel-size 2 \
  --disable-custom-all-reduce --enable-prefix-caching --mamba-cache-mode align \
  --gpu-memory-utilization 0.80 --max-model-len 32768 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5}'
```

注意：pipeline 的节点调用走 **`/v1/completions`** 并自带聊天模板与 `stop=<|im_end|>`，
服务级 `enable_thinking=false` 只影响 `/v1/chat/completions`，二者不冲突；
DAG v2 全程不使用 thinking。LLM 服务必须支持 vLLM 的
`structured_outputs: {json: ...}`（节点 guided-JSON 作答依赖它）。

Embedding 服务（端口 8019，OpenAI 兼容 `/v1/embeddings`）：

```bash
python3 -m venv .venv-nv && source .venv-nv/bin/activate
pip install -r serving/requirements-nv.txt
export NV_EMBED_MODEL_PATH=/path/to/NV-Embed-v2
CUDA_VISIBLE_DEVICES=2 bash serving/start_nv.sh
```

调用方在查询文本前拼 `'Instruct: Given a question, retrieve relevant documents that best answer the question.\nQuery: '`，
`serve_nv_embed.py` 沿用源服务器查询处理（换行转空格、4096 长度、归一化）。
也可用任何 OpenAI 兼容的 NV-Embed-v2 服务替代。跨机器部署时把服务绑到相应
`QWEN38_HOST` / `NV_EMBED_HOST` 并改配置端点。服务若需鉴权：
`export DAG_LLM_API_KEY=...` / `export DAG_EMBED_API_KEY=...`。

### 3. 冒烟验证（不需要模型）

```bash
source .venv/bin/activate
bash scripts/smoke.sh
```

校验全部模块可导入、三数据集数据路径、provenance 哈希、tokenizer 与代码资产哈希。
每个数据集独立进程执行（prepare 有模块级 CONFIG 改写副作用）。

### 4. 正式跑三数据集

```bash
bash scripts/run_hotpotqa.sh     # 或 NOHUP=1 bash scripts/run_hotpotqa.sh 后台
bash scripts/run_2wiki.sh
bash scripts/run_musique.sh
```

每个脚本先跑 2 题冒烟（smoke2），任一题非 ok 即报错停止；通过后自动续跑 full1000。
支持断点续跑：已完成的题与已缓存的请求响应精确复用，中断后重跑同一命令即可。
单数据集全量在源服务器（2×4090 TP2 + 1×4090）约需 1–2 天；三数据集可顺序排队。
**评分在 full1000 完成时自动生成**（无需手动触发）：`scores.json`（逐题）、
`summary.json`（汇总）、`RESULTS.md`（指标表）。

### 5. 端点配置

默认端点/模型名与 `dagv2/experiment.py` 内 CONFIG 一致（8020 / 8019 /
`qwen3.8-27b` / `nvidia/NV-Embed-v2`）。如需覆盖，**不要改代码**：

```bash
cp config.example.json config.json   # 修改端点/模型名
export DAGV2_CONFIG=$PWD/config.json
```

## 方法卡成绩（源服务器全量，详见 results/METHOD_CARD.md）

| 数据集 | F1 | EM | R@5 | R@20 |
|---|---:|---:|---:|---:|
| HotpotQA | 79.83 | 67.00 | 98.00 | 99.25 |
| 2WikiMultihopQA | 80.34 | 72.90 | 97.03 | 97.90 |
| MuSiQue | 64.66 | 54.50 | 81.91 | 89.76 |

与 DAG v1（F1 82.91/75.94/66.86）、终末直出消融、answer_type 判负消融的完整对照见
`results/METHOD_CARD.md`；三数据集逐指标全表见 `results/RESULTS_*.md`。

## 诚实披露（引用本成绩前必读）

1. **Reader 87–99% 照抄链尾**：终末节点已解析时，87–99% 的最终答案是链尾答案的
   逐字照抄（2Wiki 99.4%）。DAG v2 的最终答案主要由 DAG 链的终末节点决定。
2. **终末直出消融**（跳过 Reader，直接用链尾答案，离线推导自同一 run 数据）：
   HotpotQA 71.82 F1（Reader 贡献 **+8.0**）、2Wiki 78.70（**+1.6**）、
   MuSiQue 57.97（**+6.7**）。Reader 的真实贡献是答案格式归一（截短为"最短短语"）
   与链条未解析时的兜底作答。引用 DAG v2 成绩时必须同时引用该消融。
3. **无标签泄露**：生成链路（work/planner/archive/nodes/reader）零标签接触；
   evaluation_only 标签只在 1000 行全部生成完毕的断言之后、在评分阶段读取；
   注入链 100% 模型自生成；索引由纯语料构建（哈希断言）；无训练、采样温度 0；
   中间评分只读、无反馈回流。
4. **与 DAG v1 的差异**：

| 维度 | DAG v1（冻结核心） | DAG v2（本包方法） |
|---|---|---|
| 节点检索范围 | 档案池内 top20 | 全语料 top50 并入池 + top20 面板 |
| 节点调用 | thinking 2048 + 结构输出 512（两段） | 单调用 guided-JSON 512 tokens |
| Reader 输入 | 仅 top20 原文 | top20 原文 + 节点答案链注入 |
| Reader 输出 | thinking 2048 + 终态 128 | 非思考 1024 |
| 规划器 | v1 不含（输入已有 DAG） | 问题驱动 DAG 规划器 + 尾注修复 |

## 配置项说明

| 配置 | 默认值 | 说明 |
|---|---|---|
| `llm_base_url` | `http://127.0.0.1:8020/v1` | LLM 服务（chat + completions） |
| `embedding_base_url` | `http://127.0.0.1:8019/v1` | embedding 服务（/v1/embeddings） |
| `llm_model` | `qwen3.8-27b` | 须与服务端 `--served-model-name` 一致 |
| `embedding_model` | `nvidia/NV-Embed-v2` | 须与 index/manifest.json 一致（prepare 断言） |
| `request_timeout_seconds` | 600 | 单请求超时 |
| `max_identical_attempts` | 3 | 每请求最大持久化重试；耗尽后保持 pending，不记零分 |
| 并发 workers | 2（写死） | execute() 内 ThreadPoolExecutor(max_workers=2) |
| 节点输出 tokens | 512 | guided-JSON 作答上限 |
| Reader 输出 tokens | 1024 | 最终答案上限 |
| Reader 原文篇数 | 20 | 预算 k=20 的选中来源 |
| 上下文上限 | 16384 tokens | 超限即该题 context_overflow，不自动截断 |

## 常见故障排查

- **vLLM 服务起不来**：确认 3 张 24G 卡空闲（`nvidia-smi`）；TP2 需要两张同型号卡；
  `--disable-custom-all-reduce` 在多卡 4090 上必需；权重目录须完整
  （config/tokenizer/模型代码）。起好后用
  `curl http://127.0.0.1:8020/v1/models` 验证。
- **embedding 不通**：`curl http://127.0.0.1:8019/v1/embeddings -d '{"model":"nvidia/NV-Embed-v2","input":["test"]}'`；
  NV 服务首次加载模型需数分钟，看启动日志 READY 行。transformers 版本与
  `serving/requirements-nv.txt` 保持一致（模型代码 monkey-patch 依赖版本）。
- **prepare 报 hash mismatch / provenance 找不到**：说明 data/ 下文件被改动或损坏。
  prepare 会用 `inputs_manifest.json` 逐文件校验 questions.jsonl/corpus.jsonl 的
  sha256（按文件名后缀匹配，哈希值逐字节比对），失败即报错，请勿绕过；
  从 tar.gz 重新解包恢复。index/manifest.json 的 `embedding_model`/`documents`
  也与 CONFIG 和语料条目数互相断言。
- **smoke 失败（answer 非 ok）**：看 `outputs/<dataset>/smoke2/rows/*.json` 的
  error 字段与 `requests/` 缓存。常见原因：服务端模型名不一致（payload 里的
  model 被 CONFIG['llm_model'] 覆盖，须与 `--served-model-name` 一致）；
  服务不支持 `structured_outputs`；显存不足导致服务抖动。
- **续跑与锁**：一个输出目录只允许一个执行进程（pipeline.lock / writer.lock）。
  manifest 变化（配置/数据/范围）会拒绝复用旧目录，请换新输出目录。
- **Python 不要加 `-O`**：冻结实现含上下文容量等断言，`-O` 会跳过。

## 与其他方法的对比数据指路

- `results/METHOD_CARD.md`：方法卡全文，DAG v1 对照（F1 82.91/75.94/66.86），
  池化修复 / answer_type 判负 / 链注入收益三条消融结论，诚实披露。
- `results/RESULTS_{hotpotqa,2wikimultihopqa,musique}.md`：三数据集全量逐指标。
- 上游 EvLink++（图检索方法）与本方法使用同一数据/索引/评分口径，可比性说明见
  方法卡；EvLink++ 成绩请查其自身发布物，本包不代为发布。
