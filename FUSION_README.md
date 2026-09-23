# DAG v2 + 最新 Evidence BridgeTree

本目录由 `dagv2_package_20260921` 完整复制而来。原始 57 个文件保留不改；原版 Git 分支是 `original-dagv2`，融合支线是 `fusion/latest-bt`。三套数据、索引、原脚本、原方法代码都可继续使用。`original_manifest.json` 提供字节校验。

三个大型 `passage_vectors.npy` 原样保存在本地，并纳入校验，但不提交到 Git；复制或部署此目录时需一并携带 `data/`。

**这是冻结模型的检索、推理与评测实验，不更新模型参数。** 新方案的准确率是否提高，必须由真实模型实验判断。

## 融合做什么

```text
原问题 → 原 DAG 规划器（只看问题）→ 固定终端需求
                  ↓
       就绪子问题：用已支持父结论填入实体/条件
                  ↓
       最新 BT 多根搜索 + 原文条件提案
                  ↓
       所有发现候选 → 精确引文映射 → 节点支持解析
                  ↓
       AND 前提 / OR 替代支持 → 缺口回查 / 证据触发细化
                  ↓
       条件与反证审计 → 局部支持失效 → 受影响结论重评
                  ↓
       token 预算内完整来源闭包 → 原文 reader → 答案
```

BT 从当前工作区冻结为 `vendor/bridgetree/`，不是旧 Git HEAD。原目录 `/Users/mao/projects/bridgetree_preference_rag` 不作为运行依赖，也没有被改动。默认融合路径使用实际 `EvidenceBridgeSearcher` 和 `SetReranker`，包含多根调度、pair tests、pivot、受限 speculation；没有用简单 BFS 或 cosine 分数替代 BT。来源见 `vendor/bridgetree_manifest.json`。

运行需要原 DAG 的 Qwen 服务和 **NV-Embed-v2** embedding 服务，以及 BT 的 **pointwise reranker**。现成语料向量不能直接改用其他 embedding 模型。当前机器未运行这些默认端口；示例配置的 reranker 模型名需要替换为实际部署。

## 启动

Python 3.10+；建议 3.11。原模型服务部署见原 `README.md`，融合依赖如下：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-fusion.txt
cp configs/paired.example.json configs/paired.local.json
# 编辑 paired.local.json 的三个服务端点与模型名，凭证只放环境变量。

# 配好服务后一条命令后台启动全部 3×1000 题，每题旧/新各一次：
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/paired_full
```

先跑 `--limit 2` 到单独目录可以检查真实模型协议。后台任务不依赖终端，支持 `status`、`stop`、原命令恢复和显式 `--retry-failed`；某题超时或失败不会停止另一版本和后续题。成功结果不会重复生成，失败尝试始终保留。数据范围或代码变动须换输出目录。

完整命令、日志结构、故障处理见 [成对实验运行说明](docs/PAIRED_RUNNER.md)。

## 如何判断各模块有用

默认 `original + fusion` 是系统级对照。原版会向 reader 注入答案链，新版默认只传原文，因此不能把总分差直接解释成 BT 的效果。

同时使用以下四臂，可分离发现与选择：

```bash
bash scripts/run_paired.sh --config configs/paired.local.json --output outputs/factorial \
  --arms original fusion dense_dependency bt_flat dense_flat
```

四个新臂共享融合求解器、真实 tokenizer 和 raw-only reader。另有条件审计、OR 支持、失效传播、细化、chain reader，以及强制保留导航来源、取消评分代理调度的消融。`fixed_candidate_pools` 可为指定问题提供冻结候选 ID 列表，用于同候选池选择对照；它不能来自 gold，也不能作为在线召回成绩。

日志包括每次查询与命中、完整 BT 搜索事件、实际支持父项与原文引用、节点状态/版本、冲突与明确解决、闭包选取、reader 确切输入、所有调用/缓存/重试成本。评分阶段再加载标签，报告全任务、共同成功子集、救回/损害题、发现召回和选集召回。语义依赖正确性仍需独立人工检查，程序证书不证明自然语言蕴涵。

研究设计逐项对应与适配依据见 [实现核对](docs/FIDELITY_AUDIT.md)。离线验证记录见 `docs/VALIDATION.md`。这里没有将模拟接口输出当作真实模型实验收益。
