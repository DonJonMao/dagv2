# a895695 审查修复与验证

审查基线为 `a895695b20a96ec183c54b62eedeccc61e496a65`，继续使用 `feat/residual-memory-f4e38df` 和独立 worktree。保留首次检索前的一次完整 DAG 规划、逐节点依赖求解、局部 BT、零评分、联合阅读及终端直接输出。F 与 F_t 继续描述 DAG 的求解关系和剩余状态。状态仍为 `IMPLEMENTED_AND_TESTED_OFFLINE`，真实模型未验证。

## 基线复现

从 Git blob 直接加载基线 `residual.py` 和 `joint_reading.py`，未修改这两份历史源码，外围服务与窗口容量使用明确测试替身。源码 SHA256 分别为 `61b3da179486cd4df8277acd460c1a5c12dae3d9b2df69693194eebe171f5e90`、`9a2f7e7a2ddff1bdfc17ce9ce06442b0cece6d5a17c704c2eab74523778055e1`。

复现两项缺陷：同样的两份原文单批读取得到 x=7、pending 为空并可认证；拆成两批后虽然都成功，仍留下 pending=x，证书为空。合法域 [0,60] 秒、事实 180 秒则在基线产生 output=false、nonempty_proven=true 的符号证书。后者属于假设域冲突，不能解释为正常门槛失败。

## 四项修复

| 问题 | 修复行为 | 新增行为测试 |
| --- | --- | --- |
| 窗口未完成标记不清除 | 未读窗口与校验失败按身份分别登记；窗口成功只关闭未读义务，失败和其他查询义务保留；快照保存分类 | 单批/多批答案与来源相同；后续成功不抹去失败；恢复后未读清理、失败保留 |
| 数值域冲突仍认证 | 单位统一后求交；空交集拒绝绑定/返回 inconsistent；求交在部分求值和枚举之前；整批值先验证后修改事实 | 点、区间、分钟/秒、开端点、异量纲、域内声明单位、重叠裁剪、批次原子拒绝；门槛失败仍可正常认证 |
| 空探针被当作整个前沿耗尽 | 不消费前沿地检查剩余根/延续；空探针有前沿就继续；预算与服务错误单独记录 | 真实 Engine/BT 调度中首根为空、第二根找到 x=7；全部为空正常结束；预算到零和服务失败不循环 |
| 规则未完整却没有活跃任务 | 将所需规则完整性加入预规划 rules 节点的局部缺口、查询和阅读上下文；完整原文通过撤销旧路线/支持新路线更新版本 | 已知条件为真、不完整规则仍继续检索并形成肯定结果/完整失败集合；必要条件为假、一个理由契约不额外检索规则 |

校验失败仍只由同批的有范围修复关闭；新的独立成功读取不能覆盖失败反证。历史快照没有义务分类信息时保守保留 pending。合法域固定，不实现静默估计域修订。测试中的服务和首根空返回是明确替身，核心 Engine、BT 调度、来源校验和求值器使用实际实现，不构成真实模型效果证明。

## 回归与合成结果

最终完整回归：**819 passed、0 failed、0 skipped、69 warnings，52.75s**。剩余专项：**89 passed，3.64s**，其中本轮新增审查定向测试 **18 项**。新增测试位于 `tests/test_residual_review_regressions.py`，与原剩余语义、运行测试和完整 tests 范围一起执行。保留原有随机有限世界 oracle、来源撤销、审计重算、断点前沿及旧方法回归；69 条已有矩阵运算警告保留。最终完整日志位于本机 `/tmp/dagbt-residual-review-full-final.txt`，专项日志为 `/tmp/dagbt-residual-review-targeted-final.txt`，均不提交 Git。

完整合成轨迹保存于私有 `outputs/residual_memory_v1/review_fixes_a895695/synthetic.json`，六个消融对照保存于同目录 `ablation.json`。八个案例结果与历史一致，包括无法确定的布尔案例保持未完成；V7 修正案例的 ANN 从 **15 增至 21**，逻辑生成仍 **15**。这是空探针后继续实际可执行前沿的结果，不扩大原 ANN36 / LLM24 预算。其他合成调用量未变：一个原因 ANN4/生成7，全部失败 ANN8/生成13，实体链 3/7，比较与记录 2/4，未确定布尔 1/2，开放建议 1/4。

| 消融 | ANN | 生成 | set_score | rerank HTTP |
| --- | ---: | ---: | ---: | ---: |
| f4 scored | 22 | 11 | 28 | 12 |
| 匹配 unscored | 27 | 11 | 0 | 0 |
| joint | 8 | 13 | 0 | 0 |
| residual | 4 | 7 | 0 | 0 |
| 固定完整池 joint | 0 | 9 | 0 | 0 |
| 固定完整池 residual | 0 | 5 | 0 | 0 |

六个消融均完成同一虚构题，不据此宣称真实准确率或最优成本。

## 真实服务与保护核验

本轮执行固定相同 HotpotQA 前三题、PersonaMem 前三题 smoke，使用新私有目录 `outputs/residual_memory_v1/smoke_review_a895695_v2/`，manifest 保留当次源码/配置身份及固定题号。正式运行仍在 LLM `/models` 预检以 `RemoteDisconnected: Remote end closed connection without response` 失败；独立 embedding `/models` 轻量检查返回 `HTTP Error 502: Bad Gateway`。progress 与 embedding_endpoint_check.json 均保留错误。

**六题均未执行**，没有预测、准确率或题内平均成本；不能描述为六题回答成功或六题回答失败。预检 GET 与题内预算分开，题内 LLM/ANN/rerank 调用均为 0。没有读取金标，没有索引重建、预算扩大或反复运行六题。最后补充域内单位边界测试后没有因同一服务阻断重复 smoke；已有 manifest 是当次尝试记录，后续服务可用时必须使用匹配最终源码的新目录重新执行。

原始与 vendor manifest 各 57 项哈希通过；108 个受保护 tracked 文件相对 f4 基线变更为 0。原分支 HEAD、数据、凭据及历史产物保留。提交只包含本轮实现修复、测试与文档，不提交私有运行目录或原始轨迹。推送只针对既有研究分支，采用正常 push，提交号与远端 HEAD 在交付回复核验。
