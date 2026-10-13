# PersonaMem v4：安装与运行

这是基于服务器实际部署 `a5b993a` 修订的推理评测包。新方法名为
`dagbt_residual_memory_v2`；原方法为 `original`。两组继续使用既有模型、同一
PersonaMem 问题及 cutoff-safe 个人历史、相同 16384 context 设置。没有参数训练。

## 服务器安装

上传新版 tar 和 `.sha256` 到 `~/bt/`。以下命令会建立独立的新目录；已有 v3.1
输出应保留，不能在旧结果目录接着写新版。

```bash
cd ~/bt
sha256sum -c dagv2_personamem_v4.tar.gz.sha256
mkdir -p ~/dagbt_runs/20261013_personamem_v4
tar -xzf dagv2_personamem_v4.tar.gz -C ~/dagbt_runs/20261013_personamem_v4
cd ~/dagbt_runs/20261013_personamem_v4/dagv2
bash scripts/personamem_v4.sh install
bash scripts/personamem_v4.sh smoke
```

脚本默认复用 `$HOME/dagbt_runs/20260930_v3/dagv2/.venv/bin/python`，不用每次重新
export。若环境不同，设置 `DAGBT_PYTHON` 即可。安装器默认旧部署为
`~/dagbt_runs/20261011_personamem_a48/dagv2`；不同路径用 `install --base /完整路径`。
新包不包含密钥。安装器校验后复用旧部署的本地凭据与兼容索引，不会输出 key，
也不会修改旧部署。若旧部署缺必要文件或哈希不符，安装会停止并明确报告。

只有安装器提示“无兼容的已完成索引”时，才使用下面的显式备用流程。第一步只安装
已校验的冻结向量和本地凭据，输出 `ready: false`；第二步会请求 embedding 服务，
在新目录建立索引，完成后重新安装校验。已有索引不会被覆盖或自动重建。

```bash
bash scripts/personamem_v4.sh install --assets-only
bash scripts/personamem_v4.sh prepare-index
bash scripts/personamem_v4.sh install
```

## 测试、全量与结果

smoke 固定使用公开顺序中每个题型前两题，共 14 题、两组各一次。启动成功不
等于完成。检查时必须明确指向 smoke：

```bash
bash scripts/personamem_v4.sh status --output outputs/personamem_v4_smoke
bash scripts/personamem_v4.sh check
```

`check` 必须输出 `status: passed`，才运行：

```bash
bash scripts/personamem_v4.sh full
bash scripts/personamem_v4.sh status --output outputs/personamem_v4_full
bash scripts/personamem_v4.sh accuracy
```

`full` 会自行重查 smoke 的代码、配置、数据、题目、结果与审计状态。全量 589 题，
使用 `outputs/personamem_v4_full`；smoke 是独立实验，不混入全量分数。不修改输出目录
中的 manifest，不覆盖旧行。中途查看 `accuracy` 为只读评价：同时报告含失败同题比较
和双方有效同题比较；缺失与损坏行不进临时分母。生成过程不读取标签。

停止指定实验：

```bash
bash scripts/personamem_v4.sh stop --output outputs/personamem_v4_full
```

失败时查看 `launcher.log`、对应方法的 `rows/` 和 `failures/`。不要只扩大预算或将失败
移出分母。全量结果为 `outputs/personamem_v4_full/personamem/summary.json`。

## 方法和修复边界

- 上层 DAG 是预先规划的子问题/证据依赖；下层是各子问题的 dense 与桥接发现关系。
  下层发现边不自动成为证据支持边。残余需求仍决定后续局部检索。
- PersonaMem 终端必须是针对原始用户话语的语义回应选择；窄事实、偏好、时间变化
  是中间证据。终端输出为所选选项原文和字母，终端审计同时检查个性化适合性。
  其 `completion_kind=semantic_assessment`，不能宣称符号证书证明回应正确。
- 历史提取/汇总、最终回应选择、回应审计使用独立的阶段契约。历史阶段不提供选项或
  上次选择的答案，只允许返回历史变量；终端字段即使为 null 也会触发有界修复。
  每个历史变量仍须显式声明实体和时间范围；当前约束可以与当前回应共享该范围，
  任务角色由变量、依赖及阶段区分，不强迫模型改写 scope 字符串。
- 修复不相容契约、类型/展示值误比较、重复无进展读取、终端字段不一致以及审计后
  无法有限纠正的问题。保留反证、来源、scope、预算及失败状态。
- 完整原文与来源验证仍执行真实上下文计数。模型不能通过伪造来源、删去反证或
  将未来建议伪装成已发生事实获得成功状态。
- 未解决证据保留原文直到明确复核。当前不能跨批累计复核后清除一项证据义务，
  因此无法在单次上下文容纳的义务仍可能保守失败；没有通过静默丢弃来源绕过限制。
- 历史阶段若产生多个不一致的完整值，会在原预算内提供原草稿和来源供联合复核。
  模型须明确给出有源的完整值或保留真实冲突；代码不自动合并集合或采用最新值。
- 每个非确定性、无有限域的历史文本变量都必须在规划时明确选择“历史摘要”或
  “单一事实”，不能仅给后置汇总节点声明类型。单一事实保持整值一致性，不获得
  摘要修订权限；代码不会依据变量名称自动补声明。兼容的新经历可以补充历史摘要的
  旧摘要；模型需明确指定旧路径及当前版本，保留原有来源、负面经历和时间条件。
  验证及下游容量检查通过后，旧摘要作为被替代的表示保留。真实反证仍独立记录，
  并使依赖其证据的后续摘要失效；该能力不适用于普通事实或最终选项。
  完整复核后内容与证据均未变化时，保留已有事实路径和祖先关系，不生成新版本。
- 失败批次保留草稿、原文和已验证的反证关系。仅同一历史任务、同一阶段在出现新
  原文或实际父绑定变化时，才能消耗既有 LLM/JSON 修复预算进行恢复。完整复核及
  下游容量检查通过后，只解除本次获准复核的失败记录。更换查询名称不产生免费
  重试；历史复核不能代替终端选择或审计。原始负面线索不能因后来有支持证据而丢弃。
- 此版本没有把 reranker、独立 Reader 和 residual 同时混改；旧版成功题增益不能
  归因到本版。需要这些机制的因果比较时，应另外进行同数据和同资源的对照实验。

代码检查与具体测试记录见包内 `VALIDATION_PERSONAMEM_V4.json`。测试只能验证覆盖到的
行为；模型仍可能误读偏好或时间关系，14 题执行通过也不保证全量零失败或超过基线。
