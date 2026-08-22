# Safe-query acquisition：实现、失败诊断与固定假设库对照

日期：2026-08-17

> **历史阶段说明**：本报告主体记录的是 hard-margin falsifier 完成前的 v3 selector-only 阶段，用于说明 safe acquisition 为什么需要 cap、逐条反馈和 causal gate。当前 falsifier、当前 selector 对照和最终数字以 `FALSIFIER_HARD_MARGIN_LADDER_2026-08-17.md` 为准；两份报告的 IoU 方向不同，是因为候选生成器已经改变，并非同一次实验自相矛盾。

## 阶段性结论（pre-hard-margin v3）

safe-query acquisition 已从“每轮预留一个看起来可能安全的样本”改成了**逐条、自适应且受信息价值约束的查询策略**。当时的 v3 selector-only 对照在固定同一组 GPT 初始假设、固定随机种子、固定候选生成器和相同 26 次 Oracle 总预算下：

- safe-query：冠军为正确的空间障碍约束 `h_spatial_exclusion`，IoU `0.8219`；
- global-only：冠军同样为空间障碍约束，IoU `0.8309`；
- safe-query 主动获得 1 条额外 safe 标签，并把“安全区域被误判为违约”的 false-unsafe rate 从 `0.795%` 降到 `0.464%`；
- 代价是“真实违约被误判为安全”的 false-safe rate 从 `8.87%` 上升到 `13.17%`，IoU 下降约 `0.9` 个百分点；
- 两者 accuracy 几乎相同（`98.57%` 与 `98.59%`），且都不再选择错误的 terminal proxy。

因此，这次改动不是“全面击败 global acquisition”，而是把原先会失控的 label balancing 修正为一个可控的**安全覆盖—违约精度折中**。更重要的是，它修复了错误冠军和 Oracle 预算浪费问题。

## 原策略为什么失败

原实现有四个互相叠加的问题。

1. **每轮只预留一条 safe-seeking query。** 当训练标签非常不平衡时，实际可能需要多条 safe 标签，但策略只拿 `likely_safe[0]`。
2. **先选完整批次，再统一查询 Oracle。** 第一条 safe probe 若被 Oracle 判为 violation，本轮不会继续尝试同池中的其他候选。
3. **标签计数包含 holdout。** `warmup_validation` 与 `final_calibration` 不参与训练，却被算入 label balance，使系统高估训练集里的 safe 数量。
4. **“false-unsafe”名称被当成可靠语义。** 生成、投影后，很多候选连 source model 自己都不再判为 violation；还有候选被所有模型判 safe，却仍占用 safe quota。这类查询只能补数量，不能纠正任何假设。

旧 GPT 诊断中，12 条 outer-loop 查询只有 `1 safe / 11 violation`。第一版过度修正后，虽然变成 `7 safe / 5 violation`，却让 8/12 的预算被 label balance 强制占用，并最终选择了只看终点的 `h_terminal_goal_band`，私有评估 IoU 为 `0`。这说明“类别平衡”不能代替“结构信息”。

## v3 阶段实现与当前保留机制

### 1. 逐条自适应查询

每一轮先冻结候选及所有模型的 pre-query prediction，然后循环执行：

1. 仅按真正进入训练的数据重算 safe/violation 缺口；
2. 选择一条候选；
3. 立即查询 Oracle；
4. 更新标签计数与在线命中率；
5. 再决定下一条。

这样第一条 safe probe 失败后，剩余预算仍能重试，同时不会用查询后的模型重训结果污染同轮候选排序。

### 2. safe quota 有上限且必须有纠错对象

- 每轮最多 2 条 label-balance query；
- 当前 safe 候选必须同时满足：model-safe anchor、generation full/target hard margin、refined query margin，以及 source 本身是 causal rejector；
- 所有模型都预测 safe 的 coverage candidate 不得占用 safe quota；
- safe utility 同时考虑估计安全概率、expert proximity、真实 rejector 和一般信息价值，而不是只按全局 acquisition 排名。

### 3. Oracle 反馈后的去冗余

若某条候选被 Oracle 判为 safe，它已经反驳了一组 causal rejectors。同一轮中，具有相同 causal signature 的其他候选不再重复查询；非因果 unsafe voter 不进入 signature，同一 composite 的不同 source clause 也不会被错误合并。若首条被判为 violation，则仍允许继续尝试。v3 固定库运行因此只使用了 1 条 adaptive safe query，其余 11 条回到 global acquisition。

### 4. expert 到 candidate 的模型边界细化

当前系统先扫描 expert→endpoint 上第一段 safe→unsafe bracket，再局部二分，寻找仍保留 query margin 的最小扰动，并分别记录 generation 与 query certificate。composite source 必须沿 intervention 指定 clause 扫描。若不存在 expert-safe → candidate-unsafe crossing，会明确记录 `no_safe_to_unsafe_crossing`，不会伪装成成功细化。

本报告原始 v3 固定库运行中，所有生成的 false-unsafe 候选都没有形成这种 crossing。这一诊断直接推动了后续 hard-margin/radius 改造；当前 combined 运行已得到 4/13 条 certified crossing probes。

### 5. terminal representation collision 硬证伪

本次实验暴露了一个比 acquisition 更基础的漏洞：所有扰动都保持起终点不变，但安全 expert 与穿越障碍的 violation trajectory 可以拥有完全相同的终点。因此，纯 `last(x,y)` 假设不可能区分它们。

现在 evidence 层会对纯 terminal 假设按其 terminal representation 分组。若同一 representation 同时出现 safe 与 violation，直接加入：

```text
terminal_invariance_contradicted_by_oracle
```

并禁止它成为 champion。最终运行中 terminal 假设有 9 个 representation group，其中 5 个存在标签冲突，因此被结构性淘汰；空间约束不受此 gate 影响。

### 6. 固定 GPT 假设库的严格对照

新增 frozen-bank replay：从既有 `hypothesis_bank.json` 按 round-0 audit 顺序恢复假设，验证结构、冻结 revision，并记录源文件 SHA-256。该模式不会调用 GPT，`llm_interactions=0`。模型初始化种子也从 LLM 自定义 ID 改为结构指纹，避免“同一结构、不同名字、不同初始化”的混杂因素。

## 历史 v3 selector-only 对照（pre-hard-margin）

两臂共享：

- GPT 假设库：`outputs/gpt_safe_acquisition_final/hypothesis_bank.json`；
- seed：7；
- 3 个 outer rounds，每轮 4 次查询；
- 第一次更新前具有相同的候选生成与 boundary-bisection 配置；不同标签进入训练后，后续候选池会分叉；
- 冻结 revisions，0 次 LLM 调用；
- 总 Oracle 预算均为 26（14 warmup/audit/calibration + 12 outer）。

| 指标 | v3 safe-query | v3 global-only |
|---|---:|---:|
| outer safe / violation | 1 / 11 | 0 / 12 |
| adaptive / global slots | 1 / 11 | 0 / 12 |
| champion | spatial exclusion | spatial exclusion |
| IoU | 0.8219 | 0.8309 |
| false-safe rate | 13.17% | 8.87% |
| false-unsafe rate | 0.464% | 0.795% |
| accuracy | 98.571% | 98.592% |
| held-out expert safe rate | 100% | 100% |

解读：safe-query 用约 0.9 个 IoU 百分点和 4.3 个 false-safe 百分点，换取一条主动 safe 证据，以及约 42% 的相对 false-unsafe 降幅。也就是说，它减少了对真实可行区域的过度拒绝，但漏掉了更多真实违约；若系统更重视安全、希望减少“违约却判安全”，本次单次结果更支持 global-only。若更重视不过度压缩可行域，则可以保留受限的 safe-query，并继续调低 quota 或提高 causal crossing 门槛。

## 开发过程中的消融

这些行反映策略逐步收敛的过程，不应当全部当作单因素因果对照：

| 版本 | outer safe / violation | champion | IoU | 主要问题/改动 |
|---|---:|---|---:|---|
| 过强 quota 原型 | 7 / 5 | terminal goal | 0.0019 | 8/12 预算用于补 safe，选中 endpoint proxy |
| cap + causal eligibility | 2 / 10 | spatial exclusion | 0.7737 | 已修正冠军，但同一 rejector 集合仍重复查询 |
| 最终 signature dedup | 1 / 11 | spatial exclusion | 0.8219 | 去掉已被一条 safe 样本覆盖的冗余 probe |

## 验证

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe -m unittest discover -s llm_modulo_cegis\tests -v
```

当时结果：`42/42` tests passed。当前代码已扩展为 `57/57`，新增 falsifier 回归详见 hard-margin 报告。该阶段新增回归测试覆盖：

- safe probe 失败后的自适应重试；
- 已平衡时退回 global acquisition；
- trainable-only 标签计数；
- all-model-safe 候选不能占 safe quota；
- 每轮 balance cap；
- rejection-signature 去冗余；
- expert→candidate boundary refinement；
- terminal representation collision 不误伤 spatial max；
- 结构等价假设的初始化不依赖 hypothesis ID；
- frozen-bank replay 零 LLM 调用。

## 剩余瓶颈与下一步

1. **falsifier causal crossing 已实现。** hard-margin、clause-targeted certificate、model-safe anchor、radius ladder、margin-preserving refinement 与失败候选 query gate 的实现和对照见 `FALSIFIER_HARD_MARGIN_LADDER_2026-08-17.md`。当前代码的 combined 运行生成 4/13 条 certified crossing probes，其中仅 1 条被 Oracle 确认为真正的 model-false-unsafe；单一 0.32 半径也得到 4/13，说明 ladder 的稳定增益仍需多 seed 验证。
2. **当前 P(safe) 是弱先验。** 数据量很小，不应立即训练复杂校准器；可积累更多任务后，再按 intervention、归一化距离和 rejector 类型做在线 Beta/logistic calibration。
3. **finalization 未执行。** 当前诊断配置每类只留 2 条 calibration，而 finalization 最低要求 3 条；这与 acquisition 无关，但正式实验应统一配置。
4. **应跨 seed/跨障碍重复。** 当前固定库单次对照证明了机制和单次折中，不足以给出统计显著性结论。
