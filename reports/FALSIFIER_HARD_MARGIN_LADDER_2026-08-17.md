# Falsifier hard-margin crossing 与 radius ladder

日期：2026-08-17

## 结论

这次修改解决了 false-unsafe falsifier 的三个确定性问题：

1. 优化目标现在相对于模型自己的 calibrated `decision_threshold`，并使用实际推理时的 hard trajectory score；
2. 多约束 hypothesis 必须同时满足“整个 hypothesis 拒绝”和“指定 clause 拒绝”，不能由另一个 clause 冒充 crossing；
3. 搜索不再固定在 `0.08`，而是可以按 `0.04 → 0.08 → 0.16 → 0.32` 逐级扩大。

一个可查询的 false-unsafe probe 现在必须具有两份不同的证书：

```text
generation endpoint:
  known-safe anchor score <= threshold - 0.02
  full hard score         >= threshold + 0.05
  targeted-clause score   >= threshold + 0.05

refined Oracle query:
  full hard score         >= threshold + 0.02
  targeted-clause score   >= threshold + 0.02
```

固定 GPT hypothesis bank、seed=7 和 26 次 Oracle 总预算的结果表明：

- 在同为单一 `0.32` 半径时，hard-margin objective 将合格 crossing probes 从 `2/13` 提高到 `4/13`；
- hard-margin + ladder 也得到 `4/13`，其中 4 条都被查询，但只有 1 条经 Oracle 判为 safe，因此只有这一条是真正的 model-false-unsafe；
- 其余 3 条是模型正确拒绝的真实 violation，不应称为“生成了 4 个 false-unsafe”；
- 所有正式运行仍选中正确的空间结构 `h_spatial_exclusion`；
- combined 的 IoU 为 `0.8305`，没有超过全部单半径对照。

因此当前最可靠的结论是：**hard-margin target 明显改善了 falsifier reachability；扩大最大半径是必要的；radius ladder 能让部分后期 crossing 更局部，但本 seed 尚未证明它优于直接使用单一 `0.32` 半径。**

## 旧实现为什么失败

旧运行 `frozen_bank_safe_acquisition_v3` 中：

- 13 个 false-unsafe 候选全部没有 expert-safe→candidate-unsafe crossing；
- 12/13 的逐点最大扰动几乎恰好等于 `0.08`；
- 旧 smooth 目标到固定 `+0.05` 的中位缺口约为 `0.505`；
- 只有 2/13 的 endpoint 被 source model 拒绝；
- 这两个 source model 同时已经拒绝对应的已知安全 expert，因此也不是有效的因果 crossing。

根因不是简单的“优化步数不够”，而是三种口径不一致：

- falsifier 优化 smooth score；
- query 前判定使用 full-hypothesis hard score；
- loss 相对固定零点，而实际模型有校准后的非零 threshold；
- 单一 `0.08` trust region 又使很多边界根本不可达。

## 当前实现

### 1. calibrated hard-margin objective

对于原子约束，crossing loss 为：

```text
L_cross = relu(decision_threshold + 0.05 - full_hard_score)^2
```

对于 composite hypothesis，优化指定的 `intervention.clause_id`：

```text
L_cross = relu(decision_threshold + 0.05 - target_clause_hard_score)^2
```

但最终 generation certificate 同时要求：

```text
min(full_hard_score, target_clause_hard_score) >= decision_threshold + 0.05
```

所以“feature A 等式 + feature B 不等式”时，针对 A 的 intervention 不能因为 B 先违约而假装成功。未知 `clause_id` 会直接报错，不再静默退回整个 composite。

smooth score 仅作为低权重辅助梯度；ensemble uncertainty 对 false-unsafe 是惩罚项，不再被主动放大。

### 2. model-safe expert anchor

每个 source hypothesis 先从已知安全专家中寻找：

```text
full_hard_score <= decision_threshold - 0.02
```

并优先选最接近边界的合格 anchor。如果模型已经拒绝所有专家，状态记为 `no_model_safe_anchor`，该 probe 不允许消耗 Oracle。

### 3. kinematically valid hard-margin checkpoint

每一步优化后都会检查 hard certificate；初始轨迹在第一次 optimizer step 前也会检查。只有同时通过 finite、固定端点、workspace 和 max-step 校验的轨迹才保存，最终返回这些达标 checkpoint 中变形最小者，而不是机械返回最后一步。

这避免了几何正则在后续步骤把已经成功的 crossing 拉回 safe 侧。

这里的 valid 只表示运动学/几何校验通过，不表示 Oracle 已经确认任务可行；候选本来就可能是真实 violation。

### 4. radius ladder

诊断配置为：

```text
0.04 → 0.08 → 0.16 → 0.32
```

每一级使用既有 restarts；第一个存在合格 endpoint 的半径立即停止。多个数值尺度仍折叠成一个 acquisition candidate，不增加 Oracle 次数。

若当前小半径的初始 hard/smooth 梯度为零，只记录 `zero_initial_crossing_gradient`，不会终止整个 ladder；更大的半径仍可能进入非零梯度或跨界区域。非正 trust radius 现在直接拒绝，不能意外变成“无半径限制”。

### 5. 非单调 homotopy refinement

神经网络分数沿 expert→endpoint 插值不保证单调，因此不能直接对整个区间二分。现在先：

1. 用 32 个区间扫描第一个 safe→margin-satisfied bracket；
2. 在该 bracket 内做 12 步二分；
3. source composite 使用同一个 target clause 做扫描和二分；
4. 最终查询点重新计算 full 与 target-clause score；
5. 仅把查询点上仍满足 `+0.02` 的模型记为 causal rejector。

生成端点与最终查询点的字段被明确分开：

- `generation_hard_margin_*`：优化端点的 `+0.05` 证书；
- `query_hard_margin_*`：细化查询点的 `+0.02` 证书。

不会再用 endpoint 的旧 metadata 假称 refined query 仍有 `+0.05`。

### 6. Oracle query 资格

`model_false_unsafe` 只有同时满足以下条件才可进入 safe 或 global acquisition：

- model-safe anchor 成立；
- generation full/target hard margin 成立；
- refinement 后 source 仍是 causal rejector；
- query full/target boundary margin 成立。

`no_model_safe_anchor`、`hard_margin_not_reached`、错误 clause crossing 和 `no_safe_to_unsafe_crossing` 都只保留诊断，不花 Oracle。

## 多约束错误已被实际拦截

在当前 combined 运行的第 1 轮，`h_vertical_workspace` 是上下界 composite，intervention 指向 `h_vertical_workspace_lower`：

```text
full composite hard score = +0.0744
target lower-clause score  = -0.0335
```

只看整个 composite 会认为 endpoint 已经违约；但指定的 lower clause 根本没有达到 margin。新逻辑将 `generation_hard_margin_satisfied` 判为 false；候选即使进入 refinement 流程，也不会通过 refinement/query gate，因此不 query Oracle。

这正是“feature A 等式约束、feature B 不等式约束”场景所需的行为：每次 probe 有明确的 clause witness，同时整体 hypothesis 也必须真实拒绝。

默认调度也不再长期只测试第一个 clause：第一轮从 clause 0 开始，下一轮平移到 clause 1；若同一候选池内出现多个 false-unsafe probe，则按 false-unsafe 的出现序号继续轮换，而不是按所有 intervention 的绝对位置跳转。LLM 给出的 composite intervention 若缺少 `clause_id`，系统会在变量唯一匹配时补全；若变量与指定 clause 不一致，则把变量对齐到该 clause，并保留修复说明。当前 v8 中 `h_vertical_workspace` 的目标已按轮次从 `lower` 切到 `upper`，而不是每轮重复 `lower`。

## 固定 hypothesis bank 实验

共同设置：

- GPT round-0 bank SHA256：`a4b1701d79da5c5fcd4fd242a53a62d78d7aec86b198469410ae7ee7365217d4`；
- 5 个初始 hypotheses，revision frozen；
- seed=7，同一数据划分、训练配置和 Oracle；
- 每轮 4 次 outer queries，共 26 次 Oracle；
- 每次运行 `llm_interactions=0`；
- 正式 artifact 为 `outputs/*_v8`；七组 `implementation_manifest.json` 完全一致，manifest 文件 SHA256 为 `AF1222967153EDACDE3690E276E6E652839AE0AEC8F02B30182A354F65D6873A`，并且其中记录的源码 hash 与当前实现一致。

复现边界：当前 manifest 只覆盖 runner、falsifier、learner、loop 与 hypotheses 五个核心源码文件；它尚未固定全部数据文件、split、ground truth、其余模块和依赖版本，因此是本次核心实现的 provenance 证据，不是完整运行环境快照。

注意：“旧目标”组只关闭 hard-margin **优化 objective**。anchor、hard certificate、refinement 和 query eligibility 在所有组中保持一致，所以它不是整个 hard-margin 机制的完全关闭。

| false-unsafe 设置 | generation crossings | queried / Oracle-safe | outer safe / violation | false-unsafe optimizer runs | IoU | false-safe | false-unsafe |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 smooth objective，单半径 0.08 | 1/13 | 1 / 0 | 0 / 12 | 26 | 0.8391 | 8.87% | 0.707% |
| hard objective，单半径 0.08 | 1/13 | 1 / 0 | 0 / 12 | 26 | 0.8412 | 8.87% | 0.685% |
| 旧 smooth objective，ladder 至 0.32 | 2/13 | 2 / 1 | 1 / 11 | 90 | 0.7985 | 12.63% | 0.773% |
| hard objective，ladder 至 0.32 | 4/13 | 4 / 1 | 1 / 11 | 86 | 0.8305 | 9.14% | 0.773% |
| 旧 smooth objective，单半径 0.32 | 2/13 | 2 / 1 | 1 / 11 | 26 | 0.7995 | 12.10% | 0.817% |
| hard objective，单半径 0.32 | 4/13 | 4 / 1 | 1 / 11 | 26 | 0.8329 | 8.87% | 0.773% |

所有六组 champion 都是 `h_spatial_exclusion`。

### 如何解释

1. **`0.08` 是明显瓶颈。** 在单半径 `0.08` 下，换 hard objective 仍只有 1 个 crossing。
2. **hard objective 的独立收益在 `0.32` 控制中最清楚。** 相同单半径下由 2 个 crossing 增至 4 个，IoU 从 `0.7995` 到 `0.8329`。这仍只是单 seed，不能当统计结论。
3. **ladder 与更大最大半径不能混为一谈。** 与单一 `0.32` 相比，ladder 没增加 crossing 数或 Oracle-safe 数，却使用更多 optimizer runs。
4. **ladder 提供了局部性。** 第 3 轮 spatial probe 在 ladder 中首次于 `0.16` 成功，refined deformation 为 `0.1446`；单一 `0.32` 对照的 deformation 为 `0.2053`。两者 Oracle 都判 violation，因此本次局部性没有转化为标签收益。
5. **生成可达性改善不等于最终指标必涨。** combined IoU 没有超过本 seed 的最佳单半径结果。

ladder 还同时改变了总优化计算量，因此不能把 crossing 差异单独归因于“分层”这一形式。要做严格计算预算消融，需要固定总 restart×gradient-step 数；表中的 `false-unsafe optimizer runs` 是 13 个 false-unsafe 候选的 radius/restart 尝试总数（包括最终 fallback 前的失败尝试），不包含其他 intervention 的优化。当前报告显式列出该成本，不做过度因果解释。

## safe acquisition 对照

同一 hard objective + ladder：

| selector | generation crossings | queried / Oracle-safe | outer safe / violation | IoU |
|---|---:|---:|---:|---:|
| strict safe acquisition | 4/13 | 4 / 1 | 1 / 11 | 0.8305 |
| global-only | 5/13 | 1 / 0 | 0 / 12 | 0.8146 |

这不是全程静态候选池 A/B：两组只在第一次模型更新前共享相同条件；不同 Oracle 标签进入训练后，后续模型和候选池会自然分叉。因此只能说本次 strict policy 找到一个有价值的 spatial false-unsafe，并得到更高 IoU，不能据此声称稳定收益。

## 真正有信息量的反例

combined 第 1 轮的 `h_spatial_exclusion` probe：

- anchor hard score：`-0.0575`；
- selected radius：`0.32`；
- generation endpoint margin：`+0.0552`；
- refinement alpha：`0.7140`；
- refined query margin：约 `+0.0200`；
- 最终最大 expert deformation：`0.2205`；
- source model：violation；
- Oracle：safe。

这 1 条才是真正的 model-false-unsafe：模型接受 expert anchor、拒绝 candidate，而 Oracle 认为 candidate 仍可行。

另外 3 条 certified probes 被 Oracle 判为 violation。它们不是生成失败；它们确实跨过模型边界，只是模型在这些方向上判断正确。

## 验证

完整回归：`57/57 tests passed`。

新增或强化的测试覆盖：

- 非零 calibrated threshold 下的 hard-margin objective；
- smooth score 很高时不能冒充 hard crossing；
- composite 必须跨越指定 clause，而非更早的 nuisance clause；
- nuisance clause 在小半径先越界时，ladder 必须继续到 target clause 真正达标；
- 未知 clause ID 不能静默 fallback；
- 初始轨迹已达 margin 时在 optimizer step 前 checkpoint；
- 早期 hard-margin checkpoint 几何无效、后期 checkpoint 有效时保留后者；
- 小半径零梯度不能提前终止 ladder；
- radius ladder 在第一个合格 rung 停止；
- ladder 耗尽时不虚报 crossing；
- 非单调 homotopy 的第一段 crossing；
- 非正 trust radius 被拒绝；
- generation 与 query margin metadata 分离；
- generation 合格不能替代 refined query margin；
- 失败的 false-unsafe candidate 没有 Oracle query eligibility；
- 多个 radius 只形成一个 acquisition candidate、只调用一次 Oracle；
- composite clause 跨轮轮换、同轮多个 false-unsafe occurrence 轮换，以及存在 pending intervention 时的 offset 都不会饿死后续 clause。

## 5-seed 后续实验（已完成）

固定 seeds `[7,19,37,73,109]` 的严格配对实验已经完成。两臂各查询 15 条
certified crossing probe，Oracle-safe 都是 `0/15`；五个 seed 的实际查询扰动中位数
完全相同。ladder 只在 15 条中的 1 条降低扰动，却把 false-unsafe optimizer launches
从 `130` 增加到 `424`（`3.26x`）。因此预注册规则选择更便宜的 `hard single-0.32`。

完整结果见 `FALSIFIER_MULTISEED_SINGLE_VS_LADDER_2026-08-17.md`。当前默认值已改为
单一 `0.32`；ladder 仍保留为显式消融选项。这个实验还显示只有 2/5 seeds 产生 qualified
spatial champion，另外 3/5 被 gate 判为 inconclusive；下一步应优先稳定 numeric fitting
与结构选择，而不是继续增加 safe quota。
