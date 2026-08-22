# Numeric fitting stability: full evidence + source-anchor masked MIL

## 结论

这一步没有再改 radius，也没有增加 safe quota 或 Oracle 预算。修正的是轨迹级
MIL 对 violation 的数值归因方式：每个 ensemble member 使用完整的 trainable query
buffer；`warmup_validation` 与 `final_calibration` 仍不参与梯度。对于“当前
hypothesis 自己从已知安全 expert 生成、随后被 Oracle 判为
violation”的轨迹，`max` witness 不再允许落在所选特征与安全 anchor 完全相同的
状态上。

严格的三臂、五 numeric-seed 闭环实验通过了实验启动前冻结的规则，因此默认 numeric fitting
改为：

```yaml
trainer:
  bootstrap_queries: false
  violation_pooling_mode: source_anchor_changed_states
  violation_pooling_change_tolerance: 0.000001
```

公开证据上的结果是：

- classic bootstrap：`2/5` 个 qualified spatial champions，spatial 本身 `2/5` eligible；
- full buffer、旧 all-state MIL：`3/5` champions，spatial `4/5` eligible；
- full buffer + source-anchor mask：`4/5` champions，spatial `5/5` eligible。

最后一臂五个 seed 的 structure-audit 和 fit-expert safe rate 都是 `1.0`，平均
violation recall 与未加 mask 的 full-buffer arm 完全相同，都是 `0.6511`。这说明
至少在公开证据上，收益不是以降低 violation recall 换取的。

## 原错误是怎样产生的

普通 violation MIL 只知道整条轨迹至少有一个违例状态：

```text
G(V) = smooth_max_t g(V_t)
```

很多 query 是由已知安全专家轨迹 `E` 变形得到的，并且保留起点和终点。旧实现
允许 `smooth_max` 把 violation credit 分给任意时刻，包括满足
`phi_h(V_t) == phi_h(E_t)` 的共享端点。但同一个 selected-feature 表示已经在安全
anchor 中出现，它不可能解释为什么新轨迹的整轨迹标签从 safe 变成 violation。

seed 37 的 full-buffer 模型实际学到了这个 shortcut：多个 violation bag 共享的安全
起点被一个 ensemble member 打成正分，最终误拒多个专家轨迹。单纯增加 epoch、成员
数、hard-max loss、member calibration 或 unanimous vote 都没有同时保住安全率和
violation recall。

新的 causal support 为：

```text
M_h(t) = || normalize(phi_h(V_t)) - normalize(phi_h(E_t)) ||_2 > 1e-6
G_masked(V) = logsumexp_{t in M_h}(beta * g(V_t)) / beta
              - log(|M_h|) / beta
```

它只在以下条件全部成立时启用：

1. Oracle 标签是 violation；
2. `source_hypothesis_id` 等于当前拟合的 hypothesis；
3. `expert_id` 能唯一解析到已知安全 anchor；
4. candidate 与 anchor 的 horizon/shape 一致；
5. selected-feature 表示确实有变化。

其他记录继续走原 all-state MIL；完全不可表示、缺 anchor 或 shape 不一致时也回退
旧 loss 并记录诊断。Composite hypothesis 使用 clause-aware mask：无变化的 clause
既不贡献分数，也不接收梯度；`last` 只有末状态特征改变才可用。原有 latent witness
loss 没有被替换。

这个机制不使用障碍物圆心、半径、state-level collision label 或 private IoU，也不
声称定位了真实碰撞时刻。在本实验中，spatial source-matched probes 的 100 个时刻
通常有 98 个发生变化；mask 主要排除了精确共享的首尾端点。它解决的是明确的共享
anchor shortcut，而不是完整的 violation localization。

## 严格实验设计

实验启动前冻结的 seeds 为 `[7, 19, 37, 73, 109]`，统计单元是 seed。三臂为：

| arm | query fitting | violation pooling |
|---|---|---|
| `classic_all_states` | 每成员有放回 bootstrap | 全时刻 |
| `full_all_states` | 每成员使用完整 trainable buffer | 全时刻 |
| `full_source_anchor_mask` | 每成员使用完整 trainable buffer | source-anchor changed states |

所有 arm 固定同一 GPT hypothesis bank、hard single-`0.32` falsifier、safe acquisition、
三个 outer rounds 和 CPU 单线程设置。每个 run 都是 14 个 warmup queries 加每轮 4
个 outer queries，总计 26 次 Oracle；LLM interaction 为 0。后续 query pool 可以因
模型更新不同而分叉，因此这是 paired-seed 的端到端算法比较，不是假装全程共享同一
candidate pool 的静态重排。

采用 candidate 的实验前冻结条件是：

- 至少 `4/5` 个 qualified spatial champions；
- 数量严格超过两个 baseline；
- 相对任一 paired baseline 都没有公开 safe/expert/fit-expert safety-floor 回退；
- private geometry 不进入规则。

## 结果

| arm | qualified spatial champion | spatial eligible | median public safe | median public violation recall | mean fit coverage | mean spatial IoU* |
|---|---:|---:|---:|---:|---:|---:|
| classic all-state | 2/5 | 2/5 | 1.000 | 0.600 | 0.656 | 0.504 |
| full all-state | 3/5 | 4/5 | 0.857 | 0.667 | 1.000 | 0.663 |
| full + anchor mask | **4/5** | **5/5** | 0.857 | 0.667 | 1.000 | 0.666 |

`*` IoU 使用 private geometry，只是选择完成后的诊断。

Masked arm 的逐 seed 公开 spatial 证据：

| seed | final champion | spatial eligible | safe accuracy | violation recall | structure audit safe | fit expert safe |
|---:|---|---:|---:|---:|---:|---:|
| 7 | spatial | yes | 0.857 | 0.667 | 1.000 | 1.000 |
| 19 | spatial | yes | 0.857 | 0.667 | 1.000 | 1.000 |
| 37 | spatial | yes | 0.833 | 0.700 | 1.000 | 1.000 |
| 73 | spatial | yes | 0.857 | 0.667 | 1.000 | 1.000 |
| 109 | affine spatial | **yes** | 0.857 | 0.556 | 1.000 | 1.000 |

seed 109 不是 fitting 失败：spatial 已通过全部 qualification gates。它没有成为最终
champion，是因为 affine spatial 也合格，而现有结构排序的 MLP parameter/capacity
penalty 以及 spatial 较低的 intervention yield（`0.667` 对 `1.0`）共同抵消了
spatial 较高的 balanced accuracy。这里剩下的是 qualified-model
ranking / simplicity-prior 问题，不能用 private IoU 事后把 spatial 强行选回来。

## 被否决的数值方案

在同一固定-query、严格六次 fit 时序下，以下方案没有成为默认：

- coverage-preserving bootstrap：能避免遗漏记录，但重复权重仍令 gate 在 5/10 与
  6/10 violation hits 之间跳动；
- ensemble unanimous/min aggregation：公开 prequential violation recall 被压低，
  `0/5` 过 gate；
- member-wise calibration/normalized margins：同样偏向 all-safe，`0/5`；
- 纯 hard-max 训练：从 smooth surrogate 退化，full-buffer 为 `0/5`；
- safe/violation 同权 hard alignment：改善个别专家误拒，却普遍损失 violation recall；
- 对所有 source witness 直接加点级 violation loss：near-anchor probe 会把安全边界反向
  推坏；
- 额外 candidate-anchor margin：mask-only 为 `5/5` fixed-query eligible，加入 margin
  后降为 `4/5`。

因此当前改动保持 normalized smooth-max，只删除因果上不可能的共享-anchor witness。

## 可复现产物与限制

- 运行前冻结的计划：`configs/violation_pooling_multiseed_plan.yaml`
- sweep runner：`run_violation_pooling_multiseed.py`
- 原始 15 runs：`outputs/violation_pooling_multiseed_5seed_v1/`
- 汇总：`outputs/violation_pooling_multiseed_5seed_v1/violation_pooling_multiseed_summary.json`
- 自动短报告：`outputs/violation_pooling_multiseed_5seed_v1/VIOLATION_POOLING_MULTISEED_REPORT.md`

Artifact validation 为 `PASS`：15/15 runs 都是 26 Oracle、0 LLM，warmup trajectory
hash/label/role split 一致，三臂配置只含两个运行前冻结的 treatment 差异，candidate 的 mask
确实被激活且没有 unresolved/invariant spatial pair。`...unique...total` 是跨 fit/member
累加的 unique-per-fit exposure，不是整次运行的全局唯一 query 数；最终 buffer 的每成员
计数见 `spatial_final_unique_source_anchor_masked_violation_counts`。

每个 run 的 `implementation_manifest.json` 保存运行时源码与输入的哈希指纹。正式比较
完成、运行前冻结的规则选出 candidate 后，才把常规配置和 `TrainerConfig` 的默认值切换为
candidate；三臂运行命令本身始终显式指定 bootstrap 与 pooling treatment，因此这个
事后默认切换不改变上表的算法比较。

五个固定 seeds 足以做工程默认选择，不构成统计显著性声明。结论目前只覆盖这个冻结
hypothesis bank 与 Obstacle2D；其他任务仍需要重复 paired-seed 验证。下一项值得研究的
已不是继续调 fitting，而是 seed 109 暴露的 qualified spatial 与 affine 之间的结构排序。

## 后续结论：用可证明的结构矛盾处理 seed 109

后续工作没有事后重调 complexity/capacity penalty，而是加入了 linear-max 凸支撑序
证书：若 Oracle-violation candidate 的全部所选特征点位于对应安全 expert anchor 的
凸包内，则任意 affine score 的 candidate 最大值都不可能超过 anchor 最大值；同一阈值
因而无法把前者判为 violation、后者判为 safe。至少两个不同 anchors 都产生矛盾后，
enforced 模式才把该 linear-max hypothesis 排除出 champion 候选。

旧 15-artifact 开发回放把 full-buffer + mask 从 `4/5` 变为 `5/5`，并修正了 seed 109；
这只用于验证机制。随后运行前冻结的新五种子正式两臂实验中，audit-only 与 enforced
都得到 `5/5` qualified spatial champions，且公开安全门槛没有回退。因此采用 enforced
作为 sound guard，但不声称性能提高。完整证明、严格 one-sided 实现、26-Oracle 精确
配对与 artifact 哈希见
[LINEAR_MAX_SUPPORT_GATE_2026-08-17.md](LINEAR_MAX_SUPPORT_GATE_2026-08-17.md)。
