# Hard single-0.32 与 hard ladder：5-seed 配对实验

日期：2026-08-17

## 结论

默认采用 **hard-margin + 单一 `0.32` trust radius**，不默认启用
`0.04 -> 0.08 -> 0.16 -> 0.32` ladder。

在五个预注册 numeric seeds 上，ladder：

- 没有增加任何 Oracle-safe false-unsafe 命中；两臂都是 `0/15`；
- 没有降低任一 seed 的查询扰动中位数；五个 paired reduction 都是 `0%`；
- 只在 15 条查询中的 1 条降低了扰动， pooled mean 仅下降 `0.94%`；
- 将 false-unsafe optimizer launches 从 `130` 增加到 `424`，即 `3.26x`；
- paired median `Delta IoU=0`；仅 seed 37 有 `+0.0053`，其余四个 seed 为 0。

因此 ladder 没有通过预注册的稳定收益规则。该结论不支持继续增加
safe quota：本次两臂合计 30 条 certified false-unsafe probe 全部被 Oracle 判为
violation；在当前 acquisition 与候选分布下，没有证据支持为同类 probe 增加额度，观察到的
结果只会增加未命中查询。

## 实验设计

- seeds：`[7, 19, 37, 73, 109]`，在看结果前固定；
- frozen GPT round-0 bank：5 个 hypotheses，SHA256
  `a4b1701d79da5c5fcd4fd242a53a62d78d7aec86b198469410ae7ee7365217d4`；
- 每个 seed 两臂共享数据、warmup、模型配置、safe acquisition 与 Oracle 预算；
- 每次运行：14 次 warmup + 3 轮 x 4 次 outer query = 26 次 Oracle；
- LLM interactions：全部为 0，semantic revisions frozen；
- CPU，`PYTHONHASHSEED=numeric_seed`，`OMP_NUM_THREADS=1`，`MKL_NUM_THREADS=1`；
- single arm：hard objective，`radius=0.32`，`ladder=[]`；
- ladder arm：hard objective，`[0.04, 0.08, 0.16, 0.32]`；
- 统计单元是 seed，不把同一 run 内的 query 当独立样本。

预注册计划 SHA256：
`efba21d2d82ad58498d296a4e140183a8389cb099808028b26cb6344e9752d3a`。

## Warmup 与 artifact 审计

最初两次尝试没有作为正式结果使用：

1. v1 的旧随机 warmup 在 seed 73 只能得到 4 个 safe 标签，并使 seed 109
   使用 31 次总 Oracle，破坏了固定预算；
2. v2 虽恢复 14 次固定 warmup，但旧的逐标签 tail split 把 pair 2/3
   拆到了 train 与 validation，存在 correlated-family leakage。

v3 使用 deterministic demo-relative pairs，并按整个 pair family 分配
train/validation/calibration。十次运行全部满足：

- warmup trajectory sequence SHA256：
  `efc902562b7d9a15b708cfdcac95a8fec1d2f0801dac21f104f49f911e281594`；
- label sequence：`10101000101010`，即 8 safe / 6 violation；
- role split：train `{0,1,3}`，validation `{2,4}`，calibration `{5,6}`；
- 同一 pair 没有跨 evidence role；
- 26 次 Oracle、0 次 LLM；30 条实际提交的 false-unsafe queries 全部通过
  hard-margin generation/query certificate 与 query gate；
- 10 个运行的代码、输入数据与 frozen-bank fingerprints 一致。

Artifact validator 最终为 `PASS`，0 errors。
切换默认值并补齐 warmup/sweep 回归后，完整测试为 `67/67 passed`。

v3 的 `implementation_manifest.json` 保存的是运行当时的源码/输入指纹（路径与哈希清单）。实验完成后才执行
“把默认值切到 single-0.32”这一决策，并补了通用 warmup 的直线轨迹 fallback 与可分性停止
条件，因此当前源码 hash 与 v3 manifest 不应相同。两臂在 sweep 中通过 CLI 显式指定 radius，
默认值变化不改变实验语义；当前实现重放本 benchmark 的前 14 条 warmup 与 v3 artifact
仍然逐字节完全相同。

## 每个 seed 的原始配对结果

这里的 deformation 是已 refinement、实际送给 Oracle 的轨迹相对 expert anchor 的
`max_expert_deviation`，不是 generation radius，也不是按 radius 归一化的值。

| seed | single safe/query | ladder safe/query | single median deformation | ladder median deformation | single/ladder launches | single IoU | ladder IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0/3 | 0/3 | 0.13814 | 0.13814 | 26/88 | 0.0000 | 0.0000 |
| 19 | 0/2 | 0/2 | 0.11925 | 0.11925 | 26/82 | 0.2876 | 0.2876 |
| 37 | 0/4 | 0/4 | 0.17294 | 0.17294 | 26/86 | 0.6614 | 0.6667 |
| 73 | 0/3 | 0/3 | 0.13562 | 0.13562 | 26/88 | 0.0000 | 0.0000 |
| 109 | 0/3 | 0/3 | 0.13768 | 0.13768 | 26/80 | 0.0000 | 0.0000 |

汇总：

| arm | Oracle-safe/query | pooled median deformation | pooled mean deformation | optimizer launches | gradient-step proxy | mean IoU | median IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| hard single-0.32 | 0/15 | 0.13562 | 0.15081 | 130 | 7,800 | 0.1898 | 0.0000 |
| hard ladder | 0/15 | 0.13562 | 0.14940 | 424 | 25,440 | 0.1909 | 0.0000 |

ladder 的 pooled mean 下降来自 seed 37 的一条 spatial probe：`0.07165 -> 0.05049`。
其余 14/15 条查询扰动完全相同。这个孤立改善不足以满足“至少 4/5 seeds 稳定降低”的规则。

## 为什么 ladder 几乎没有改变最终查询

ladder 确实经常在 `0.16` 而不是 `0.32` 首次找到 generation endpoint；但 endpoint
随后还要沿 known-safe expert 到 endpoint 的 homotopy，细化到统一的 query margin
`threshold + 0.02`。如果两种搜索找到的是同一模型边界分支，refinement 会把它们缩到
几乎同一个 Oracle query。因此 ladder 多花了小半径的 optimizer runs，却通常没有改变
最终查询。

这也解释了为什么本实验不能只报告“selected radius 更小”：真正有意义的是 refinement
后的绝对 deformation 和 Oracle 标签。

## 额外发现：当前瓶颈不再是 radius schedule

两个 arm 的 15 条 certified crossing probes 全部是 Oracle violation，即模型在这些方向上
拒绝得正确；没有产生真正的 model-false-unsafe。与此同时，能否找到合格 champion 对
numeric seed 很敏感：

- seed 19/37 得到 qualified、eligible 的 `h_spatial_exclusion`；
- seed 7/73/109 的 selection status 是 `inconclusive`，`champion_eligible=false`，原因是
  `violation_recall_below_gate`；`result.json` 的最佳可用/占位 ID 是 `h_speed_cap`，其
  private-evaluation IoU 为 0，但不能称为合格胜出；
- 两臂的 median IoU 都是 0。

所以这次实验回答了 radius 选择问题，却也暴露出更大的下游问题：hypothesis selection
与 numeric fitting 对初始化不够稳健。下一步应优先提高“稳定地产生 qualified spatial
champion”的概率，并保留当前 proxy gate，而不是继续扩大 safe-query quota。

## 已实施默认值

- `FalsifierConfig.false_unsafe_trust_radius = 0.32`；
- `FalsifierConfig.false_unsafe_radius_ladder = ()`；
- `obstacle_avoid_gpt_diagnostic.yaml` 使用 `false_unsafe_radius_ladder: []`；
- ladder 保留为显式实验选项：
  `--false-unsafe-radius-ladder 0.04 0.08 0.16 0.32`。

另用 seed 7、相同单线程环境运行了一次不带 radius override 的 post-decision smoke。
它与 v3 `hard_single_032/seed_7` 的 `observations/actions/labels/outer_rounds`、
`result.json` 和 `query_diagnostics.json` 全部完全相同；resolved config 除 output 路径外也
完全相同，并明确记录 `false_unsafe_radius_ladder: []`。

五个 seeds 只支持工程稳定性决策，不构成统计显著性证明；并行 wall-clock 也不是本报告的
速度指标。正式机器可读结果位于
`outputs/falsifier_multiseed_5seed_v3/multiseed_summary.json`。
