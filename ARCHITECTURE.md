# Semantic–Numeric Dual CEGIS Architecture

## 1. 问题定义

训练侧可见：

- 任务语言描述 `d`；
- 公共观测变量 schema `S`；
- 专家安全轨迹集合 `D_E`；
- 仅返回整条轨迹 `safe/violation` 的 membership Oracle `O(τ)`。

训练侧不可见：障碍物圆心、半径、状态级碰撞标签、评估 IoU 和真实违规时刻。

目标是同时搜索结构 `H` 和数值约束参数 `ω`：

```text
H = (variables, coupling, relation, temporal_operator, model_family)
g_(H,ω)(x_t) = state violation logit
G_(H,ω)(τ) = temporal_aggregate_t g_(H,ω)(x_t)
```

约定 `G <= 0` 为安全，`G > 0` 为违约。

## 2. 外层语义 CEGIS

Qwen 初始输出候选群 `B_0={H_1,...,H_K}`，而不是唯一答案。编译器验证变量、枚举值、维度与关系语义。

每个外层轮次后，Qwen 只接收 EvidenceCompiler 产生的摘要：

- safe accuracy；
- violation recall；
- held-out expert consistency；
- false-safe/false-unsafe 数量；
- ensemble uncertainty；
- 假设专属 intervention 的 violation yield；
- 结构复杂度和综合分数。

Qwen 可输出：

```text
retain_and_query
retire_hypothesis
change_variables
change_coupling
change_temporal_operator
change_model_family
split_hypothesis
add_hypothesis
propose_intervention
```

每轮必须指定一个 semantic champion。编译器禁止同轮同时保留并淘汰同一冠军。

## 3. 内层数值 CEGIS

每个 active hypothesis 拥有独立模型，不共享偷偷编码的真值特征。

训练使用 multiple-instance semantics：

- 安全轨迹的所有状态都应安全；
- 违规轨迹至少存在一个违规状态；
- `max` 使用 normalized smooth-max；
- latent witness 由相对已知安全状态的新颖度估计。

Falsifier 在保持起终点、工作空间和步长限制下优化轨迹。目标根据 intervention 改变：寻找 model-false-safe、model-false-unsafe、边界高不确定点、捷径或局部特征 stress。候选交给唯一的轨迹级 Oracle 后加入共享 evidence buffer。

## 4. 编译语义

结构字段不是注释，而会改变计算：

| 字段 | 编译结果 |
|---|---|
| variables | 输入特征及归一化范围 |
| coupling=joint | 一个联合 MLP |
| coupling=independent | 每变量独立 MLP，再以 OR/max 合成 |
| model_family=mlp/linear | 非线性隐式边界或线性对照边界 |
| forbidden_region | 所选神经模型的隐式等值面 |
| upper/lower_bound | 单变量可学习阈值和正尺度 |
| max/mean/last | 不同轨迹级聚合计算图 |
| intervention | 不同 Falsifier 初始化与目标 |

含多个变量的 `upper_bound/lower_bound` 因语义不明确会被编译器拒绝。

## 5. 防泄漏边界

`CircularEvaluationOracle` 有两种能力：

- `membership_view()`：只暴露 `query(trajectory)` 和查询计数，传给学习器；
- `state_violation_mask/evaluation_geometry`：只由 `evaluation.py` 在训练选择之后调用。

EvidenceCompiler 不接收 evaluation oracle。外层 reasoner 调用发生在 evaluator 之前，而且输入文件中没有 IoU 或几何信息。终端可以打印 evaluation-only IoU，但它不会进入 prompt。

## 6. 为什么这比第一版更深入

第一版是：

```text
Qwen once → fixed (x,y,max,MLP) → Neural CEGIS
```

第二版是：

```text
Qwen hypothesis population
 → independently compiled learners
 → hypothesis-specific experiments
 → Oracle evidence
 → Qwen revises population and experiments
 → repeat
```

因此删除或固定语义环，会实际改变活跃模型、查询方向、Oracle 数量和最终约束结构，可通过 `run_ablation.py` 检验。
