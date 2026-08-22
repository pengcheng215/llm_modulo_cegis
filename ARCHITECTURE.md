# Semantic–Numeric Dual CEGIS Architecture

> 本文保留早期设计正文，并将后续实现与验证按时间追加；当前更新到 2026-08-21。

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
- latent witness 默认读取 falsifier 在 Oracle 前记录的 source-model witness；它只是
  生成过程提供的弱定位信号，不是 ground-truth violation-time label。`novelty` 仍保留
  为显式消融模式。

默认 numeric fitting 不再对小型 query buffer 做有放回 bootstrap；每个 ensemble
member 都看到每条可用于梯度拟合的 query 记录。`warmup_validation` 与
`final_calibration` 仍严格排除在训练外。对于 Oracle 判为 violation、且由当前
hypothesis 从已知安全 expert anchor 生成的轨迹，`max` 的可用 support 进一步限制为：

```text
M_h(t) = ||normalize(phi_h(candidate_t)) - normalize(phi_h(anchor_t))||_2 > 1e-6
```

只有 `t in M_h` 的状态可以接收该 source-matched violation 的 MIL credit。因为
`phi_h(candidate_t) == phi_h(anchor_t)` 的表示已经出现在安全轨迹里，它不可能解释
整轨迹标签为何改变。缺 anchor、shape 不一致、表示完全不变或非 source-matching 的
记录继续使用旧 loss；composite 模型按 clause 分别建立 mask。这个支持集只排除因果上
不可能的共享状态，并不等于真实 violation-time label。

三臂五种子闭环实验中，classic bootstrap、full-buffer all-state、full-buffer + mask
分别产生 `2/5`、`3/5`、`4/5` 个 qualified spatial champions；最后一臂的 spatial
模型本身在 `5/5` seeds 都通过 qualification，且每个 seed 的 fit/audit expert safe
rate 都为 `1.0`，violation recall 没有相对 full-buffer all-state 下降。因此默认采用
full-buffer + source-anchor mask，bootstrap/all-state 保留为显式 ablation。

### 3.1 Linear-max 凸支撑序结构门

数值 qualification 之后、nested-minimality 比较之前，EvidenceCompiler 还执行一个
parameter-independent 的结构可实现性检查。设已知安全 anchor 在所选特征上的采样点集为
`A`，与它通过 `expert_id` 关联、被 Oracle 判为 violation 的 candidate 点集为 `V`。
对于 affine state score `g(z)=w^Tz+b`：

```text
V subset conv(A)  =>  max_{v in V} g(v) <= max_{a in A} g(a)
```

因此同一个 `max` 阈值不可能同时接受 anchor、拒绝 candidate。这个结论对所有模型参数
成立，也不读取 obstacle geometry、状态级标签或 IoU。

硬门只用于单 clause、`joint + forbidden_region + linear + max` 的 1D/2D hypothesis。
它忽略 `final_calibration`，要求 `expert_id` 唯一解析到已知安全轨迹，并默认需要至少两个
不同 safe anchors 都产生矛盾。`linear_max_support_tolerance` 只去重近重复 anchor 点，
不会向外放宽 1D 区间、退化线段或 2D 多边形；数值不确定时宁可不发证书。因此它是
one-sided 的充分拒绝条件：不包含、不可解析、不支持的维度或结构全部 fail open，不能
据此证明另一个模型正确。

audit-only 与 enforced 都记录 pair、contradiction、distinct-anchor、unresolved、triggered
和 applied 计数；只有 enforced 把证书命中的 hypothesis 标为 champion-ineligible，原
`selection_score` 和 `query_priority` 不变。已被该结构门淘汰的简单模型也不再通过
nested-minimality 去压制其特征超集。

运行前冻结的五种子两臂实验中，两臂均为 `5/5` qualified spatial champions；证书在
两臂 `5/5` 触发，只在 enforced 的 `5/5` 应用。由此默认采用 enforced 作为 sound guard，
但因为 champion 计数和公开率均打平，不声称性能提高。完整定理、旧 seed 109 回放和
artifact 审计见
[reports/LINEAR_MAX_SUPPORT_GATE_2026-08-17.md](reports/LINEAR_MAX_SUPPORT_GATE_2026-08-17.md)。

Warmup 使用同一 expert、同一尺度的成对 demo-relative deformation：一条向 endpoint chord 收缩，另一条沿专家 detour 方向延伸。整个 pair family 只能进入 training、prequential selection validation 或 threshold calibration 中的一组，不能跨组泄漏；numeric seed 不改变该候选序列。另有独立的 safe-expert structure-audit split，不能与 `warmup_validation` 混为一组。

Falsifier 在保持起终点、工作空间和步长限制下优化轨迹。目标根据 intervention 改变：寻找 model-false-safe、model-false-unsafe、边界高不确定点、捷径或局部特征 stress。它先生成候选池；acquisition 只把少量候选交给唯一的轨迹级 Oracle，再把标签加入共享 evidence buffer。数值优化的多个 restart 或 radius 不会各自调用 Oracle。

false-unsafe 不是靠 intervention 名称认定。source model 必须接受已知安全 expert anchor，并在生成端点达到 calibrated `threshold + 0.05`；系统再沿 expert→endpoint 扫描第一段 crossing，缩回仍保留 `threshold + 0.02` 的查询点。失败候选只有诊断记录，没有 Oracle query 资格。5-seed 配对实验后，默认使用单一 `0.32` trust radius；可选 ladder 逐级尝试 `0.04/0.08/0.16/0.32`，只用于显式实验，因为它没有稳定提高 Oracle-safe yield 或降低查询扰动，却约增加到 `3.26x` false-unsafe optimizer launches。

对 composite hypothesis，生成端点和细化查询点都必须同时满足整个 hypothesis 与 intervention 指定 clause 的 hard margin。因此 feature A 的等式约束和 feature B 的不等式约束可被分别测试，B 的较早 crossing 不能冒充 A 的反例。

默认 false-unsafe 调度按 outer round 和“本轮第几个 false-unsafe probe”轮换 clause，避免多约束模型长期只测试第一个 clause；前面插入 LLM pending intervention 不会改变该 occurrence 计数。语义 intervention 缺少 clause 时仅在可唯一解析或确定性轮换时补全，变量与 clause 冲突时对齐到指定 clause，并把修复写入 rationale 供审计。

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

因此删除或固定语义环，会实际改变活跃模型、查询方向、Oracle 数量和最终约束结构，可通过 `experiments/run_ablation.py` 检验。

## 7. 架构演进记录（2026-08-18，保留原设计）

第 6 节保留的是相对于更早 `llm_guided_cegis` one-shot 原型的概念对照。需要强调：
2026-08-13 的首个 `llm_modulo_cegis` 已经有 typed hypothesis bank、独立 learner、
trajectory membership Oracle 和 outer fit/query/evidence/revise 循环；它不是纯 one-shot。
当时的问题是 Qwen 测试主要依赖 canonical fallback/augmentation，revision 几乎只有
retain，因而语义闭环在实践中作用很弱。本节追加此后累计变化，不覆盖前面各阶段当时的
设计和实验数字。

### 7.1 三个架构阶段

前身 one-shot 原型（早于本目录的 population/outer-loop 实现）：

```text
task text + public schema
  → Qwen once（不合规范时整库 fallback）
  → fixed hypothesis / joint MLP
  → heuristic falsifier for each hypothesis
  → Oracle label
  → bootstrap fit
  → one mixed scalar score
```

中期双层循环：

```text
typed hypothesis population
  → independent learners and hypothesis-specific candidates
  → shared global acquisition budget
  → frozen pre-query predictions + Oracle labels
  → prequential qualification
  → GPT/Qwen revision for the next round
```

当前架构（语义输入和 demonstration 数据路径分开）：

```text
task text + public schema
  → GPT strict output / Qwen repaired surface form
  → trusted typed compiler
  → typed, canonicalized, possibly composite hypothesis bank

safe demonstrations
  → split fit experts / structure-audit experts / evaluation-only experts
  → deterministic demo-relative warmup pairs + whole-trajectory Oracle
       ├─ trainable evidence
       ├─ held-out prequential selection validation
       └─ disjoint final calibration
  → for each outer round
       1. fit every active model on the complete trainable buffer
       2. freeze validation/pre-query predictions
       3. generate clause-targeted candidate pools;
          false-unsafe probes additionally require hard crossing certificates
       4. sequentially spend one shared Oracle budget
       5. refit, compile prequential evidence, and apply structural gates
       6. choose an eligible champion, or return a provisional ID with
          selection_status=inconclusive; revise only if another round remains
  → optional transactional finalization
  → evaluation-only metrics after each round's semantic decision and at the end;
    never feed them back into evidence, prompts, acquisition, or selection
```

多尺度优化、多个 restart 和多个 active hypotheses 都不会各自自动获得一次 Oracle
调用。它们只扩大尚未标注的候选池；真正提交给 Oracle 的仍是共享预算选出的少量整轨迹。

### 7.2 逐项变化、原因与当前边界

| 导师指出或实验暴露的问题 | 根因 | 当前处理 | 仍然不能声称什么 |
|---|---|---|---|
| Qwen 输出经常不可执行，实际一直 fallback | 小模型的 schema 遵循和语义枚举不稳定 | GPT 使用 strict Structured Outputs；Qwen 做表面修复后仍经过可信编译器；成功响应的 parse/rejection 会审计，backend failure 默认 fail-fast，只有显式允许时才 fallback | 不能声称 1.5B Qwen 已可靠完成多轮语义推理 |
| joint all-feature MLP 看似能覆盖所有约束 | 假设表示有包含关系，单一分数又同时奖励拟合与惩罚复杂度 | typed alternatives 和独立 qualification 给简单结构候选空间；nested/progress/dynamics 以“需要明显增益”的可配置证据先验抑制代理变量；terminal collision 与 linear-max support order 才是精确结构矛盾 | 不能把经验性 minimality prior 当定理，也不能全面禁止 MLP；当多个可表达结构都合格时，一般排序仍未完全解决 |
| falsifier 太 heuristic | smooth surrogate、固定小半径和最终 iterate 不等于真实模型 crossing | 对 false-unsafe 增加 model-safe anchor、calibrated full/target-clause hard margins、有效 checkpoint、非单调扫描括区和 refined query certificate | “model_false_unsafe” 仍只是探针目标；只有 Oracle 返回 safe 才是真正反例；false-safe/shortcut/boundary/stress 仍是一般优化与几何校验，不是 hard-certified |
| 每个 hypothesis 都查 Oracle，成本高 | 生成和查询没有分层 | 全 bank 先生成并冻结预测，再用一个 acquisition policy 顺序分配全局预算；一次标签同时为所有模型生成 evidence | 没有消除 membership Oracle 假设，也没有证明 26 次是所有任务的最小预算 |
| safe 标签不足，但盲目加 quota 会挤掉信息查询 | 旧策略只预留一个槽且 batch 预选，失败后不重试；后来又可能选择全模型都判 safe 的无纠错候选 | 只统计 trainable labels，按每次 Oracle 返回重算 safe/violation deficit；两类 balance-seeking 合计最多两槽，safe 侧还要求 causal rejector，并按 clause-aware signature 去重 | label quota 只改善数据覆盖，不等价于结构信息，也不能保证 Oracle-safe yield |
| feature A 等式 + feature B 不等式可能串扰 | composite 的 full score、target clause、refinement 和调度身份曾不一致 | clause ID 贯穿 intervention、objective、generation certificate、refinement、query gate、dedup；跨轮按 false-unsafe occurrence 轮换 clause | 当前 convex-support hard gate 不覆盖 composite；复杂组合仍需逐 clause 验证 |
| “专家绕障碍时改变 y”被误解成“任务本身要求某个 y 走廊” | 稀疏安全演示让 detour 形状与可行性相关；早期候选/负例分布和较弱 ranking 允许 y-only 或 progress proxy 存活，数值阈值还可能放宽成近 all-safe，而不是真正学到严格高-y 规则 | warmup 沿每条专家自身 detour 做 toward/continue 成对变形，而不是全局正/负 y 抖动；保留多 anchor；per-class qualification、held-out expert consistency、progress/dynamics proxy gain、terminal representation collision 和结构查询共同约束代理规则 | 整轨迹二值标签仍不提供真实碰撞时刻；有限 demonstrations 下不能从统计相关性自动证明因果障碍结构 |
| terminal goal-band 曾错误成为 champion | 保持 endpoint 的 safe expert 与 violation shortcut 在 pure-last 表示中完全相同，但旧评分没有做可表达性冲突检查 | 对 pure-last 表示分组；相同 terminal representation 出现相反标签即 champion-ineligible | 该门只否定 pure-last 结构，不证明 spatial MLP 正确 |
| numeric fitting 对 seed 敏感 | 小 buffer 有放回 bootstrap 会让每个 member 漏掉大量已付费记录；all-state MIL 会把未改变的安全端点当 violation witness | 默认 full trainable buffer；source-matched violation 只在相对安全 anchor 已改变的表示状态上聚合 MIL credit | changed-state mask 不是 ground-truth violation-time mask，也不适用于没有可解析 anchor 的所有记录 |
| qualified spatial 仍可能被 affine 复杂度先验反超 | champion scalar 混入 adaptive yield、模型复杂度、参数量和未校准 uncertainty | 两个独立 anchor 上的 convex-support order contradiction 可参数无关地淘汰 atomic 1D/2D joint linear-max `forbidden_region`；正式新五种子 audit/enforced 均为 spatial `5/5`，故只作为无回退 sound guard 采用 | 正式比较没有 champion-rate 增益；旧 seed 109 翻转仅是开发机制回放 |
| 训练后再调 threshold/refit 可能污染选择 | calibration、selection 和 gradient data 曾可能角色不清 | pair-family 级 train/validation/calibration 隔离；finalization 先校准候选，再用独立 validation 一次性选择，失败则回滚 | 小型 2S/2V holdout 仍然分辨率有限，不能当统计显著性证据 |

#### “终点已经下降”究竟能否反驳高-y 假设

这里必须先纠正一个容易混淆的说法：`temporal_operator=max` 聚合的是每个状态的
**violation score**，不是无条件计算 raw feature 的 `max(y)`。因此要同时看 relation 和
temporal operator：

- `lower_bound(y)` 的 state violation score 随 y 降低而增大，再做 temporal `max` 等价于
  检查全轨迹最坏的低-y 状态。因此如果专家在终点真的降到所学阈值以下，它**会**反驳
  “全程保持高 y”的严格规则。
- `upper_bound(y)` 配合 temporal `max` 检查的是最坏的高-y 状态；`forbidden_region` 的
  MLP/linear score 更不能直接解释成 raw `max(y)`。
- `last` 只看终点。如果 safe expert 和保持 endpoint 的 violation shortcut 终点相同，
  pure-last 结构无法解释相反标签；当前 representation-collision gate 会直接否定它。
- `mean` 或 progress proxy 仍可能利用数据集中偶然稳定的路径相关性。当前通过不同
  detour families、成对 counterfactual、held-out expert safety 和相对简单模型的增益门来
  削弱这种代理解释，但 whole-trajectory Oracle 无法提供完全的状态级因果识别。

旧运行里所谓“高-y 误解”更准确地说，是语义候选把 y 相关性当成任务结构，而数值模型
又可能把阈值放得过宽、接近 all-safe；当时的 violation-recall/structure gates 没有及时
淘汰它。它不意味着模型在数学上成功拟合出一条会无视低终点的 strict lower-bound。

因此修正不是硬编码“障碍一定从上/下绕”，而是让候选同时看到不同专家路径，并主动
构造只改变路径内部、保留任务 endpoint 的结构区分轨迹。

### 7.3 当前默认值由哪些实验决定

| 当前默认 | 对照结果 | 采用含义 |
|---|---|---|
| complete trainable buffer，`bootstrap_queries=false` | classic bootstrap、full all-state、full + source-anchor mask 分别得到 spatial champion `2/5`、`3/5`、`4/5`；mask 下 spatial 本身 `5/5` eligible | 不再随机遗漏已付费 Oracle evidence |
| `violation_pooling_mode=source_anchor_changed_states` | 在正式三臂实验中通过公开安全/召回门，fit/audit expert-safe 均稳定 | 排除因果上不可能解释标签变化的共享状态，不宣称定位真实违约时刻 |
| false-unsafe single radius `0.32` | 五种子中 single 与 ladder 都没有稳定提高 Oracle-safe yield/扰动；ladder optimizer launches 约为 `3.26x` | 选择较便宜搜索，不声称 single 更准确 |
| `linear_max_support_gate_enforced=true` | 新五种子 audit-only/enforced 都是 qualified spatial champion `5/5`；证书两臂都触发，仅 enforced 应用 | 采用严格结构保护且未观察到公开门槛回退，不声称性能提升 |
| total label-balance cap `2` | safe/violation deficit acquisition 可以顺序重试，同时保留至少一半 round budget 给全局信息候选 | 防止继续扩大标签 quota 取代结构辨别 |

这些结果的统计单位是 numeric seed，而不是单条 query。五个 seed 支持工程默认选择，不
构成统计显著性结论；冻结-bank 实验验证的是 numeric/acquisition/selection 机制，不等同
于证明 live GPT revision 在所有任务上稳定。

### 7.4 没有改变的任务假设

当前方法仍要求三类用户信息：任务描述、已知安全 demonstrations，以及对少量新整轨迹
的 `safe/violation` membership 判断。Oracle 不接收或返回障碍几何与状态级标签；private
IoU、false-safe/false-unsafe grid metrics 会在每轮 provisional champion 与 semantic
actions 已确定后作为 evaluation-only side channel 计算，并在 optional finalization 后重算；
它们不进入 EvidenceCompiler、LLM prompt、revision、acquisition 或 champion score。当前保证
是代码路径上的 non-use，不是把 private evaluator 从进程中物理隔离。LLM 负责提出结构和
实验方向，数值模型负责边界，Oracle 只负责回答主动选择的整轨迹是否可行。

### 7.5 实现能力、运行 profile 与剩余边界

- CLI 默认仍选择 Qwen profile；普通 `gpt.yaml`/Qwen 配置和正式 diagnostic profile 的
  audit expert 数、rolling window、qualification floors 与 finalization 开关并不完全相同。
  因而上文的“当前能力”不等于每个配置都默认启用全部机制；普通配置关闭 finalization 时，
  也不能宣称每轮都执行了独立 threshold calibration。
- live GPT integrated run 验证过真实 structured-output/revision 路径；此后的 falsifier、
  acquisition、numeric-fitting 与 linear-gate 五种子实验主要固定 GPT round-0 bank、冻结
  revision、使用 `0` 次 LLM。它们不能证明 Qwen 已改善，也不能证明 live GPT 多轮语义稳定。
- source-anchor mask 只约束 source-matched violation；无法解析 anchor、表示完全不变或其他
  hypotheses 共享同一标签时仍可能回退到旧 MIL bag。它排除已知不变状态，不是真实碰撞时刻。
- falsifier 只检查端点、workspace、max-step 等运动学条件，没有通用机器人动力学或任务可行性
  证明；membership Oracle 也尚无 abstain/noise model。
- 正式证据仍集中在 Obstacle2D、固定 hypothesis bank 和五个 numeric seeds；一般多结构排序、
  distractor-rich/heterogeneous benchmark、跨任务泛化及 live-semantic 多种子验证仍待完成。
- artifacts 保存查询、预测、证据与最终模型摘要，但不保存所有未查询候选的完整轨迹或每轮完整
  checkpoint，因此能审计决策，尚不能从单个 artifact 完全重放每次中间优化。

### 7.6 当前轨迹生成范围与尚未实现的全局生成（2026-08-21）

设被选中的专家轨迹为 $E=(e_0,\ldots,e_{T-1})$，端点弦为
$L_t=(1-s_t)e_0+s_t e_{T-1}$。warmup 的 `toward_chord` 与
`continue_detour` 分别沿

$$
E-\alpha(E-L),\qquad E+\alpha(E-L)
$$

构造一对 demonstration-relative 轨迹。outer-loop falsifier 的初始化则是

$$
X^{(0)}=(1-\mu)E+\mu L
+a\sin(\pi s)\sin(k\pi s)d,
$$

随后只优化内部 waypoint，并固定 $e_0,e_{T-1}$。所有 intervention mode 都有相对 $E$ 的
距离正则；只有 `model_false_unsafe` 额外满足逐时刻
$\max_t\lVert X_t-E_t\rVert_2\le 0.32$ 的 hard trust region，并需要 full-model 与指定
target-clause 的 crossing certificate。因此当前实现应称为
**expert-anchored, model-guided counterfactual search**，而不是全局轨迹生成。

这种局部设计有合理性：已知安全 anchor 使 safe-to-model-unsafe crossing 有明确参照，并减少
无意义或运动学无效的 Oracle 候选。但它也把搜索支持偏向已演示路径及其 endpoint chord。
多方向 smooth-basis restart 可能到达其他区域，却不保证覆盖下绕/上绕等新的 path family 或
homotopy class。这里的 **global acquisition** 只表示在跨 hypothesis 的共享候选池中统一分配
Oracle 预算，不表示轨迹是在全局空间中直接合成。

下一步拟议的 `global_direct_spline` 分支仍是**设计，尚未实现**。它应只使用显式 public
`TaskSpec` 中的 start/goal/horizon/workspace，采样低维控制点并插值成平滑样条，经
workspace/step/curvature 检查后，按冻结模型的 disagreement、boundary proximity、novelty 和
structure discrimination 排序。如果端点仍来自 demonstration，只能准确地称为“不使用专家
内部 waypoint”，不能称为完全 expert-independent。该分支不得读取隐藏障碍几何，也不能自造
safe/violation 标签；最终标签仍来自同一个 whole-trajectory Oracle。

合理的验证不是给它增加额外查询，而是在固定 hypothesis bank、numeric seeds、候选计算预算和
Oracle budget 下比较 `local-only`、`global-only` 与 `hybrid`。主要指标应是结构恢复、proxy
淘汰、新 path-family 覆盖与 query efficiency；private IoU 仍只作 selection 后的诊断。如果
hybrid 没有稳定增加结构区分信息，就不应为了“更全局”而保留额外复杂度。

对应的逐阶段实验证据仍保留在旧报告中；2026-08-17—18 的累计验证索引追加在
[reports/IMPLEMENTATION_AND_TEST.md](reports/IMPLEMENTATION_AND_TEST.md#7-2026-08-1718-后续迭代补充)。

## 8. CarryWaterActive：从专家局部扰动扩展到公开全局 rollout 池（2026-08-21）

第 7.6 节记录的是 Obstacle2D 路径在当时的真实边界；本节追加一个任务适配器，**不覆盖**旧实现。
CarryWaterActive 不再从专家内部 waypoint 出发做扰动，而是先依据公开动力学在控制空间独立采样
512 条新轨迹，再由所有 hypothesis 共享的 acquisition policy 选择少量轨迹询问 Oracle。

```text
任务文字 + 18 个公开候选特征
  → GPT / frozen bank 提出 typed atomic 或 composite hypotheses

公开 safe demonstrations ──→ 数值拟合与 structure-audit safety
公开动力学 + 随机控制 ─────→ 512 条未标注、可重放的全局 rollout
                                ↓
                    每个 hypothesis 冻结预测/评分
                                ↓
                    一个共享 acquisition 排序器
                                ↓ 仅被选中的轨迹
                   whole-trajectory membership Oracle
                                ↓ 一次标签供所有 hypothesis 使用
                    refit → qualification → champion/abstain

训练冻结后：private matched pairs → 只做 post-hoc evaluation
```

### 8.1 状态、动作与目标约束

原始 observation 为 12 维：世界坐标位置、到目标的相对位置、三维速度和 roll/pitch/yaw；动作是
`[T-1,6]` 的平移加速度与角速度。公开 `TaskSpec v2` 还注册了 `speed`、
`tilt_from_vertical`、`abs_roll`、`abs_pitch`、`tilt_linf` 和 `progress` 等确定性特征。
NumPy、Torch 特征计算与公开动力学回放使用同一语义。

隐藏目标结构是三个不同类型的同时约束：

1. `target_dz` 的逐时刻 equality band；
2. 三维 `speed` 的逐时刻 upper bound；
3. 与 yaw 无关的 `tilt_from_vertical` upper bound。

编译语义仍是 `composition=any_violation`：任一 clause 违约，整条轨迹就违约。这直接覆盖了导师提出
的“feature A 等式 + feature B 不等式”情形；各 clause 保留独立 ID、关系、时间算子和数值头，
不会被错误压成一个含义不清的单标量 relation。

### 8.2 信息隔离和可辨识性构造

`public/carrywater_active` 只含 64 条已知安全专家轨迹（40/12/12 split）、公开动力学、
TaskSpec 和 512 条未标注候选。候选文件采用字段白名单，若出现 label、clause、group、target 或
其他未注册字段会直接拒绝加载。private seed 不写入公开 artifact；更换 private seed 后，公开目录
逐文件 SHA-256 保持一致。

`private/carrywater_active` 才含 analytic Oracle、期望结构和 1,024 条平衡评测轨迹。评测集由
512 个 safe/unsafe matched pairs 组成，height-only、speed-only、tilt-only、multi 各 128 个
unsafe member。它还刻意构造三种**精确代理碰撞**：

| 错误代理 | 成对样本保持完全相同 | 真标签为何仍相反 |
|---|---|---|
| 世界坐标 `z_position` | 整条世界 z 序列 | requested height/reference 不同，`target_dz` 一侧违约 |
| 单轴速度 | 被代理观察的整条速度分量 | 另一轴改变使三维 speed 一侧越界 |
| `yaw` | 整条 yaw 序列 | roll/pitch 改变总倾角，yaw 本来应自由 |

因此，只学世界高度、单方向速度或 yaw 的模型在普通相关数据上可能看似准确，但不可能同时正确
分类对应的 exact pairs。评测除总体 balanced accuracy 外，新增 worst pair-target balanced
accuracy、exact-pair accuracy、pair-ranking accuracy、minimum clause recall 与分 clause recall。

### 8.3 新 falsifier 的准确边界

CarryWaterActive 的 `PoolHypothesisFalsifier` 是“直接构造新数据”路线的一种可审计实现：

- 轨迹在生成数据集时由公开控制空间和动力学独立 rollout，不读取专家 waypoint；
- 运行时 falsifier 从未标注池中按 frozen model score、跨模型分歧、边界邻近和标准化 novelty 选候选；
- warmup 与 outer-loop 共享同一池且去重，不会把同一候选再次伪装成新 query；
- 多个 hypothesis 只扩大候选评分，不自动各获得一次 Oracle query；一次标签共享给整个 bank；
- Obstacle2D 的欧氏 `radius=0.32` 没有被错误解释为 12 维物理 trust radius。

这解决了“所有新轨迹都只是专家扰动”的覆盖问题，但没有声称得到通用机器人规划器。候选支持仍由
预生成控制分布决定；动力学是公开合成模型，不是接触丰富的真实机械臂动力学；whole-trajectory
Oracle 仍只返回二值标签，不返回真实违约时刻。

### 8.4 已验证结果与尚未完成的论文结论

自动化验证共 115 项，覆盖 12D 数据加载、全部公开/私有轨迹动力学回放、NumPy/Torch 一致性与
有限梯度、公私隔离、三类 exact collisions、多 clause 标签及 SemTraj2D 回归兼容。

冻结-bank 的 13-query smoke 得到 3 safe / 10 violation 标签，完成 7 个模型的 fit/query/freeze，
训练冻结前没有加载 private evaluator。它返回 `selection_status=inconclusive`；provisional
world-z proxy 未通过 qualification。事后私有评测为 balanced accuracy `0.500`、worst pair-target
balanced accuracy `0.500`、exact-pair accuracy `0.000`。对全部七个模型的 diagnostic 进一步显示，
13 次查询也没有把正确 composite 拟合好。因此当前结果证明了数据与评测能暴露 proxy 和欠拟合，
**不证明**本架构已经优于 baseline。

另一次获授权的 GPT semantic-only smoke 接受 4/4 个结构化 hypotheses，`fallback=0`、
`augmentation=0`、Oracle query 为 0；第一条正是 target-height equality band + 3D speed upper
bound + total-tilt upper bound，并明确 yaw free。这表明此前 Qwen 的 schema/fallback 问题在该次
GPT 调用中没有复现，但也只验证语义候选质量，不验证 numeric learner。

下一步正式实验应冻结 semantic bank、公开数据、query budget 和训练超参数，仅使用 5 个预注册
numeric seeds 比较本方法与 demonstration-only、random-query、无 LLM/扁平 all-feature 等基线；
把 query budget 提高到能够检验拟合的规模（先用 `Q=48`，不足时预注册扩展到 `Q=64`），同时报告 abstention、结构恢复、
exact-pair、worst-clause、性能-查询曲线和运行成本。如果 5-seed 结果仍不能稳定得到 qualified
composite champion，应优先修 numeric fitting / acquisition，而不是继续增加数据集机制。
