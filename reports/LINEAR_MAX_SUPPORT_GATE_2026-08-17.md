# Linear-max convex-support gate: from seed 109 to a sound structural rejection

## 结论

这一阶段没有继续调整 numeric fitting、radius、safe quota，也没有事后修改
complexity/capacity penalty。改动针对的是一个更窄、但可以严格证明的结构错误：当一个
Oracle-violation 轨迹在所选特征上的全部采样点，都落在与它关联的已知安全 expert
轨迹的凸包内时，`linear + max` 模型不可能用同一个阈值把安全 anchor 接受、同时把该
candidate 拒绝。

系统现在对这种矛盾生成公共证书。两个不同安全 anchor 都产生证书后：

- audit-only 模式只记录证书；
- enforced 模式把对应的 atomic joint linear-max hypothesis 从 champion 候选中排除；
- 原有 `selection_score` 与 `query_priority` 不变；
- MLP 等未被定理覆盖的模型不受影响。

新的五 numeric-seed、两臂闭环实验中，audit-only 和 enforced 都得到 `5/5`
qualified spatial champions；enforced 在 `5/5` runs 都触发并应用结构门，且所有公开
安全/召回门槛都通过。因此默认采用 enforced，理由是它是一个没有观察到回退的可靠
结构防护。两臂的 champion 计数相同，**不能声称它提高了性能**。

## 旧 seed 109 暴露的不是 fitting 失败

在上一阶段的 `full_source_anchor_mask` arm 中，seed 109 的非线性 spatial 模型已经
通过全部 qualification gates：

| hypothesis | balanced accuracy | safe accuracy | violation recall | audit/fit expert safe | intervention yield | selection score |
|---|---:|---:|---:|---:|---:|---:|
| `h_spatial_exclusion` (MLP) | 0.706 | 0.857 | 0.556 | 1.000 / 1.000 | 0.667 | 0.499 |
| `h_affine_spatial` (linear) | 0.635 | 0.714 | 0.556 | 1.000 / 1.000 | 1.000 | 0.552 |

spatial 的公开预测证据更好，但原标量排序还混入 intervention yield、结构复杂度和参数
量先验；这些项让同时合格的 affine 模型得分更高。直接减小复杂度惩罚或为 spatial
加分，会是针对一个 seed 的事后调参，而且无法说明 affine 在结构上为什么不成立。

关键的新观察来自 Oracle 标签与安全 demonstration 的几何支撑关系，而不是 private
障碍物几何：多条被 Oracle 判为 violation 的轨迹，其 `(x, y)` 采样点完全位于对应
安全 expert 的 `(x, y)` 凸包内。对 `linear + max`，这是与参数无关的矛盾。

## 凸包支撑序定理

设安全 anchor 的所选特征点集为 `A={a_i}`，Oracle-violation candidate 的点集为
`V={v_j}`。atomic linear head 的状态分数为：

```text
g(z) = w^T z + b
G(T) = max_{z in T} g(z)
```

如果 `V` 的每个点都在 `conv(A)` 内，则每个 `v` 都可写为
`v = sum_i lambda_i a_i`，其中 `lambda_i >= 0` 且 `sum_i lambda_i = 1`。因为 `g`
是 affine：

```text
g(v) = sum_i lambda_i g(a_i) <= max_{a in A} g(a)
```

所以：

```text
max_{v in V} g(v) <= max_{a in A} g(a)
```

这个不等式对任意 `w,b` 都成立。若同一个阈值把 `A` 判为 safe，就不可能把 `V` 判为
violation。输入的 affine normalization 保持凸组合关系，因此不改变结论。

这不是“当前这次优化没拟合好”，而是该结构族对这对标签不可实现。它也不需要知道
障碍物中心、半径、状态级碰撞时刻或 private IoU。

## 严格 one-sided 实现

硬门只覆盖同时满足以下条件的 hypothesis：

- 只有一个 atomic clause；
- `coupling=joint`；
- `relation=forbidden_region`；
- `model_family=linear`；
- `temporal_operator=max`；
- 选择 1 或 2 个特征。

证书只读取 Oracle-violation query、query 中的 `expert_id`，以及该 ID 唯一解析到的已知
安全 expert。`final_calibration` 记录被排除。对每个可解析 pair，系统在归一化特征空间
检查 candidate 的每个采样点是否位于 anchor 的凸包内，并统计：

```text
linear_max_support_pair_count
linear_max_support_contradiction_count
linear_max_support_distinct_anchor_count
linear_max_support_unresolved_pair_count
linear_max_support_gate_triggered
linear_max_support_gate_applied
```

默认至少需要两个不同的 contradictory anchors，避免一条可能误标轨迹直接造成硬淘汰。
`linear_max_support_tolerance=1e-5` **只用于去重近重复的 anchor 点**；1D 区间、退化线段
和 2D 多边形的包含判断都不会把凸包向外扩张。靠近但位于凸包外的 candidate 会返回
“没有证书”，而不是被近似容差误判为结构矛盾。

这是严格的 one-sided test：

- 包含成立时，可以否定被定理覆盖的 linear-max 结构；
- 包含不成立、anchor 无法解析、维度大于 2，或结构类型不受支持时，只能保持未知并
  fail open；
- 证书不能反过来证明 MLP 正确，也不改变 Oracle 标签或连续时间语义；
- 被结构门淘汰的简单模型不会再通过 nested-minimality gate 淘汰更复杂模型。

## 开发回放与正式实验不是同一类证据

开发阶段先在上一轮已经完成的 15 个 numeric-fitting artifacts 上离线重算资格，作为
机制检查：

| historical arm | 原 qualified spatial champion | 应用证书后的离线结果 |
|---|---:|---:|
| classic bootstrap | 2/5 | 2/5 |
| full-buffer all-state | 3/5 | 4/5 |
| full-buffer + source-anchor mask | 4/5 | 5/5 |

其中旧 seed 109 从 affine champion 变为已经合格的 spatial champion。这个回放使用了
曾经观察过的旧 artifacts，只说明证书确实命中原问题；它不是新种子上的泛化结果，也
没有被用来声称性能提升。

正式比较在运行前冻结了新的 numeric seeds
`[1007, 1019, 1037, 1073, 1109]` 和两臂规则：

- `audit_only`：计算同一证书，但不改变 champion eligibility；
- `enforced`：证书触发后排除 certified affine hypothesis。

除此以外，两臂固定同一个 GPT hypothesis bank、hard single-`0.32` falsifier、safe-query
acquisition、完整 trainable buffer、source-anchor changed-state pooling、3 个 outer
rounds 和 CPU 单线程设置。每个 run 都使用 26 次 Oracle、0 次 LLM interaction，其中
18 条 query record 可用于梯度训练。

## 正式五种子两臂结果

| arm | spatial eligible | qualified spatial champion | qualified final champion | affine gate triggered | affine gate applied | certified-affine champion | median public S/V |
|---|---:|---:|---:|---:|---:|---:|---:|
| audit-only | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 0.857 / 0.600 |
| enforced | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 0/5 | 0.857 / 0.600 |

五个 seed 的 affine 证书都来自 4 个不同安全 anchors；最终 pair 数为 15 或 16，包含
矛盾的 pair 数为 11 或 12，无法解析的 pair 为 0。enforced 的每个 champion 都通过
运行前冻结的公开门槛：safe accuracy `>=0.60`、violation recall `>=0.55`、audit/fit
expert safe rate `>=0.90`；两臂之间没有任何公开门槛从 pass 变为 fail。

artifact validation 为 `PASS`。同一 seed 的两臂不仅 warmup 相同，完整有序的 26-query
轨迹、动作、标签和 outer-round arrays 也逐项哈希一致：

| seed | paired full-slate SHA256 |
|---:|---|
| 1007 | `ecc5aa9bbf5e780570124b7e0c45e81edbabf0651cc6f5ecf64589e37e637535` |
| 1019 | `0203b8f6cc144a7ff6f8caa95fc9b2cd1a61cae53a3090f568693a9d44f640d6` |
| 1037 | `d508b036ff005694e86d865477b6b73f454028f458420e7249ead443970311c8` |
| 1073 | `896e3ab6e03dc2ba519c7557cb46df2b8166f0c723cc0c3c6bd7166be05ea5ae` |
| 1109 | `574af58cb173a0f1284202dc0cdfbe4371e5357838792fb134f1111dca149471` |

因此 treatment isolation 也比一般的 paired-seed 比较更强：resolved configs 只在
`output_dir` 和 `evidence.linear_max_support_gate_enforced` 上不同，本次执行中 gate 没有
改变 query slate。private geometry 只在 champion 已选定后产生 evaluation-only IoU，
不在 decision allowlist 中。

新的 audit-only seeds 本来就已经是 spatial `5/5`，所以“若 baseline 选中 certified
affine，enforced 必须 rescue”的条件在这批正式 runs 中是 vacuous：没有 baseline
affine champion 可供 rescue。公开率也全部为零差异。这支持“启用一个 sound guard 没有观察到回退”，不支持
“gate 使 champion rate 或边界质量提高”。

## 采用决定与边界

当前默认启用 enforced gate；audit-only CLI 保留用于消融。这个决定的含义仅是：不再
允许一个已经被公共凸支撑序证书严格否定的 affine-max 模型成为 champion。

仍需保留以下限制：

1. 五个固定 seeds 足够做工程默认选择，不构成统计显著性结论。
2. 结论目前只覆盖这个冻结 hypothesis bank 和 Obstacle2D；其他任务需要重复验证。
3. 证书是充分而非必要条件。凸包外的 violation 轨迹不表示 linear 模型正确。
4. 当前 hard gate 只支持 atomic 1D/2D joint linear-max；composite、`mean/last`、MLP
   和更高维结构均保持未知。
5. 凸包只由离散采样状态构成，不提供两采样点之间的连续时间保证。
6. 两-anchor 要求减小单点误标风险，但不能证明 Oracle 永不出错。
7. 它没有解决所有“多个都合格”的一般排序问题；当剩余结构都没有可证明矛盾时，仍需
   更有针对性的结构区分查询，而不是继续调一个全局 complexity penalty。

## 可复现产物

- 运行前冻结的计划：`configs/linear_max_support_gate_multiseed_plan.yaml`
- sweep runner：`run_linear_max_support_gate_multiseed.py`
- 完整 10-run artifacts：`outputs/linear_max_support_gate_multiseed_5seed_v1/`
- resolved plan：`outputs/linear_max_support_gate_multiseed_5seed_v1/experiment_plan_resolved.yaml`
- 公共决策汇总：`outputs/linear_max_support_gate_multiseed_5seed_v1/linear_max_support_gate_multiseed_summary.json`
- 每 run 指标：`outputs/linear_max_support_gate_multiseed_5seed_v1/linear_max_support_gate_per_run_metrics.csv`
- 自动短报告：`outputs/linear_max_support_gate_multiseed_5seed_v1/LINEAR_MAX_SUPPORT_GATE_MULTISEED_REPORT.md`
- 每个 run 的代码/输入指纹：对应目录中的 `implementation_manifest.json`

冻结 artifact 中记录的哈希：

```text
plan SHA256   = 53bcff42ef2fd9f4941e2755f123f99dd44432dcb355994bf16044394c82ac51
runner SHA256 = f421343a4a5f0c1d71ebb93fd1d0937befd0c0e6054e08518aefa556177a4291
```

复现实验：

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe llm_modulo_cegis\run_linear_max_support_gate_multiseed.py `
  --plan llm_modulo_cegis\configs\linear_max_support_gate_multiseed_plan.yaml `
  --output-root llm_modulo_cegis\outputs\linear_max_support_gate_multiseed_5seed_rerun `
  --max-workers 3
```

只重新校验并汇总已有 artifacts 时可加 `--summarize-only`；这不会重新训练模型。
