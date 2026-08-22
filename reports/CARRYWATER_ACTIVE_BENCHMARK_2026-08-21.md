# CarryWaterActive：主动约束辨识基准与首轮验证

日期：2026-08-21

## 结论先行

CarryWaterActive 已经形成一个可运行、可审计的 12 维复合约束基准。它解决了原二维避障实验难以区分“真实约束”和“演示相关代理变量”的问题：新轨迹不是对专家轨迹加噪声，而是从新的任务上下文和运动 profile 出发，生成动作后经过统一的公开动力学 rollout；私有评测又通过成对轨迹制造了三类逐字节完全相同的代理表示碰撞。

当前结果必须分成两部分理解：

- 语义入口成功：一次 GPT semantic smoke 接受 `4/4` 个合规假设，`fallback=0`、`augmentation=0`、无解析错误；其中第一个假设精确表达了 `target_dz` 等式带、三维 `speed` 上界和 `tilt_from_vertical` 上界的 OR 组合。
- 小预算闭环未成功：13 次 Oracle 查询后，系统最终仍把错误的绝对 `world_z` 等式带代理排在首位。该候选被 gate 判为不合格，最终状态是 `inconclusive`；私有 1024 条轨迹上的 balanced accuracy 为 `0.5`，exact-pair accuracy 为 `0.0`。

因此，当前已经证明“数据、接口、LLM 输出链路和失败检测机制可以工作”，但尚未证明“LLM-Modulo CEGIS 的学习性能优于 baseline”。13-query 结果是一个真实且有价值的失败边界，不是论文主结果。

## 1. 基准要回答什么问题

任务描述为：机器人端持开放水杯，依次完成 lift、transport 和 place；跟随当前请求的高度参考，运动保持平缓，杯子保持竖直，但允许绕竖直轴旋转。

隐藏约束由三个异质 clause 通过 `any_violation` 组合：

1. `target_dz` 的逐时刻等式带约束；
2. 三维平移速度模长 `speed` 的逐时刻上界；
3. 与 yaw 无关的 `tilt_from_vertical` 逐时刻上界。

数值边界只存在于 private Oracle 中，本报告不披露。任务描述确实提供了“高度参考、平缓运动、保持竖直”这些定性提示，所以该基准主要测量：

- 能否把自然语言转成正确的 typed multi-clause 结构；
- 能否从有限二值 membership query 中拟合未知数值边界；
- 能否拒绝 `world_z`、单轴速度和 `yaw` 等演示相关代理变量；
- 能否在 Oracle 预算有限时选择真正有辨识力的查询。

它不能被描述为“在完全没有语义提示时盲发现三个 clause”。

## 2. 公开任务与动力学契约

每条轨迹固定为 `T=120`，时间步长 `dt=0.05`。观测为 12 维：

| 索引 | 内容 | 语义 |
|---:|---|---|
| 0–2 | `world_x, world_y, world_z` | 杯子在世界坐标系中的位置 |
| 3–5 | `target_dx, target_dy, target_dz` | 当前任务参考位置减去杯子位置 |
| 6–8 | `vx, vy, vz` | 世界坐标系三维速度 |
| 9–11 | `roll, pitch, yaw` | 杯子姿态 |

动作维度为 6，形状严格是 `[T-1, 6] = [119, 6]`：

```text
[ax, ay, az, omega_roll, omega_pitch, omega_yaw]
```

不存在人为补出的 terminal action。唯一公开动力学为：

```text
v[t+1] = clip(0.98 * v[t] + dt * a[t], -1.2, 1.2)
p[t+1] = p[t] + dt * v[t+1]
angles[t+1] = wrap(angles[t] + dt * omega[t])
```

线加速度动作限制为 `±2.0`，角速度动作限制为 `±1.5`。`reference_xyz[T,3]` 和 `phase_ids[T]` 是公开 sidecar；phase 仅包含 lift、transport、place，不参与隐藏规则切换。

learner 使用 TaskSpec v2、`trajectory_adapter=carrywater_active_v1` 和 18 个已注册可微 feature。三个因果 feature 的 learner-facing 名字分别是 `target_dz`、`speed` 和 `tilt_from_vertical`。

## 3. 数据组成

### 3.1 Public bundle

公开目录为：

```text
llm_modulo_cegis/data/CarryWaterActive/public/carrywater_active/
```

包含：

- `task_spec.json`：严格可由当前 TaskSpec v2 loader 读取；
- `dynamics_spec.json`：观测、动作、sidecar 和动力学递推；
- `expert_trajectories.npz`：64 条已知安全专家轨迹；
- `splits.json`：`train/validation/test = 40/12/12`；其中 `test` 是公开 structure-audit expert split，不是私有安全测试；
- `candidate_trajectories.npz`：512 条无标签候选轨迹；
- `manifest.json`：公开文件哈希、shape 和信息边界说明。

64 条专家覆盖 `8` 个运动方向、`4` 个目标高度和 `2` 种路径风格的笛卡尔积，每个 cell 恰好一条。专家采用对全部允许私有边界都保守安全的 margin，因此公开专家字节不依赖 private seed。

512 条公开候选组成 256 个相邻的 acquisition pair。公开的 `pair_ids/pair_members` 只用于 evidence split 和查询调度；这些 pair 是任意的公共相关性分组，不是已标注的反事实对，也不能被当作安全/违约答案。candidate archive 的字段严格为：

```text
actions, lengths, observations, pair_ids, pair_members,
phase_ids, reference_xyz, trajectory_ids
```

其中不存在 `labels`、clause label、severity、pair role、collision type 或私有数值边界。

### 3.2 Private bundle

私有目录为：

```text
llm_modulo_cegis/data/CarryWaterActive/private/carrywater_active/
```

包含解析 Oracle、`expected_structure.json`、私有 matched evaluation bank 和私有 manifest。评测 bank 有 512 个 pair、1024 条轨迹，标签严格平衡为 `512 safe / 512 violation`。

每个 `pair_target` 有 128 个 pair，即 128 条安全 mate 和 128 条违约轨迹：

| pair target | pair 数 | 违约结构 |
|---|---:|---|
| `height_only` | 128 | 仅高度 clause 违约 |
| `speed_only` | 128 | 仅速度 clause 违约 |
| `tilt_only` | 128 | 仅倾斜 clause 违约 |
| `multi_clause` | 128 | 两个或三个 clause 同时违约 |

`multi_clause` 内四种模式 `(height,speed,tilt)` 为 `(1,1,0)`、`(1,0,1)`、`(0,1,1)`、`(1,1,1)`，每种恰好 32 条违约轨迹。三个 clause 在整个 private bank 中各有 224 条违约轨迹，设计上完全对称。

## 4. 新轨迹不是专家扰动

生成流程是：

```text
新采样的方向/目标高度/phase 时长/初态
        ↓
独立构造 velocity、orientation、reference profile
        ↓
用公开递推的逆关系求 [119,6] actions
        ↓
唯一 public dynamics rollout
        ↓
重放 validator 检查 action、workspace、sidecar 和 dynamics residual
```

公开 candidate 从新的全局任务上下文直接构造，不读取某条专家轨迹作为 anchor，也不对专家 observation 加噪声。private matched pair 会从一个新生成的安全 base 同时构造 safe/unsafe 两个成员，以隔离单个语义因素；这个 base 同样不是公开专家。两个成员都重新求 action 并通过 public rollout，生成器不会直接修改写盘 observation。

runner 的 query diagnostics 仍可能记录一个 `source_expert_id`，因为 acquisition 调度需要为 hypothesis 选择参考上下文；这不意味着候选数据是该专家轨迹的局部扰动。实际被查询的 candidate 是预生成的独立全局 rollout。

## 5. 三类 exact proxy collision

这里的 “exact” 不是数值接近，而是写盘后的 float32 完整时间序列通过 `np.array_equal`。每个 pair 都是一条 safe 和一条 violation，且标签相反。

### 5.1 world height 代理陷阱

128 个 `height_only` pair 的整条 `world_z[T]` 逐字节相同，但 `target_dz[T]` 不同且标签相反。

因此，只使用绝对世界高度的确定性分类器，不可能同时正确判断一个 pair 的两个成员。这直接针对“不同目标高度下，把绝对高度当作相对高度约束”的错误。

### 5.2 单轴速度代理陷阱

128 个 `speed_only` pair 分别轮换保持 `vx`、`vy` 或 `vz` 中一个分量的整条时间序列逐字节相同，同时通过另一个水平分量改变三维速度模长，使标签相反。

因此，任一单轴速度都不能稳定替代 `sqrt(vx²+vy²+vz²)`。这也覆盖不同移动方向和正负速度，避免把某个常见方向误当成速度规则。

### 5.3 yaw 代理陷阱

128 个 `tilt_only` pair 的整条 `yaw[T]` 逐字节相同，但 roll/pitch 共同决定的竖直倾角不同且标签相反。

真实任务允许绕竖直轴旋转；安全约束关心杯子是否倾斜，而不是 yaw 数值。该碰撞使 yaw-only upright proxy 无法在 pair 上取巧。

对于完全相同的代理表示和相反标签，确定性 proxy-only 模型的 row accuracy/balanced accuracy 上限是 `0.5`，而“pair 两个成员都预测正确”的 exact-pair accuracy 上限是 `0.0`。这两个指标不能混为一谈。

## 6. Public/private 信息隔离

当前实现完成了以下代码级隔离：

- learner 配置只把 public task 目录作为 dataset root；
- membership Oracle 对外只返回合法轨迹的一个二值标签；
- post-hoc evaluator 在模型与 result 冻结后才读取 private bank 和 expected structure；
- public JSON/NPZ 使用字段白名单，候选 archive 没有标签或 clause 注释；
- private seed 不写入任何 artifact；
- 固定 public seed、改变 private seed 后，public 目录六个文件的逐文件 SHA-256 全部相同；
- public candidate 的内容、顺序和任意 pair 分组只由 public seed 决定，不用 hidden label 或 private RNG 排序。

这证明了 artifact 和 loader 路径的隔离，但不是对恶意参与者的密码学安全边界。当前 public/private 仍位于同一仓库，生成器源码也不应在正式盲测时交给 learner。论文正式评测需要把 membership capability 放入独立进程或容器，训练侧不挂载 private root，并由评测方保管 sealed private instance。

## 7. 已完成的契约与回归测试

2026-08-21 在项目指定 Python 环境运行：

```text
python -m unittest discover -s llm_modulo_cegis/tests -p "test_*.py"
Ran 115 tests in 26.792s
OK
```

这 115 项包含 legacy core、SemTraj2D、CarryWaterActive 和 runner contract。CarryWaterActive 专项 14 项、SemTraj2D 专项 22 项均通过。专项检查覆盖：

- TaskSpec v2、12D observation 和 transition action shape；
- 64 条 expert 与 512 条 candidate 的 public dynamics replay；
- experts 在解析 Oracle 下全部安全；
- candidate 无标签且 256 个 pair 完整相邻；
- NumPy/Torch 18-feature parity 与有限梯度；
- 1024 个 private label 可由解析 Oracle 逐条重算；
- 512 个 matched pair 全部一 safe 一 violation；
- 三类各 128 个 exact proxy collision；
- 四种 multi-clause pattern 配额；
- public arrays 对 private seed 不变。

单元和集成测试证明实现符合已声明契约，不等价于五个随机种子上的学习性能证据。

## 8. GPT semantic smoke：语义链路成功

产物：

```text
outputs/carrywater_active_gpt_semantic_final/semantic_initial_smoke.json
```

本次使用 OpenAI `gpt-5.6-sol`，只运行 semantic-only，不查询 Oracle。一次交互请求 4 个 hypothesis，结果为：

| 项目 | 结果 |
|---|---:|
| LLM hypotheses accepted | 4/4 |
| fallback | 0 |
| augmentation | 0 |
| parse error | 0 |
| Oracle queries | 0 |

第一个 `h_direct_tracking_speed_tilt` 精确包含：

```text
Always target_dz equality_band
OR Always speed upper_bound
OR Always tilt_from_vertical upper_bound
```

另外三个假设分别检验单轴速度/独立 roll-pitch、mean temporal aggregation 和 terminal-only height 等合理替代解释。它们不是四个都正确；`4/4` 的含义是四个输出都符合 schema 并被接纳，其中至少有一个精确结构候选。这个结果说明此前 Qwen 经常格式失败并 fallback 的问题，在本次 GPT prompt/model 上没有复现。

它尚不能证明：GPT 在不同描述、不同模型版本、不同 temperature 或多个 hidden task instance 上都能稳定给出正确结构。

## 9. 13-query closed-loop smoke：真实失败边界

产物：

```text
outputs/carrywater_active_smoke_final/
```

该 smoke 使用冻结的 7-hypothesis bank：正确 composite、三个 atomic 规则，以及 `world_z`、单轴速度和 yaw 三类代理。它只用于验证 data loader、训练、全局 candidate acquisition、Oracle capability、freeze 和 post-hoc evaluator 能串通。

实际 Oracle 调用为：

- 12 次 warmup/warmup-validation；
- 1 次 outer-loop active acquisition；
- 总计 13 次，标签为 3 safe / 10 violation；
- 唯一 active query 来自 `h_x_velocity_speed_proxy` 的 shortcut candidate，Oracle 返回 violation。

最终结果：

| 项目 | 结果 |
|---|---:|
| frozen champion | `h_world_z_band_proxy` |
| selection status | `inconclusive` |
| champion eligible | false |
| held-out expert safe rate | 0.5833 |
| private trajectory BA | 0.5000 |
| safe accuracy | 0.6172 |
| violation recall | 0.3828 |
| AUROC / AUPRC | 0.5000 / 0.5000 |
| worst pair-target BA | 0.5000 |
| exact-pair accuracy | 0.0000 |
| pair-ranking accuracy | 0.0000 |
| minimum clause recall | 0.3750 |
| exact structure recovery | false |
| qualified exact recovery | false |

不合格原因包括 `violation_recall_below_gate`、`structure_audit_expert_safe_rate_below_gate` 和 `fit_expert_safe_rate_below_gate`，因此 finalization 没有执行。系统没有把错误 proxy 宣布为合格成功，这是 gate 的正确行为；但学习器确实没有在这个预算下找到正确 champion。

失败原因不能简单归咎于 LLM：这个 closed-loop smoke 使用冻结 bank，`llm_interactions=0`，没有测试 outer-loop semantic revision。更直接的限制是 12/13 查询已用于 warmup，真正主动查询只有一条，而且这条是 violation，并未提供能反驳 `world_z` 代理的 safe counterfactual。小训练预算下的 numeric fitting 也没有产生 qualified composite champion。

因此该结果证明：

- 13 次总查询足以跑通系统，但不足以支持性能结论；
- 新 benchmark 能把错误 world-height proxy 明确暴露为 chance-level；
- 当前 safe-query acquisition 尚未在这次 smoke 中获取到有辨识力的安全反事实；
- 不能用“没有 fallback”替代“闭环学对约束”的证据。

## 10. 当前已经证明什么

有直接产物和测试支持的结论是：

- 数据 schema、12D feature adapter、public rollout 和解析 Oracle 可以一致工作；
- 所有写盘轨迹来自统一动力学，公开 candidate 不是专家轨迹扰动；
- public/private artifact 在字段、loader 和随机流上分离；
- 64 条专家已由 Oracle 验证安全，private bank 标签可解析重算；
- 三类代理变量各有 128 个 exact opposite-label collision；
- equality band、upper bound 与三 clause OR 能在现有 IR 中同时表达；
- GPT 在本次 semantic smoke 中产生了 4 个合规、非 fallback 候选，并包含精确结构；
- runner 能在 13 次真实 Oracle 调用下完成训练、冻结和私有 post-hoc 评测；
- gate 会拒绝当前错误且不合格的 world-z champion；
- 当前完整代码回归为 115/115 通过。

## 11. 当前尚未证明什么

目前不能声称：

- LLM-Modulo CEGIS 优于 all-feature joint MLP、无 LLM typed bank 或 one-shot GPT；
- outer-loop hypothesis revision 带来提升，因为 13-query smoke 没有 LLM revision；
- active acquisition 已稳定找到 safe counterfactual；本次唯一 active query 是 violation；
- 正确三 clause 结构能被 numeric fitting 稳定选为 qualified champion；
- 13 次 Oracle 足以学习本任务；
- 单个 private instance 和单个 numeric seed 能代表总体性能；
- 合成动力学结果可以直接外推到真实水杯操作、接触动力学或传感噪声；
- 当前同仓库 public/private 布局构成恶意环境下的安全隔离；
- GPT 的 `4/4` 合规率能跨 prompt、模型版本和任务稳定复现。

## 12. 下一步正式评测：5 个 numeric seeds 与更大 Q

### 12.1 立即执行的固定协议

先固定当前 dataset instance、公开 candidate 顺序、私有 bank、semantic bank 和所有 gate，只改变训练/初始化随机性。使用预注册 numeric seeds：

```text
[7, 19, 37, 73, 109]
```

Oracle 预算统一按真实调用总数 `Q` 计，warmup 也算入预算。第一阶段把主预算提高到 `Q=48`，并保存以下前缀 checkpoint：

```text
Q = 13, 20, 32, 48
```

其中 `Q=13` 与当前 smoke 对齐；若到 `Q=48` 仍未出现性能饱和，再预注册扩展到 `Q=64`，不要在看到结果后临时选择有利预算。

每个 numeric seed 必须复用同一批候选和相同查询上限。LLM 调用次数、optimizer launches、训练 epochs 和最终 Oracle 次数分别报告，不能只对齐 Oracle 而让某个方法获得更多隐性优化预算。

### 12.2 最小主表 baselines

在相同 Oracle 预算下至少比较：

1. 完整 GPT LLM-Modulo CEGIS，允许 outer-loop revision；
2. one-shot GPT：冻结初始 4-hypothesis bank，禁止 revision；
3. 无 LLM typed-bank CEGIS：使用预注册 atomic/composite/proxy bank；
4. all-feature joint MLP；
5. oracle-structure numeric ceiling：只给正确结构，不给数值边界。

另报告三个 proxy-only sanity baselines：`world_z`、单轴速度和 `yaw`。它们不是竞争方法，而是验证 exact collision 指标确实给出预期 ceiling。

### 12.3 主指标

不能只报告平均 trajectory accuracy。主表与 query curve 应包括：

- private trajectory balanced accuracy 与 AUPRC；
- safe accuracy、violation recall；
- worst pair-target balanced accuracy；
- exact-pair accuracy 与 pair-ranking accuracy；
- height/speed/tilt 每 clause recall 及 minimum clause recall；
- exact structure recovery 与 qualified exact structure recovery；
- held-out expert safe rate；
- 首次出现 qualified champion 所需的 Oracle queries；
- performance–query AUC；
- `inconclusive` 率和 erroneous-qualified-champion 率。

五个 numeric seeds 应逐个列出，并汇报均值、标准差、中位数和最差 seed。numeric seeds 是同一 task instance 上的优化重复，不应被当成五个独立任务。若要形成论文级跨实例结论，下一阶段还需生成五个独立 hidden dataset/private-rule instances，并以 task instance 为统计单位。

### 12.4 可以形成论文结论的最低证据

在完成上述协议前，不应写“架构优于 baseline”。正式结论至少应满足：

- 完整方法在预注册主指标和 performance–query curve 上稳定优于 all-feature MLP 与无 LLM bank，而非只赢一个 seed；
- 提升来自更多 qualified exact recovery 或更好的 proxy-pair 指标，而不是仅提高 ID accuracy；
- outer-loop arm 相对 one-shot GPT 有可复现增益，或诚实报告 revision 没有贡献；
- safe-query acquisition 在多个 seed 中实际得到 safe counterfactual，并降低错误 proxy 的存活时间；
- 私有评测始终只在 freeze 后加载，且失败 seed 与 `inconclusive` 也完整报告。

## 13. 关键入口与可复核产物

- 生成器：`llm_modulo_cegis/generate_carrywater_active.py`
- 数据说明：`llm_modulo_cegis/data/CarryWaterActive/README.md`
- public task：`llm_modulo_cegis/data/CarryWaterActive/public/carrywater_active/`
- private evaluator input：`llm_modulo_cegis/data/CarryWaterActive/private/carrywater_active/`
- task adapter/validator：`llm_modulo_cegis/src/llm_modulo_cegis/carrywater_active.py`
- runner：`llm_modulo_cegis/run_carrywater_active.py`
- post-hoc evaluator：`llm_modulo_cegis/evaluate_carrywater_active.py`
- CarryWaterActive tests：`llm_modulo_cegis/tests/test_carrywater_active.py`
- GPT semantic smoke：`outputs/carrywater_active_gpt_semantic_final/`
- 13-query closed-loop smoke：`outputs/carrywater_active_smoke_final/`

当前最重要的下一步不是继续扩充数据字段，也不是增加更多 heuristic hypothesis，而是按固定五个 numeric seeds 把 `Q` 提高到 48，观察正确 composite 是否能稳定成为 qualified champion，以及 safe counterfactual acquisition 是否真正淘汰三个精心设计的代理规则。
