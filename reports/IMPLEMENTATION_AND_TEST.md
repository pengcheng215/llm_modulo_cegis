# 实现与测试记录

日期：2026-08-13

> 本文前 1–6 节保留 2026-08-13 当时的基线、`9/9` 测试数和 smoke 结果，
> 不用后续实验覆盖历史记录。2026-08-17—18 的累计架构与验证状态追加在第 7 节；
> 2026-08-21 的轨迹生成能力边界与下一实验追加在第 8 节。

## 验证范围

数据：`LLMConstraint-master/data/Obstacle2D`。训练使用 9 条 train expert；6 条 validation/test expert 只参与评估。隐藏圆形几何只存在于 `private_evaluation/ground_truth.json`。

## 1. 单元测试

命令：

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe -m unittest discover -s llm_modulo_cegis\tests -v
```

结果：9/9 通过；新增覆盖截断 JSON 数组中完整对象的安全恢复，其余覆盖项如下：

- 可微派生特征；
- 多类假设编译；
- 不合法多变量阈值关系拒绝；
- `linear/MLP` 模型族确实编译成不同计算图；
- 假设替换和版本状态；
- trajectory-membership 神经训练；
- 证据策略保留、查询和淘汰；
- membership view 不暴露评估几何。

所有 Python 文件另外通过 `py_compile`。

## 2. 最终 smoke 闭环

配置：`configs/obstacle_avoid_smoke.yaml`。

结果：

```text
warmup queries=20, safe=4, violation=16
outer round 1 semantic champion=h_planar_joint
outer round 2 semantic champion=h_planar_joint
active after revision: y-only, planar-joint, planar-independent
retired: x-only, speed
total Oracle queries=28
```

快速配置最终 IoU 约 `0.108`。它只训练 35 epoch、每假设每轮查询一次，因此只能作为控制流测试，不能用来主张边界性能。

## 3. 强数值配置验证

使用正式模型宽度、3-member ensemble、140 epoch 和多次 Falsifier restart，运行一个完整外层轮次：

```text
semantic champion: h_planar_joint
Oracle queries: 34
boundary IoU: 0.5925
false-safe rate: 0.0593
false-unsafe rate: 0.0481
grid accuracy: 0.9510
```

这表明多假设外层没有破坏原神经边界学习能力。该实验只有一个外层轮次，不能证明闭环优于 one-shot；完整论文实验必须运行多 seed、多轮和消融。

## 4. Qwen smoke

环境：本机 Python 3.12，`torch 2.4.1+cu118`，`transformers 4.57.3`，本地 `Qwen2.5-1.5B-Instruct`。

最终严格编译版本中，Qwen 成功被调用两次：

1. 初始阶段输出两个 `(x,y)` 候选；
2. revision 阶段读到真实 EvidenceCompiler 报告并尝试输出动作。

第一次输出包含不合理的二维 `lower_bound/upper_bound`，被严格编译器以 `upper_bound/lower_bound require one scalar variable` 拒绝，并回退到覆盖性候选 bank。第二次输出没有遵守完整 revision action schema，还提出未允许的算子和模型名，因此同样被记录为解析失败，确定性 evidence policy 接管。本次 `llm_fallbacks=2`；所有原始文本和错误保存在 `semantic_interactions.json`。

结论不是“Qwen 已可靠完成闭环”，而是：

- Qwen 已真正接收闭环证据；
- 其动作会在合法时改变下一轮 bank 和 Falsifier；
- 小型 1.5B 模型的结构化修订可靠性仍不足；
- 系统不会静默接受幻觉或非法数值。

## 5. Smoke 消融

`run_ablation.py` 比较 semantic closed-loop 与 frozen one-shot bank。快速实验中：

- closed-loop 选择语义正确的 `(x,y)` joint forbidden region，并淘汰两个假设；
- frozen bank 按纯训练分数选择 `y-only` forbidden band；
- frozen 的 smoke IoU 反而更高（约 `0.230` vs `0.108`），说明低训练预算下“结构选对”不保证神经边界已经拟合好。

这个结果必须如实保留。论文需要用正式训练预算、多 seed、相同 Oracle budget 比较结构恢复率和达到目标 IoU 的查询数，而不能只引用这次 smoke。

## 6. 下一阶段实验要求

1. 为 ObstacleAvoid 增加与任务无关的干扰观测维度；
2. 加入错误初始 bank 恢复实验；
3. 比较 Qwen closed-loop、Qwen one-shot、枚举、随机和 oracle-structure；
4. 至少 5–10 个随机 seed；
5. 报告结构恢复、Oracle query efficiency、IoU、false-safe 和跨起终点迁移；
6. 单独统计 Qwen schema success、fallback rate 和每种修订动作的实际收益。

## 7. 2026-08-17—18 后续迭代补充

本节是在原测试记录上的追加，不把多个不同代码快照、不同 seed 或 frozen/live semantic
设置合并成一次“最终实验”。尤其需要区分：GPT end-to-end 验证了真实 structured-output
语义调用；后续大多数 falsifier、acquisition、numeric-fitting 与 selector 消融使用冻结的
GPT round-0 hypothesis bank、`freeze_revisions=true` 和 `0` 次 LLM，以隔离单个机制。

### 7.1 从原 smoke 到当前架构的验证时间线

| 阶段 | 当时发现的问题 | 追加实现 | 主要结论/证据 |
|---|---|---|---|
| GPT structured smoke 与 gated diagnostic | Qwen 经常不符合 schema；弱模型 fallback 掩盖了真实语义质量 | GPT Responses strict Structured Outputs、backend error 与 semantic fallback 分离、qualification gates | GPT 可以生成并执行多样 typed hypotheses；不再把网络/API 失败伪装成成功 fallback |
| integrated live GPT | 需要确认语义 backend 真正参与，而非 frozen/canonical bank 机制回放 | live GPT initial proposal、可执行 revision 与 intervention-localized weak witness | `0` fallback、正确 spatial champion、IoU `0.8146`、held-out expert-safe `1.0`、`26` Oracle、`3` 次 GPT interaction；这是 live 路径证据，不能与后续 frozen-bank 消融合成一次运行 |
| proxy guard 与 transactional finalization | 高-y/progress/terminal proxy 可能靠相关性过门；重训后模型可能破坏专家安全 | nested/progress/dynamics gates、pure-last representation collision、独立 calibration/selection、失败回滚 | endpoint-preserving safe/violation 冲突会淘汰 pure terminal；finalization 不再无条件覆盖 incumbent |
| safe-query acquisition | 旧 batch 每轮最多盲留一条 likely-safe，失败后不继续；holdout labels 还被误算进平衡 | 只数 trainable labels、逐 answer 重算 safe/violation deficit、两类 balance query 合计两槽上限；safe 侧要求 causal crossing eligibility 与 clause-aware dedup | safe-query 能重试但不能占满预算；全模型都判 safe 的候选不能靠 quota 获得优先权 |
| hard-margin falsifier | 固定 `0.08` 和 smooth `epsilon` 目标经常没有真正跨过 source model；composite 可跨错 clause | `threshold+0.05` generation certificate、model-safe anchor、有效 checkpoint、非单调 scan+refine、`threshold+0.02` query certificate、显式 target clause | 未达到 full/source/target-clause crossing 的候选不能查询 Oracle；多 radius/restart 仍只形成一个待选候选 |
| single radius vs ladder 五种子 | ladder 增加 reachability，但不确定是否值得额外计算 | paired seeds、固定 bank、相同 Oracle budget，比较 single `0.32` 与 `0.04/0.08/0.16/0.32` | 两臂都没有稳定提高 Oracle-safe yield或降低扰动；ladder约 `3.26x` false-unsafe optimizer launches，故默认 single `0.32` |
| numeric fitting 三臂五种子 | bootstrap member 漏掉大量已付费 query；all-state MIL 把共享安全端点当 violation witness | full trainable buffer + source-anchor changed-state pooling | classic bootstrap、full all-state、full + mask 的 qualified spatial champion 为 `2/5`、`3/5`、`4/5`；mask 下 spatial 本身 `5/5` eligible，fit/audit expert-safe 均为 `1.0` |
| linear-max convex-support gate | 旧 seed 109 中已合格 spatial 被同样合格的 affine 依靠 yield/complexity/capacity 项反超 | 两个独立安全 anchor 上的严格 convex-support order certificate；audit/enforce treatment | 旧 artifact replay 可修正 seed 109，但只作机制证据；新五种子正式两臂均为 qualified spatial `5/5`，只能支持 sound/no-regression，不能声称性能提升 |

更完整的架构对照已追加到 [../ARCHITECTURE.md](../ARCHITECTURE.md#7-架构演进记录2026-08-18保留原设计)。

### 7.2 当前闭环与信息边界

当前一轮的可执行顺序是：

```text
fit active models on trainable records
  → freeze held-out/pre-query predictions
  → generate clause-targeted candidate pools
      ↳ false-unsafe additionally requires generation/refined hard certificates
  → sequential shared-budget acquisition
  → whole-trajectory membership Oracle
  → refit on the complete trainable buffer
  → prequential evidence + expert qualification + structural gates
  → eligible champion, or provisional ID with selection_status=inconclusive
  → semantic revision only when a later round exists
```

`warmup_validation` 和 `final_calibration` 不进入 gradient fitting；同一个 warmup pair family
不会跨 train/validation/calibration；另有独立的 known-safe structure-audit experts。
每次 Oracle answer 都保留所有 active hypotheses 的冻结预测，因此一条付费标签可以同时支持
或反驳多个结构。evaluation-only geometry、IoU、false-safe 和 false-unsafe 会在每轮 semantic
actions 已决定后另行计算，但不进入 EvidenceCompiler、LLM prompt、revision、acquisition 或
champion score；这是 controller 内的数据流隔离，不是 private evaluator 的进程级物理隔离。

用户信息假设仍是三部分：任务描述、安全 demonstrations、以及对主动选出的少量新整轨迹
作二值 feasibility 判断。系统没有把这一假设改写成 demonstration-only learning，也没有
向学习器暴露障碍中心、半径或真实 violation timestep。

### 7.3 当前回归与正式 artifact 快照

2026-08-18 在当前工作树重新执行：

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe -m unittest discover -s llm_modulo_cegis\tests -v
```

结果：`79/79 PASS`；`run_obstacle_avoid.py`、正式 sweep runner、`evidence.py` 与 `types.py`
同时通过 `py_compile`。这不覆盖第 1 节的历史 `9/9`，而是新的回归快照。

最新 linear-max gate 正式比较使用新 seeds
`[1007, 1019, 1037, 1073, 1109]`，每个 seed 两臂的完整 26-query
`observations/actions/labels/outer_rounds` 哈希完全一致。每个 run 均为：

- `26` 次 Oracle；
- `0` 次 LLM（冻结 bank 机制实验）；
- `18` 条 trainable query records；
- warmup 共 `14` 条，其中 train `4S/2V`、validation `2S/2V`、calibration `2S/2V`；
- full-buffer coverage `1.0`；
- source-anchor mask 实际激活；
- private geometry 未进入采用规则。

正式结果：

| arm | spatial eligible | qualified spatial champion | affine gate triggered | affine gate applied | public safe / violation range |
|---|---:|---:|---:|---:|---:|
| audit-only | 5/5 | 5/5 | 5/5 | 0/5 | safe `0.833–0.857`; violation `0.556–0.667` |
| enforced | 5/5 | 5/5 | 5/5 | 5/5 | safe `0.833–0.857`; violation `0.556–0.667` |

audit-only 在这批新 seeds 上本来就是 spatial `5/5`，所以 paired rescue 条件是空条件成立。
默认启用 enforced 的理由是结构定理 sound 且未观察到公开门槛回退，而不是本实验显示了
champion rate 或 private IoU 提升。

这些正式 numeric/gate artifacts 使用 diagnostic profile、冻结 GPT round-0 bank、
`freeze_revisions=true` 和 `0` 次 LLM。CLI 默认仍是 Qwen profile，普通 GPT/Qwen YAML 也不一定
启用相同的 finalization、rolling window、audit 数量和 gates；因此这里验证的是指定 profile
中的机制，而不是“所有默认命令”或 live semantic outer-loop 的五种子稳定性。

### 7.4 阶段报告索引

- GPT 真实调用与最初问题：[`GPT_SMOKE_2026-08-17.md`](GPT_SMOKE_2026-08-17.md)
- qualification/proxy/finalization：[`GPT_PROXY_GUARDED_IMPLEMENTATION_2026-08-17.md`](GPT_PROXY_GUARDED_IMPLEMENTATION_2026-08-17.md)
- integrated live-GPT 结果：[`GPT_INTEGRATED_FINAL_2026-08-17.md`](GPT_INTEGRATED_FINAL_2026-08-17.md)
- safe-query 历史诊断：[`SAFE_QUERY_ACQUISITION_2026-08-17.md`](SAFE_QUERY_ACQUISITION_2026-08-17.md)
- hard-margin 与多约束修复：[`FALSIFIER_HARD_MARGIN_LADDER_2026-08-17.md`](FALSIFIER_HARD_MARGIN_LADDER_2026-08-17.md)
- single-radius 五种子选择：[`FALSIFIER_MULTISEED_SINGLE_VS_LADDER_2026-08-17.md`](FALSIFIER_MULTISEED_SINGLE_VS_LADDER_2026-08-17.md)
- full-buffer + source-anchor mask：[`NUMERIC_FITTING_STABILITY_2026-08-17.md`](NUMERIC_FITTING_STABILITY_2026-08-17.md)
- exact linear-max 结构门：[`LINEAR_MAX_SUPPORT_GATE_2026-08-17.md`](LINEAR_MAX_SUPPORT_GATE_2026-08-17.md)

### 7.5 更新后的下一步

原第 6 节列出的 distractor dimensions、错误初始 bank、更多 baselines 和跨任务验证仍未完成。
在当前 Obstacle2D/frozen-bank 机制内，最值得继续的已不是扩大 safe quota、增加 radius 或
堆更多 ensemble members，而是：

1. 对“多个仍可表达且都合格”的结构生成共享 endpoint、共享 anchor 的最大区分查询；
2. 在含 distractor features、上下两侧绕行和 heterogeneous composite constraints 的新任务上复验；
3. 用新的 semantic seeds 运行 live GPT multi-round，而不是只固定已观察过的 round-0 bank；
4. 报告结构恢复率、qualified-set 大小和 performance-versus-Oracle-query 曲线，而不只报告最终 IoU；
5. 若 Oracle 可能含噪声，把两-anchor hard gate 扩展成显式噪声模型，而不是降低几何证书的严格性。

## 8. 2026-08-21 轨迹生成范围审查与下一实验（非新结果）

本节是代码能力边界审查和实验计划，不修改第 7 节的 `79/79` 回归与正式 artifacts，也不报告
尚未运行的性能。当前所有会进入 Oracle acquisition 的新候选仍以一条 expert trajectory 为
anchor；`global acquisition` 指共享预算下的全局**选择**，不是全局轨迹**生成**。

| 候选来源 | 当前状态 | 实际生成范围 |
|---|---|---|
| warmup pair | implemented | 对同一专家 detour 做 `toward_chord` / `continue_detour` 成对变形，固定起终点 |
| shortcut / false-safe / boundary / feature-stress | implemented | 从 expert/chord 插值和多方向 smooth-basis restart 初始化，带 expert-distance、平滑、workspace 与 step 正则；没有统一的 `0.32` hard radius |
| false-unsafe | implemented | 从 source model 明确接受的 known-safe expert 出发，受逐时刻 `0.32` trust radius 约束，并经过 generation hard margin、target clause、边界细化与 query gate |
| direct global spline | **not implemented** | 尚无不使用 expert 内部 waypoint 的控制点/样条生成器，也无对应测试或 artifact |
| local/global hybrid | **not implemented** | 尚无共享候选池中的 local-vs-global 固定预算比较 |

当前局部生成适合寻找专家附近的可信反事实，也为 false-unsafe 提供已知安全侧；它不能保证探索
新的上下绕行方式或 homotopy class。多方向 basis restart 只能扩大局部初始化，不能作为全局覆盖
保证。hard certificate 证明的也只是“模型和指定 clause 已 crossing”，不是轨迹真实安全、真实
碰撞或满足机器人动力学。

下一实验拟比较三个显式 treatment：

1. `local-only`：保持当前 expert-anchored falsifier；
2. `global-only`：只用 public TaskSpec 生成平滑控制点/样条候选；
3. `hybrid`：两类候选进入同一 acquisition，并共享完全相同的 Oracle budget。

三臂必须固定 bank、seeds、训练数据角色、候选计算预算和 Oracle 上限。global 分支只使用公开
start/goal/horizon/workspace 与冻结模型，不读取 private obstacle geometry，不自行赋标签，也不
占用额外 safe quota。采用依据是公开结构恢复、proxy 淘汰、路径族覆盖和 query efficiency；若
hybrid 没有稳定收益，应继续采用更简单的 local-only。
