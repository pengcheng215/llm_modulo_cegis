# 实现与测试记录

日期：2026-08-13

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
