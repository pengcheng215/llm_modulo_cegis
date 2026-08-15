# ObstacleAvoid：Qwen 驱动 LLM-Modulo CEGIS 测试

日期：2026-08-13

## 1. 测试目标

本测试验证本地 Qwen2.5-1.5B 是否真实进入语义—数值 CEGIS 闭环，并在 `Obstacle2D` 专家轨迹和仅返回整轨迹安全标签的 Oracle 下选择、训练二维障碍约束。隐藏圆形几何、状态级违规标签和 IoU 只供最终评估，不能进入 LLM prompt、EvidenceCompiler 或训练标签。

## 2. 配置与命令

- 数据集：`LLMConstraint-master/data/Obstacle2D`
- 专家划分：9 条训练轨迹，6 条留出轨迹
- LLM：本地 `Qwen2.5-1.5B-Instruct`
- 神经约束：3-member ensemble，隐藏层 `[32, 32]`
- 训练：每轮 90 epochs，共 2 个外层语义轮次
- Oracle 预算：实际 32 次整轨迹查询
- 完整配置：`../configs/obstacle_avoid_qwen_test.yaml`

运行命令：

```powershell
py -3.12 llm_modulo_cegis\run_obstacle_avoid.py `
  --config llm_modulo_cegis\configs\obstacle_avoid_qwen_test.yaml `
  --overwrite
```

前置回归：

```powershell
C:\Users\rpc21\miniconda3\envs\pucl\python.exe -m unittest discover -s llm_modulo_cegis\tests -v
```

结果为 9/9 通过，全部 Python 文件通过 `py_compile`。新增测试确认：Qwen 输出末尾被截断时，只恢复已经完整闭合的数组对象，残缺对象不会被执行。

## 3. Qwen 在闭环中的实际作用

总计 3 次 Qwen 交互，`llm_fallbacks=1`。

初始生成中，Qwen 给出的一个完整合法候选 `h3` 被接收：它假设 `(x, y, vx, vy)` 联合构成 forbidden region。三个多变量 `lower_bound/upper_bound` 候选因类型不合法被逐项拒绝。由于合法候选不足，系统补充了 x-only、y-only 和 `(x,y)` joint 三个覆盖性候选；因此最终正确的二维候选不是 Qwen 独立首发，而是安全补全得到的。

两轮 revision 中，Qwen 均返回了合法的：

```text
retain_and_query(h_planar_joint)
```

两次均未触发整轮回退。其余字段缺失的 intervention/change_variables 对象被逐项拒绝，不影响合法 retain 动作。首轮动作决定下一轮继续查询该结构，第二轮动作决定最终冠军。这里 Qwen 的决定与数值证据排名一致，并未逆转数值冠军。

这说明 Qwen 已处在控制路径上，但本次实验只证明“会基于证据选择”，不能证明 1.5B Qwen 能独立发现正确约束结构。

## 4. 逐轮结果

| 外层轮次 | 轨迹标签 safe / violation | 假设排名（selection score） | 冠军 IoU |
|---|---:|---|---:|
| 1 | 6 / 22 | planar-joint 0.715, x-only 0.670, y-only 0.579, Qwen-h3 0.451 | 0.450 |
| 2 | 8 / 24 | planar-joint 0.843, y-only 0.597, x-only 0.560, Qwen-h3 0.527 | 0.600 |

Oracle 查询来源：24 次 warmup、4 次 shortcut、4 次 local-feature stress。最终共 32 次，其中 8 条 safe、24 条 violation。

最终指标：

| 指标 | 数值 |
|---|---:|
| boundary IoU | 0.6001 |
| grid accuracy | 0.9508 |
| false-safe rate | 0.0293 |
| false-unsafe rate | 0.0508 |
| predicted unsafe fraction | 0.1207 |
| held-out expert safe rate | 0.5000 |

## 5. 边界图评价

输出图：`../outputs/obstacle_avoid_qwen_test/learned_boundary.png`。

黑色学习边界包围了真实圆形障碍的大部分区域，证明轨迹级标签、MIL latent witness 和反例查询能够形成有效二维边界。然而边界明显向左扩张并呈椭圆形，而非恢复出真实圆；留出专家整体安全率也只有 50%。主要原因是专家和查询轨迹对障碍左侧/上下边界的覆盖不均，整轨迹标签又没有指出具体违规时刻，神经网络会选择一个同样能解释当前轨迹标签的更大区域。

因此，本次测试的结论是：

- 双层闭环、严格语义编译、神经约束训练、Falsifier 和整轨迹 Oracle 已端到端跑通；
- 系统找到了正确的二维联合约束类别，并得到中等质量的非参数边界；
- 当前边界尚未被数据唯一辨识，不能声称恢复真实障碍轮廓；
- `held-out expert safe rate=0.5` 是当前最需要改进的指标，后续应优先加入边界两侧的定向反事实和安全查询，而不是单纯增加网络容量。

## 6. 无泄漏审计与可复现文件

对 `evidence_history.json` 和 `semantic_interactions.json` 搜索 `iou`、`obstacle_center`、`safety_radius`、`state_violation_mask`，结果为空。EvidenceCompiler 明确只暴露轨迹级准确率、反例率、不确定度、干预命中率和复杂度。

本次运行的主要产物位于 `../outputs/obstacle_avoid_qwen_test/`：

- `result.json`：最终汇总；
- `semantic_interactions.json`：prompt、原始 Qwen 输出、逐项解析错误与动作；
- `evidence_history.json`：反馈给 Qwen 的无泄漏证据；
- `evaluation_history.json`：隔离保存的评估真值指标；
- `oracle_query_log.json`、`query_diagnostics.json`：32 次查询及其来源；
- `hypothesis_bank.json`：候选和动作审计；
- `constraint_models.pt`：学习到的模型；
- `learned_boundary.png`、`semantic_trace.png`：边界和语义轨迹图。
