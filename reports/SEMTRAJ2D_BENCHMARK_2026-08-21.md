# SemTraj2D v1.1：发表导向数据集与首轮验证

日期：2026-08-21

## 结论先行

已经新增一套比原始二维单障碍实验更有辨识力的受控基准
`SemTraj2D v1.1.0`。它不是简单增加轨迹数量，而是把导师关心的几类问题
拆成可以分别测量的实验：简单约束能否胜出、是否把演示风格当成约束、动态
约束、多个异质约束、Oracle 查询效率，以及假设语言无法表达任务时能否拒答。

首轮 GPT 检查说明新基准确实能定位故障阶段：GPT 没有 fallback，并生成了
正确的空间结构和“`y` 等式带 + speed 上界”组合结构；但是一次小预算闭环仍
因为数值模型过度保守而不合格。当前可以说“基准已建成并暴露了数值拟合
瓶颈”，还不能说“架构已优于 baseline”。

完整 legacy + benchmark 回归测试共 99 项，当前全部通过。

## 为什么原始二维避障不够

原实验只有一个圆形障碍、少量位置 feature 和单一类型真值。即使方法总选
joint MLP，也可能得到不错的 IoU，因而无法回答下面的问题：

- 简单的一维上界或 equality band 是否有真实胜出空间；
- 专家都从上方绕行时，系统是否误学成“始终保持高 `y`”；
- 约束依赖速度而不是位置时，falsifier 和拟合器是否仍有效；
- feature A 是等式约束、feature B 是不等式约束时能否同时表达；
- 多个假设是否共享 Oracle 查询，而不是逐个浪费预算；
- 真实规则不在当前 IR 中时，系统是否会诚实地返回 `inconclusive`。

SemTraj2D 对这些现象设置了独立真值、反事实 strata 和结构评分，因而能把
“语义结构错”“数值边界错”“分布外泛化错”分开报告。

## 数据集组成

每条轨迹为 100 个二维点。学习器可用 12 个可微 feature：位置、分量速度、
速度模长、分量加速度、加速度模长、航向正余弦、累计路长和归一化进度。
每个任务只有一至三个 feature 真正有因果作用，其余是相关干扰量。

| 任务 | 隐藏结构 | 核心问题 | 私有测试数 |
|---|---|---|---:|
| `disk_clean` | joint `(x,y)` 禁入区 | 上下绕行均出现的空间基准 | 384 |
| `disk_upper_proxy` | 与 clean 完全相同 | 只给上绕专家时的 proxy/generalization | 384 |
| `diagonal_halfspace` | joint 线性位置边界 | 简单线性假设是否能胜出 | 384 |
| `lane_band` | `y` equality band | 一维等式约束 | 384 |
| `speed_limit` | speed upper bound | 动态而非静态位置约束 | 384 |
| `disk_and_speed` | 空间禁入 OR 速度上界 | 异质组合及单 clause 违约 | 672 |
| `lane_and_speed` | `y` 等式带 OR 速度上界 | 导师提出的等式 + 不等式 | 672 |
| `eventually_visit_open_set` | 最终必须访问 checkpoint | 当前 IR 不可表达时拒答 | 384 |

每个任务有 30 条 Oracle 验证安全的公开专家轨迹，固定划分为 `18/6/6`。
atomic/open-set 私有测试在 `id`、`boundary`、`counterfactual`、`ood` 四组内
分别平衡 safe/violation。组合任务另含 `spatial_only`/`lane_only`、
`speed_only` 和 `multi_clause` strata。

### 专门防止“上绕 = 高 y”的成对设计

`disk_clean` 与 `disk_upper_proxy` 使用逐字节相同的任务描述、隐藏规则、私有
评测数组、专家起终点、x 曲线、轨迹 ID 和数据划分。唯一系统变化是公开专家
的绕行侧：clean 上下平衡，proxy 全部上绕。私有 counterfactual/OOD 中仍有
相同起终点的下绕安全轨迹。因此，两个任务之间的性能下降可以归因于演示
homotopy 偏置，而不是换了障碍或测试难度。

### 不再把 x 速度当成 speed 的替代品

speed 任务包含水平、垂直、斜向和局部回退的高速轨迹。生成后审计要求真实
`speed` 的最佳阈值 balanced accuracy 为 `1.0`，而任一带符号单轴速度代理
低于 `0.90`。这修复了初版几乎一直向右运动、`x_velocity` 可近乎完美替代
speed 的不可辨识问题。

## 公私信息边界

公开目录只包含任务文字与 feature 定义、安全演示、split 和文件 hash。
私有目录包含解析 Oracle、结构真值和全局生成的评测轨迹。私有评测轨迹由
cubic Bezier 控制点直接构造，不是对专家内部 waypoint 的扰动；圆形碰撞按
整段 polyline 连续检查，避免采样点跨过障碍却漏标。

公开采样 seed 与私有规则 seed 已分离。未指定私有 seed 文件时，生成器使用
不可导出的随机 256-bit seed；task 随机流按稳定 task key 派生，不再依赖任务
在列表中的顺序。当前 runner 在训练结束后写 `freeze_manifest.json`，独立
post-hoc 进程校验 checkpoint/result hash 后才读取私有评测文件。

这属于强代码路径隔离，但还不是安全边界：正式盲测仍应把 membership Oracle
放到独立服务或容器中，训练容器不挂载 private root，并在代码/配置冻结后由
评测方秘密生成实例、限制提交次数。

## 新增的评价口径

主指标不再只看 state-grid IoU，而是：

- private trajectory balanced accuracy、AUPRC、safe accuracy、violation recall；
- `id/boundary/counterfactual/ood` 中最差组 balanced accuracy；
- composite 的各单 clause recall 和 multi-clause recall；
- Oracle 调用总数与 performance-query AUC；
- permutation-invariant typed structure match；
- open-set correct-inconclusive rate。

结构评分现在明确拆成两项：

1. `exact_structure_recovery`：冻结候选的变量、coupling、relation、temporal、
   model family 和组合结构是否匹配真值；
2. `qualified_exact_structure_recovery`：结构匹配，并且该候选在查看私有数据前
   已通过全部训练证据 gate。

这样不会把“结构猜对但数值模型不合格”错误记为纯粹的结构识别失败。静态 IoU
只对纯空间任务作为辅助指标；对 speed 和异质 composite 它没有统一含义。

## 已完成的 GPT smoke

### 1. `disk_upper_proxy`，只检查语义生成

- backend/model：OpenAI `gpt-5.6-sol`；
- 接受 3 个 LLM hypothesis；fallback `0`，augmentation `0`；
- 第一候选为 joint `(x_position, y_position)`、`forbidden_region`、`max`、MLP；
- 另外保留 separable MLP 和 affine linear 两个可证伪替代项；
- 没有 all-feature MLP，也没有“始终高 y”假设。

### 2. `lane_and_speed`，只检查多约束语义

- 接受 2 个 LLM hypothesis；fallback `0`，augmentation `0`；
- 第一候选精确包含 `y_position equality_band` 与 `speed upper_bound`，并用
  `any_violation` 合成；
- 第二候选把 equality band 等价拆成 `y` 下界、`y` 上界和 speed 上界；
- 这验证了“feature A 等式 + feature B 不等式”在 IR、schema 和 GPT 输出链路
  中都可以表达，不会退化为单个全 feature joint MLP。

### 3. `disk_upper_proxy`，两轮闭环与独立私有评测

| 项目 | 结果 |
|---|---:|
| Oracle queries | 26 |
| LLM interactions / fallbacks | 2 / 0 |
| 冻结候选 | `h_joint_position_keepout` |
| selection status | `inconclusive` |
| typed structure match | 是 |
| qualified exact recovery | 否 |
| trajectory balanced accuracy | 0.640625 |
| safe accuracy | 0.28125 |
| violation recall | 1.000000 |
| worst-group balanced accuracy | 0.500000 |
| AUROC / AUPRC | 0.427002 / 0.422983 |

分组 balanced accuracy 为 boundary `0.500`、counterfactual `0.78125`、ID
`0.78125`、OOD `0.500`。模型找到了所有违约轨迹，却把大量安全轨迹也判成
违约；训练期也因 `safe_accuracy_below_gate` 没有宣布 qualified champion。
因此当前失败是“正确语义结构上的数值边界过宽/过保守”，而不是 GPT 输出不
合规或 fallback。这个结果不能用于宣称性能优越，但正说明新 benchmark 能把
原实验掩盖的问题暴露出来。

## 发表前必须完成的对照实验

预注册模板固定五个 hidden instance public seeds：`20260821`、`20260822`、
`20260823`、`20260824`、`20260825`；每个实例另用不可公开的独立私有 seed。
每个冻结语义 bank replay 五个 numeric seeds：`7, 19, 37, 73, 109`。主要预算
为总计 `Q=32`，并报告 `Q=0/8/16/24/32` 曲线；warmup 也计入总预算。

在相同 Oracle 调用数和候选优化次数下至少比较：

1. 完整 GPT LLM-Modulo CEGIS；
2. 相同 round-0 bank、禁止 revision 的 one-shot GPT；
3. 无 LLM 的预注册 typed-bank CEGIS；
4. all-feature joint MLP；
5. 只给正确结构、不提供数值边界的 oracle-structure ceiling。

统计单位应是 hidden task instance，而不是把五个 optimizer seeds 当成五个独立
任务。主结论应来自最差组 trajectory BA、query-performance AUC、paired
clean-to-proxy degradation、composite 最差单 clause recall 和 open-set 拒答；
不能只选择平均 ID accuracy 或某个成功 seed。

## 当前能说与不能说

现在有证据支持：

- 数据与接口能系统地区分语义结构、数值拟合和分布外错误；
- GPT 在两个代表任务上能按 schema 生成有用结构，未发生 fallback；
- 多约束和简单约束都有独立表达与评分空间；
- 私有全局 probe 能揭示上绕专家带来的未见安全路线问题。

现在仍不能声称：

- 完整架构优于 all-feature MLP 或无 LLM baseline；
- outer-loop revision 已带来统计显著提升；
- learner falsifier 已能主动发现全新 homotopy——当前学习侧候选仍以专家为锚；
- 已实现物理隔离的盲 Oracle 服务；
- 二维受控结果可替代一个独立来源的高维机器人仿真实验。

## 关键入口与产物

- 数据说明：`data/SemTraj2D/README.md`
- 生成器：`generate_semtraj2d.py`
- suite runner：`run_semtraj2d.py`
- 独立 evaluator：`evaluate_semtraj2d.py`
- 结构评分：`src/llm_modulo_cegis/structure_evaluation.py`
- 合同测试：`tests/test_semtraj2d.py`
- 发表协议：`configs/semtraj2d_publication_plan.yaml`
- disk GPT semantic：`../../outputs/semtraj2d_v11_disk_upper_gpt_semantic_20260821_net/`
- composite GPT semantic：`../../outputs/semtraj2d_v11_lane_speed_gpt_semantic_20260821/`
- disk closed loop：`../../outputs/semtraj2d_v11_disk_upper_gpt_closed_loop_20260821_retry/`

## 设计依据

SemTraj2D 沿用 trajectory-level inverse constraint learning 的评测设定，并参考
[ICRL benchmark](https://arxiv.org/abs/2206.09670) 将可行性判断作为轨迹级
监督。专门加入同端点 counterfactual 和 proxy reversal，是因为被动演示中的
相关性可能造成 causal confusion，参见
[Gupta et al.](https://proceedings.mlr.press/v213/gupta23a.html)。使用多个隐藏
task instances 而不是只换 optimizer seed，是为了避免把单环境拟合误当成结构
不变性；这一动机与
[Ahuja et al.](https://proceedings.mlr.press/v119/ahuja20a.html) 对多环境不变
学习的讨论一致。
