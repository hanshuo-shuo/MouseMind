# MouseMind：连续控制与可验证 Post-Training 重构

本轮状态：连续控制实现、BC 训练、反事实数据生成及首轮 SFT/DPO 训练已完成，
正在审查闭环开发结果。已有开发集指标，尚无最终测试上的提升结论，没有 push。
详细结果和失败记录见 [集群进展](mouse_llm/continuous/CLUSTER_PROGRESS_20261102.md)。

同 GPU 的 80 局开发评测中，commanded-destination 连续 BC 完成 79 局、
无捕获完成 12 局；旧 DPO 为 75 局和 11 局。新 SFT 则始终 HOLD，完成率为零。
因此，目前能展示的是连续执行器改进和可复现实验流程，不能声称新 SFT/DPO
带来了控制收益。后续先诊断实际输出与技能选择，再决定是否改变决策目标。

## 目标和项目主线

保留 MiniMind 作为经过 post-training 的高层决策模型，底层从 295 类动作
预测改为连续绝对目标坐标 `(x, y)`。三个技能 `go_to_goal`、
`evade_predator`、`hold_position` 是高层语义接口，不再要求底层只能到达
295 个 catalog 点。

工作流：历史轨迹/合格 RL 教师 → 连续技能执行器 → 仿真验证的 SFT 和偏好数据
→ MiniMind LoRA SFT → DPO → 当前策略采集和纠错 → 配对闭环评估。

面试中分别说明执行器改进和 LLM 权重更新的效果。已有的 DPO、GRPO 失败记录
保留；本轮首先完成连续控制与 SFT/DPO 主线，不把算法数量当作贡献。

## 连续动作与标签

| 项目 | 旧版 | 本地候选 |
| --- | --- | --- |
| 低层输出 | 0–294 action ID | `[x, y]`，绝对坐标，范围 `[0,1]` |
| 模型监督 | 交叉熵分类 | 连续 mixture-density BC，默认五分量 |
| 部署选择 | 最大 logit 对应的 cell | 最大权重分量的二维均值，不平均不同路线 |
| 环境导航 | 目标被路由为 cell center | 图路由 + 精确 XY 末端，原碰撞/捕获物理 |
| 高层历史 | 离散 action ID | XY 历史、已观测捕食者位置和记忆年龄 |
| Post-training | 旧执行器的短分支标签 | 新执行器重新分支、统一 utility、同一后续策略 |

默认 `next_position` 标签严格取 `next_obs[:2]`，表示下一时刻实际到达的位置。
另一对照 `commanded_destination` 取历史 `action_catalog[action_id]`，恢复旧命令
对应的目标坐标。两者不能混名或混用 checkpoint。

源文件共 118,861 个 transition、4,916 个重建 episode。两种标签在 `1e-6`
容差下全部不同，平均 XY 距离约 0.01896。历史数据来自仿真策略导出。
`next_position` 是合理的短步 waypoint 模仿目标，但不等于当时执行的原动作。
这些重标注记录只验证为 BC 数据；原 reward/next_state 不能作为新动作的 RL
transition。SAC replay 仅接受在线环境中真实执行的动作。

原到点阈值 `0.01` 保留。很短或静止的 next-position 标签可能落入到点死区；
源数据中约 42.86% 的一步位移不超过该阈值，约 7.74% 不超过 `1e-6`。
连续回归损失下降不代表 agent 会前进。数据审计和闭环目的地拒绝率都要报告。
本轮不随意放松阈值或成功判定来提高成绩。

## 先验证底层，再训练高层

1. 在相同 episode 切分上训练 `next_position` 与 `commanded_destination`
   两个连续 BC，固定训练预算。单分量作为结构对照，五分量避免对互斥路线取平均。
2. 在新开发种子比较原离散 MLP、两种连续 BC、连续规则层级。
   主要看 clean success、完成率、捕获次数和拒绝/停滞情况。
3. 用在线 SAC 训练独立全策略教师；可先让其 actor 模仿冻结 BC 预测初始化，
   再收集真实在线经验。旧 TD-MPC2 / VP-TD-MPC2 权重可作为另一个候选，
   但观测字段、`[x,y,wait]` 语义、reset 和指标尚未迁移，不能直接宣称可替代。
4. 完整 RL 教师是整体基线或训练数据来源。它本身会在前进/逃避间决策，
   不应直接改名为 `go_to_goal` 并据此声称高层 MiniMind 有贡献。

当前连续 `evade` 仍是带已观测位置记忆的几何技能，`hold` 是当前位置目标；
本轮没有虚构“所有技能均已通过 RL 强化”的结论。后续技能条件化 RL 可替代它们，
每次改变执行器都必须重新生成高层标签。

RL 教师接口已经实现，但本轮尚未训练 SAC，也未移植旧 TD-MPC2 权重。
在启动 RL 前需要审查原 reward 的到目标附近持续奖励，避免学会在目标旁停留。
如果高层数值决策对照也无法改善冻结 BC，下一轮应优先强化逃避和恢复技能。

## 训练数据与 LLM Post-Training

采集只使用新训练种子 50000–50199。开发集 51000–51079，新的最终测试集
52000–52199。历史 final pool 不再充当未见测试。

每个 anchor 重放 XY 前缀，检查公开观测和物理/控制/RNG fingerprint；
对三个候选技能执行相同决策周期，然后统一交回冻结的连续 BC 专家继续评估。
`--continuation rule` 仅作为显式对照；首轮规则策略表现很差，不宜作为默认后续策略。
默认候选 8 步、总 horizon 32，逐分支重复两次核验结果，拒绝不一致回放。
这不是通用完整 snapshot API，也不把某个 continuation 下的优选技能称为全局最优。

按 seed 隔离 train/validation/test；跨分区去重当前上下文，分区内保留多模态监督。
在每个 seed 的不同时间点稳定采样，保留正常与危险状态。未来可替换为当前 DPO
策略采集，但继续只用训练种子。

SFT 的 chosen 技能、DPO 的 chosen/rejected 使用同一个标量排序，拒绝近 tie。
只把公开 context 发送给模型，回放隐态只存在私有诊断。LoRA manifest 绑定基础模型、
技能 adapter、连续 specialist、数据和执行参数的哈希。SFT 用 chosen response
训练，DPO 从该 SFT 初始化且冻结同一 SFT reference。

数据增强优先做环境可验证的状态覆盖和语义不变的指令改写。坐标扰动、地图旋转、
动作变化都需要重新验证动力学与标签；不能仅改变坐标后沿用旧偏好。

## 归因与验收

| 对照 | 保持不变 | 回答的问题 |
| --- | --- | --- |
| 原离散 vs 连续 BC | 源 episode、信息权限、预算、评测环境 | 连续监督是否改善执行？ |
| next-position vs commanded XY | 网络/训练预算/episode 划分 | 哪种目标含义闭环更好？ |
| 连续规则 vs 连续 MiniMind SFT | 同一冻结执行器 | 学习的规划器贡献多少？ |
| 连续 SFT vs 连续 DPO | 同一执行器、SFT 起点、数据协议 | LLM preference post-training 贡献多少？ |
| 最佳连续系统 vs 旧完整 DPO 层级 | 新 200 种子、统一环境条件 | 整体是否超过历史方法？ |

新环境的精确末端导航与回合 reset 有显式版本。所有配对对照在同一新合同下重跑，
不能把旧报告 22% 与新种子的百分比直接相减。原生历史环境的结果仍保留原标签。

`promotion.py` 检查完整 seed 覆盖和逐 episode 一致性，要求配对 bootstrap 的
clean-success 改善区间下界大于零，task-success 差值下界不低于 -2 个百分点，
captures 差值区间上界不大于零。通过也只产生“可人工复核”，不提交、不推送。
这组区间反映 episode seed 随机性；正式主张还需重复训练种子与预先冻结配置。

最终简历/README 应展示数据版本、checkpoint lineage、SFT/DPO 训练日志、
同条件提升、退化诊断和失败视频。没有闭环证据之前，只能写“实现/验证了流水线”，
不能写“连续动作提高了成功率”。

## 本地资产

代码位于独立 worktree，历史主目录保留。
两套私有连续数据已在相邻的 `MouseMind-private/continuous-v1/` 生成，
每套 train/validation/test 为 60000/5000/5000。原 CSV 未修改。

执行命令见 [RUNBOOK.md](mouse_llm/continuous/RUNBOOK.md)。正式训练、SAC teacher
资格评估、新 MiniMind SFT/DPO、200-seed 最终对照尚待执行。
