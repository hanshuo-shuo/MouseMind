# 保留旧高层，只替换连续 BC 执行器

用户要求的第二轮实验。保留历史 DPO 高层权重，接入已训练的连续
commanded-destination BC；不重新训练高层，不 push。首轮新 SFT/DPO
的失败记录仍见 [完整开发报告](CLUSTER_PROGRESS_20261102.md)。

## 改动边界

- GO_TO_GOAL：旧离散 MLP 改为冻结的连续 BC，直接执行原始 XY 请求。
- EVADE_PREDATOR / HOLD_POSITION：继续调用旧几何控制器及原目录目标。
- 高层：原 MiniMind base、旧 DPO LoRA、旧上下文格式、1024-token prompt、
  greedy 解码、8 步规划周期和可见威胁中断逻辑均保留。
- 历史适配：旧高层只接收离散动作历史，因此连续 GO 请求映射为最近目录 ID
  作为历史代理。该 ID 不用于执行；它近似描述请求目标，仍可能造成输入分布变化。
- 环境：沿用同一 project-v2 导航、fresh-reset-v1、0.01 投影 inset 和
  原捕获/成功定义。接口不引入额外捕食者信息。

实现复用 `ProposeVerifyPolicy`，只注入 GO 执行器适配。已有兼容开关会把
旧模型接到新上下文，因此本实验使用独立 `legacy-minimind-continuous-bc`
策略，不借用该开关。

## 同条件对照

| 系统 | 高层 | GO 执行 | EVADE / HOLD |
| --- | --- | --- | --- |
| legacy | 旧 DPO | 旧 MLP | 旧规则 |
| bc | 无高层 | 连续 BC | 无独立技能切换 |
| hybrid | 同一个旧 DPO | 同一个连续 BC | 同一旧规则 |

先用开发种子 51000–51003 检查真实运行，再对三组全部使用
51000–51079 的 80 个开发种子，在相同 GPU 上顺序重跑。记录到达率、
无捕获到达率、每局捕获次数、技能用量以及按种子配对的差值。
开发集 bootstrap 区间只作探索比较，不能替代封存的最终测试。

冻结输入：

- 旧 DPO LoRA SHA256：`038b87549eeec280045f951fef0f38b3cfa8808f185bfe7b555e149d6b6d1d02`
- 连续 BC SHA256：`f16dd4a259c646620d1f0e697c8c492ce2a4a8d918649746d6d6d99696b6feab`
- 执行合同 SHA256：`c577a3b8828ba19cb100f5f73b24fde946c3aeef60a05c2c45906b8537c09ece`

## 当前验证

30 项轻量测试通过，覆盖旧行为等价、精确连续 GO、历史代理、旧 EVADE/HOLD、
reset、中断、元数据和 CLI 来源约束。此处不代表新的策略表现；集群评测结果待回填。
所有私有轨迹、权重和逐局记录保留在 Git 外，原系统与首轮实验不覆盖。
