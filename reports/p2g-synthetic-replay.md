# P2-G：合成 Replay 原型

## 实验问题

当轻量模型给出一个候选动作时，系统是否能把“预测”与“最终决策”分离，让确定性的
Replay mock 拒绝不满足政策前置条件的候选？

## 实验材料

- 50 条完全虚构 fixtures：25 条 complaint、25 条 exchange；
- Laya 只读取最小语义字段：`case_id`、`locale`、`customer_message`、`evaluation_only`；
- 业务状态与最终判断仅由合成的确定性 Replay mock 使用；
- 没有真实客户、真实业务政策、工具调用、Memory 写入或训练标签。

## 结果

| 指标 | 结果 |
| --- | ---: |
| 候选动作命中率 | 23/50（46.0%） |
| Replay 拒绝数 | 27 |
| Replay 拒绝率 | 54.0% |
| Replay 后最终 policy match | 50/50（100%） |
| Laya 推理速率 | 137.36 cases/s |

这意味着：模型候选本身并不可靠，但当最终动作必须经过可解释的规则验证时，系统可以
拒绝错误候选并保持最终输出与合成政策一致。它证明的是“候选与验证分离”的工程模式，
不是模型具备真实业务决策能力。

完整 machine-readable 指标见 [p2g_synthetic_replay_metrics.json](p2g_synthetic_replay_metrics.json)。
