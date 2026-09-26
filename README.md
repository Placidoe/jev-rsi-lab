# JEV-RSI Lab

一组可复现的 Memory-RSI（recursive self-improvement）研究实验：从轨迹中生成候选经验，
再用独立复核与确定性 Replay 决定是否可在**离线评测**中使用。

> 本仓库不训练模型权重、不执行真实业务动作，也不把模型共识当作事实真值。

## 研究问题

```mermaid
flowchart LR
  T[轨迹 / 公开数据] --> A[候选动作或经验]
  A --> V[独立复核 + Replay]
  V -->|通过| M[staging Memory]
  V -->|拒绝| X[隔离与错误分析]
  M --> E[冻结集评测]
  E --> R[成功率、回退率、风险指标]
```

核心问题不是“模型能否写出反思”，而是：被外部可验证信号约束的经验，是否能在未见样本
上带来收益，同时不引入回退或不安全结果。

![受约束 Memory-RSI 闭环](assets/diagrams/rsi-closed-loop.svg)

## 已纳入的实验

| 实验 | 数据 | 目标 | 状态 |
| --- | --- | --- | --- |
| Public retail semantic routing | Bitext 公开电商客服数据集，运行时下载 | 仅测语义路由，不做业务决策 | 可复现 |
| P2-I constrained RSI | 脚本内生成的合成样本 | 验证 Replay 门控的最小闭环 | 可复现 |
| P2-G synthetic Replay | 50 条合成 complaint / exchange fixtures | 验证候选动作与确定性 Replay 的职责分离 | 已记录 |
| P3-A three-model Memory-RSI | 100 条完全合成、多语种样本 + 虚构版本化政策 | 评估 Replay-validated Memory 的冻结集收益 | 已记录 |
| Banking77 prototype memory | Banking77 公开数据，运行时下载 | 受真实标签约束的原型检索实验 | 已记录 |
| P3-B private-gold boundary audit | 私有人工审核输入（不入库） | 验证“无外部批准策略时不准入 Memory”的否定门控 | 已归档为聚合结论 |
| P4-A / D public evidence audits | 固定版本的公开 Reflexion 日志，运行时获取 | 审计公开的反思与长程环境证据，冻结下一轮干预设计 | 已归档 |
| P4-B / C frozen HumanEval | 公开 HumanEval，运行时获取 | 比较原始检索与选择性蒸馏 Memory 的跨任务迁移 | 已归档；仅候选信号 |

P3-A 的实际结果是冻结集 direct-policy pass rate 从 **37.5%** 到 **40.0%**（+2.5pp，
纠正 1 条、没有已成功样本回退）。这只证明该受控合成环境有微弱可测收益；**不代表生产安全、
模型可自证真值或模型权重实现自进化**。详见 [报告](reports/p3a-synthetic-memory-rsi.md)。

P4 的完整公开证据链见 [P4 报告](reports/p4-public-memory-evidence.md)：P4-B 在 36 个冻结
HumanEval 任务上发现原始 top-3 检索使通过率从 66.67% 降至 52.78%；P4-C 的选择性蒸馏在同一
冻结集得到 69.44%（+1 个净纠正）。后者复用了 P4-B baseline，且样本很小，故只是下一轮独立
复现的候选信号，不能作为因果或生产结论。

## 数据与隐私边界

- 仓库不保存 Bitext 原始数据；脚本从上游数据集运行时下载，使用前应复核其许可证与使用条款。
- 仓库中的 P2-I 与 P3-A 样本、政策、Memory 均为合成评测材料，不含真实客户、员工或业务数据。
- 含人工标注、个人标识或内部协作记录的旧材料没有迁入本仓库。
- P3-B 的原始人工审核集、工作簿、逐条结果和截图均不入库；仅保留重新撰写的脱敏聚合报告。
- P4 不复制公开上游的任务、对话、逐题生成或完整轨迹；仅保存来源钉扎信息、聚合指标和原创图。
- 任何新数据进入仓库前，必须先做来源、再分发许可、脱敏和可公开性检查。

## 结构

```text
scripts/    Kaggle 可执行实验脚本
data/       仅合成 P3-A 评测输入、虚构政策与 staging Memory
reports/    结果、口径和局限
docs/       实验协议与数据来源说明
```

## Kaggle 跑法

启用 GPU 和 Internet，上传单个 `scripts/*.py` 或把仓库作为 Dataset 挂载：

```bash
python /kaggle/input/jev-rsi-lab/scripts/laya_public_retail_benchmark.py
python /kaggle/input/jev-rsi-lab/scripts/laya_p2i_simulated_constrained_rsi.py
python /kaggle/input/jev-rsi-lab/scripts/laya_p3a_three_model_memory_rsi.py
```

P3-A 还需要把 `data/synthetic/p3a/p3a_synthetic_business_policy_v1.json` 作为 Kaggle Input
提供给脚本；脚本会同时校验政策文件的唯一性与版本。

实验文档与数据产物索引见 [reports/artifact-index.md](reports/artifact-index.md)。
P4 的公开复现命令、临时产物边界与环境变量见 [P4 复现说明](docs/p4-reproduction.md)。
