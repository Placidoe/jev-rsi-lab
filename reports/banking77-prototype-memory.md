# Laya 上的受约束 RSI：Banking77 原型记忆实验

## 结论

第一版已在本机 CPU 上实际跑通。它不是让模型“自由修改自己”，而是一个可审计的、**由真实标签约束**的自进化环：从训练集取少量已标注意图作为每类原型；在独立验证集上找出错误；每类最多把一个真实错误样本写回该类经验记忆；然后重新检索候选并调用 Laya 决策头。Laya 模型权重和冻结测试集标签均未被修改。

这个选择有针对性：Laya 官方文档明确提醒，大候选集会受候选 token 预算影响；Banking77 有 77 个银行意图，正好是一个公开、可复现的压力场景。[Laya 文档](https://huggingface.co/convaiinnovations/laya/blob/main/README.md?code=true) · [Banking77 数据集](https://huggingface.co/datasets/PolyAI/banking77)

## 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | `convaiinnovations/laya` 英文 checkpoint；CPU 本地推理 |
| 数据 | Banking77 公开英文意图分类数据；训练集用于经验，测试集冻结 |
| 初始经验 | 每个意图 1 条已标注样本，共 77 条 |
| 验证 | 每个意图 2 条，共 154 条 |
| 自进化规则 | 第一轮验证错误中，每个真实标签最多提升 1 条入经验库 |
| 候选机制 | 用 Laya embedding 检索 77 类原型，仅把 top-12 候选交给 Laya `choice` 决策 |
| 冻结测试 | 每类均衡取 1 条，共 77 条；不参与任何更新或策略选择 |

## 结果

| 指标 | 冻结测试基线 | 验证集：更新前 | 验证集：更新后 | 冻结测试最终 |
|---|---:|---:|---:|---:|
| 候选召回率 | 35.1% | 39.6% | 74.0% | 59.7% |
| 最终意图准确率 | 27.3% | 29.2% | 55.2% | **46.8%** |
| 平均单样本耗时 | 147.7 ms | 153.7 ms | 158.9 ms | 157.2 ms |
| 经验库大小 | 77 | 77 | 143 | 143 |

从冻结测试看，准确率提升 **19.5 个百分点**，候选召回提升 **24.6 个百分点**。说明主要收益首先来自“把正确类别放进更短的候选集合”，随后 Laya 的 decision head 才能更可靠地做选择。

## 实现边界

- 这是**经验/检索层 RSI**，并非权重级自训练；因此可回滚、可审计，适合先验证机制。
- 更新信号只来自真实的训练标签，避免模型用自己猜的伪标签滚雪球。
- 测试集保持冻结；报告中基线只作观察，未用于挑选更新样本或超参数。
- 当前 checkpoint 在加载时给出温度参数不在有效范围的警告，因此这里的 `confidence` 不应作为自动化高风险决策依据；本实验只以准确率和候选召回率评估。
- 77 条测试样本是每类一条的快速、均衡 smoke test；下一步应使用完整测试集、多随机种子，并报告均值与置信区间。

## 如何复跑

代码在 `work/laya-rsi/rsi_harness.py`，原始结果在 `work/laya-rsi/results/rsi-banking77.json`。在当前工作目录执行：

```bash
HF_HOME=work/laya-local/hf-cache USE_TF=0 \
  work/laya-local/.venv/bin/python work/laya-rsi/rsi_harness.py \
  --data-dir work/laya-rsi/data \
  --output work/laya-rsi/results/rsi-banking77.json \
  --rounds 2 --test-limit 77
```

## 下一步：真正的“模型级”自进化

1. 把上述经验库与错误案例沉淀为 versioned replay buffer；每次更新生成可比对的 run manifest。
2. 做候选检索器的消融：`top-k=8/12/16/20`、原型数、错误写回上限、多随机种子。
3. 使用官方 Laya 微调路径，在 GPU 上仅训练其决策头或 adapter，并用完整 Banking77 测试集与冻结 replay buffer 做门禁。
4. 只有在离线指标、校准、延迟和回归测试均通过时，才将新 checkpoint 标记为可部署版本；线上继续保留旧版本回滚。

Laya 代码及 checkpoint 标为 Apache-2.0，便于继续做派生实验；但发布或商用前仍应按仓库与数据集的完整许可证核对。[Laya GitHub](https://github.com/NandhaKishorM/laya) · [Banking77 数据卡](https://huggingface.co/datasets/PolyAI/banking77)
