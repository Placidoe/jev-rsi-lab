# 数据来源与再分发规则

## 公开语义路由评测

- 来源：[`bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset)
- 使用方式：`scripts/laya_public_retail_benchmark.py` 在运行时调用 `datasets.load_dataset` 下载。
- 仓库策略：不提交原始数据副本；请在运行前自行阅读并遵守上游 Dataset Card、许可证和平台条款。
- 任务边界：只将五个原始 intent 映射为三类**语义路由**，不推断退款资格、权限或真实业务动作。

## 合成 Memory-RSI 评测

`data/synthetic/p3a/` 中的案例、虚构政策和 staging Memory 均由实验脚本生成；它们仅用于
离线评测 Replay 门控机制。它们不是实际业务政策，也不能作为生产规则。

## 新数据准入清单

在添加任何数据或导出前确认：

1. 数据来源可追溯，且用途与许可证相符；
2. 没有个人身份信息、机密业务字段或受限内部材料；
3. 可再分发，或改为提供下载脚本与 manifest；
4. train / validation / test 的隔离方式可复现；
5. 数据卡记录了局限、偏差与禁止用途。
