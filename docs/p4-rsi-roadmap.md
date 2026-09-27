# P4 RSI 后续路线图

## 已完成

- [x] 修复行动状态传播与 parser fallback 混杂；
- [x] 在公开环境预注册 discovery / hidden 边界；
- [x] 拒绝无效的“更长文字 Memory”候选；
- [x] 冻结显式状态控制器后进行一次 hidden 验证；
- [x] 发布脱敏聚合指标、可执行脚本、Markdown 报告与原创 SVG。

## 必须先做的确认实验

1. 多 seed：至少 3 个预先声明 seed，报告每个 seed 的成功、纠正、回退、fallback 与 decoder 违规。
2. 第二模型：同一冻结 controller 与相同协议，只替换公开模型和已声明 revision。
3. OOD 长程任务：使用互斥公开环境/任务集合，不能根据其结果调整控制器。

## 升级门槛

只有三项确认实验均通过，才可以讨论更大规模的 offline RSI prototype。即便如此，也不自动授权：

- 生产 Memory 写入；
- 训练标签生成；
- 真实业务系统动作。

这些授权需要各自独立的数据、风险、人工审核与发布流程。
