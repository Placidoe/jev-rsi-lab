# P4-E 至 P4-J：公开 ALFWorld RSI 实验记录

这是根据每轮原始运行报告、可执行脚本和聚合指标重新撰写的 Markdown 实验记录。它不是任何协作
平台文档的导出或复制品。为避免再分发上游公开任务，本文不含任务标识、prompt、Memory 原文、
逐 case 结果、动作轨迹或压缩包。

## 研究问题与固定边界

研究对象是：当模型已经看到固定的文字 Memory 时，显式任务状态和合法动作约束是否能在公开
长程文本环境中改善完成率。所有阶段都是 offline evaluation：

- 不训练或更新模型权重；
- 不写入生产 Memory；
- 不把实验输出变成训练标签；
- 不执行真实业务动作；
- 每轮的 raw task、模型输出、逐步轨迹和逐 case JSONL 只留在被忽略的 results/。

## 时间线

| 阶段 | 目的 / 变更 | 旧 arm → 候选 arm | 结论 |
| --- | --- | --- | --- |
| P4-E | 初始冻结 pilot | 0/8 → 0/8 | 无成功率信号 |
| P4-F2 | 加共享 loop guard | 0/8 → 0/8 | 仍没有能力地板 |
| P4-F3 | 去除数字索引位置偏差 | 0/8 → 0/8 | 仍不可识别 Memory 效果 |
| P4-F4 | 加固定公开 scaffold | 0/8 → 0/8 | 仍无成功率证据 |
| P4-F5 | 修复 env.step 后 action-state 未刷新 | 6/8 → 8/8 | 首次能识别差异；但 parser fallback=62，仍有混杂 |
| P4-F6 | prefix-constrained 合法命令解码 | 5/8 → 7/8 | +25pp、2 纠正、0 回退；fallback=0 |
| P4-G | 分层 discovery / hidden 预注册 | 1/12 → 2/12 | 仅 discovery，p=1.0，不作确认性声明 |
| P4-H | 仅用 discovery 演化更长文本 Memory | 2/12 → 2/12 | 负结果；hidden 未打开 |
| P4-I | typed state compiler + phase filter | 2/12 → 11/12 | discovery +75pp；冻结候选 |
| P4-J | 一次性 hidden validation | 4/12 → 11/12 | +58.33pp、7 纠正、0 回退、p=0.015625 |

## 关键接口修复

P4-F5 发现了一个重要的实验接口缺陷：环境执行动作后返回的 infos 没有进入下一轮决策，导致
后续决策仍使用初始 admissible-command state。修复后，P4-F5 首次出现非零完成率；这不是
Memory 增益证据本身，而是后续因果比较的前提。

P4-F6 又去掉了“模型输出无法解析时静默回退到一个默认动作”的混杂：decoder 只允许当前合法命令
的 token prefix，违规即失败而非替代执行。该轮 parse fallback 与 constrained-decoder violation
均为 0。

## 预注册与负结果

P4-G 以 6 类任务、每类 2 个 discovery 和 2 个 hidden 游戏的方式建立互斥分割。仅执行
discovery 时，已有文字 Memory 为 1/12→2/12，配对精确检验 p=1.0；它不是 confirmatory evidence。

P4-H 使用 discovery 失败模式生成一版更长的、任务无关状态机文字 Memory。old/candidate 均为
2/12，净纠正 0，p=1.0；因此候选被拒绝、hidden 结果没有被查看。这排除了“只增加 Memory
文字长度就足够”的解释。

## 状态控制器证据

P4-I/J 的候选不是再修改文字 Memory，而是把目标与进度编译成 typed state，在每一步按 phase
过滤环境当前给出的合法命令。候选与对照保持模型、冻结 text Memory、公开 scaffold、decoder、
游戏和 counterbalanced arm order 一致。

| 指标 | P4-I discovery | P4-J hidden |
| --- | ---: | ---: |
| 游戏数 | 12 | 12 |
| 旧 text Memory 成功 | 2 | 4 |
| 状态控制器成功 | 11 | 11 |
| 成功率增益 | +75.00pp | +58.33pp |
| 纠正 / 回退 | 9 / 0 | 7 / 0 |
| 双侧精确 McNemar | 0.00390625 | 0.015625 |
| parser fallback / decoder violation | 0 / 0 | 0 / 0 |

P4-J 的完整解释、限制、机器可读聚合指标和原创图见
[P4-I/J 验证报告](p4ij-state-controller-validation.md)。

## 当前结论与下一门槛

当前能说的是：在一次固定模型、固定公开环境、固定 split 的受控离线比较中，显式状态与相位过滤
在 hidden split 中优于冻结文字 Memory。当前不能说跨 seed、跨模型、跨环境或生产有效。

下一轮必须是预先定义的受控复制：至少 3 个 seed、第二个公开模型和一个 OOD 长程环境；所有三项
完成前，production Memory、training labels 和 business actions 继续保持未授权。
