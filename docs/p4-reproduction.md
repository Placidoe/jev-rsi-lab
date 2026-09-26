# P4 公开实验复现

P4 是推理/评测实验，不训练模型权重。四个脚本均在运行时获取固定版本的公开来源；`results/`
已经被 `.gitignore` 忽略，因此逐题任务、模型输出、Memory 文本和轨迹只能留在本次临时运行目录，
不得提交回仓库。

## 共同前提

```bash
git clone https://github.com/noahshinn/reflexion.git work/reflexion_public_pinned
git -C work/reflexion_public_pinned checkout 218cf0ef1df84b05ce379dd4a8e47f17766733a0
```

P4-A/D 仅依赖 Python 标准库和该公开仓库。P4-B/C 还要求 CUDA、PyTorch 与 Transformers；实际运行
前应锁定自己的依赖版本、模型 revision、GPU 名称和随机种子。模型下载与上游数据的许可证、使用
条款需在运行前自行核验。

## P4-A：公开 Reflexion 日志审计

```bash
python3 scripts/p4a_public_reflexion_audit.py \
  --repo work/reflexion_public_pinned \
  --output results/p4a_public_reflexion_audit
```

该脚本审计 HumanEval 配对日志与 WebShop 累计摘要。它会生成逐题的临时证据文件以便本次审计，
但这些文件不应进入 Git；提交时只能导出脱敏聚合指标。

## P4-B：冻结 HumanEval 的原始检索对照

```bash
JEV_MODEL_ID=Qwen/Qwen2.5-Coder-3B-Instruct \
JEV_REFLEXION_DIR=work/reflexion_public_pinned \
JEV_OUTPUT_DIR=results/p4b_frozen_humaneval \
python3 scripts/p4b_frozen_humaneval_memory_transfer.py
```

脚本会用固定 hash 划分构建/冻结任务，用同一模型、解码设置与资源受限单元测试比较 baseline 和
raw top-3 Memory。它刻意写出临时逐题结果，供 P4-C 使用；该目录保持在 `results/` 下，不可提交。

## P4-C：选择性蒸馏 Memory

```bash
JEV_P4B_ROWS=results/p4b_frozen_humaneval/p4b_frozen_case_results.jsonl \
JEV_P4B_SCRIPT=scripts/p4b_frozen_humaneval_memory_transfer.py \
JEV_OUTPUT_DIR=results/p4c_selective_distilled_memory \
python3 scripts/p4c_selective_distilled_memory.py
```

P4-C 依赖同一次 P4-B 的冻结集与 baseline 行。它复用 baseline，不能替代独立全量复现；报告中必须
声明这一点，并报告纠正、回退、注入比例、token 和延迟。

## P4-D：ALFWorld 长程日志审计

```bash
JEV_REFLEXION_DIR=work/reflexion_public_pinned \
JEV_OUTPUT_DIR=results/p4d_public_alfworld_longitudinal \
python3 scripts/p4d_public_alfworld_longitudinal_audit.py
```

P4-D 不调用模型，也不执行 ALFWorld 动作；它将公开累计日志转换为观察性证据与确定性
build/holdout 划分。若输出目录已存在，脚本会拒绝覆盖，避免误删此前结果。

## P4-E：冻结 ALFWorld 选择性 Memory 试点

P4-E 是第一个真正的公开环境干预，不是 P4-D 的观察性日志审计。先在临时运行目录安装环境与
下载公开 ALFWorld 数据，再以 fresh subprocess 跑 pilot：

```bash
JEV_ALFWORLD_DATA_DIR=work/alfworld_data \
python3 scripts/p4e_kaggle_setup.py

JEV_ALFWORLD_DATA_DIR=work/alfworld_data \
JEV_OUTPUT_DIR=results/p4e_frozen_alfworld \
JEV_PILOT_GAMES=8 JEV_MAX_STEPS=30 \
python3 scripts/p4e_frozen_alfworld_selective_memory.py
```

输出含完整环境轨迹、命令、模型输出与 Memory 文本，必须始终留在 `results/` 或其他被忽略的临时
目录中。进入仓库的只能是重新撰写的聚合报告与 metrics。该脚本若发现输出目录已存在会拒绝覆盖。
当前 8-game 试点两臂均为 0/8 成功；详见 [P4-E 报告](../reports/p4e-frozen-alfworld-pilot.md)。

## 环境变量

| 变量 | 用途 |
| --- | --- |
| `JEV_MODEL_ID` | P4-B/C 使用的开源模型 ID |
| `JEV_REFLEXION_DIR` | 固定提交的公开 Reflexion 检出目录 |
| `JEV_OUTPUT_DIR` | 本次运行的临时输出目录；应位于 `results/` |
| `JEV_P4B_ROWS` | P4-C 读取的 P4-B 临时 baseline 行 |
| `JEV_P4B_SCRIPT` | P4-C 导入的 P4-B 脚本路径 |
| `JEV_ARCHIVE_BASE` | 可选的临时 zip 输出前缀 |

无论环境变量如何设置，原始题目、逐题模型输出、Memory 文本、完整轨迹和压缩包都不应被加入 Git。
