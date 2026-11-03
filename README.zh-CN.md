# LingBot-VLA 2.0 × RoboTwin 2.0 —— 微调与评测

在 RoboTwin 2.0（Cobot Magic）机器人操作基准上对 **LingBot-VLA 2.0（6B）** 做 LoRA 微调，
并在 `clean`（seen）/ `randomized`（unseen）两个 setting 下评测全部 50 个任务。

- **团队**：
- **基础模型**：[`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
- **提交 checkpoint**：`lora_tuned_1500`（LoRA rank 64 / alpha 128，训练 1500 步后 merge 回基座）
- **模型权重**：Hugging Face [`Dwyane/lingbot-vla-lora_tuned_1500`](https://huggingface.co/Dwyane/lingbot-vla-lora_tuned_1500)
- **评测规划器**：cuRobo v0.7.8

> English version: [`README.md`](README.md)

## 训练演进过程

最终提交的模型是四个阶段中的最后一站，而不是一次训练的结果。所有阶段使用相同的数据、
环境与合规约束，变化的是训练配方。

| 阶段 | 是什么 | devset | 全量评测 |
|---|---|---:|---|
| **1. 全参数 SFT 11000 步**（`hf_final_11000`） | 官方 RoboTwin 配方，未改任何超参，训练 50 个 clean 任务 | — | clean **48.55%**（670/1380），randomized **24.00%**（180/750） |
| **2. 续训至 6000 步**（`lora6000_base`） | 同一配方，lr 1e-4 | 27/50 = 54.0% | — |
| **3. 在该基座上 LoRA 1500 步**（`lora_tuned_1500`） | LoRA rank 64 / alpha 128，作用于 `action_expert`，lr 5e-5 | **31/50 = 62.0%** | clean **62.8%**（314/500），randomized **37.0%**（185/500） |
| 4. 未采用的备选 | v3：batch 8→16、lr 5e-5→7e-5、前 25 步监督 + 时序平滑 · v2：视觉噪声 0.02 / 状态扰动 0.01 | 29/50 = 58.0%<br>23/41 ≈ 56% | — |

关于阶段 1 的几个要点，后续各阶段都建立在其之上：

- 严格按官方 `train_full_sft.sh` launcher 执行，**未做任何超参调优**，只从外部设定运行规模
  （8 卡、`max_steps=11000`、`save_steps=1000`）。
- 实测训练速度 19.3–22.1 s/it（8 × W7900D），11000 步约需 59–68 小时。
- 为在本平台跑通，只做了基础设施层面的改动（checkpoint 轮转、续训开关、两个环境变量），
  均不影响训练数值。
- 它的全量评测成绩低于阶段 3，原因是步数更少且未做调优；正是
  [`docs/reproduction_report.md`](docs/reproduction_report.md) 中的 devset 对比，才确定了
  改用 LoRA 微调这一方向。

详细报告：[`docs/reproduction_report_hf11000_history.md`](docs/reproduction_report_hf11000_history.md)
（阶段 1，含环境适配与评测口径分析）与 [`docs/reproduction_report.md`](docs/reproduction_report.md)
（阶段 2–4）。

## 结果

最终提交模型（`lora_tuned_1500`）：

| setting | 任务数 | 成绩 | 成功率 |
|---|---:|---:|---:|
| clean2clean（seen） | 50 / 50 | **314 / 500** | **62.8%** |
| clean2random（unseen） | 50 / 50 | **185 / 500** | **37.0%** |

早期全参数 SFT 模型（`hf_final_11000`）：

| setting | 任务数 | 成绩 | 成功率 |
|---|---:|---:|---:|
| clean2clean（seen） | 50 / 50 | 670 / 1380 | 48.55% |
| clean2random（unseen） | 50 / 50 | 180 / 750 | 24.00% |

两次评测的每任务集数不同（明细见报告）；最终模型为每任务 10 集。

每任务 10 集（赛题允许未测满 100 次如实上报，最终以官方复测为准）。
逐任务明细见 [`results/results.json`](results/results.json)，
分析见 [`results/eval_report_clean_randomized.md`](results/eval_report_clean_randomized.md)。

> 任务族差异远大于 setting 差异：`place_*` 摆放/对齐类普遍 60–90%，
> 堆叠/开合/排序类普遍 0–40%；个别任务（如 `stack_bowls_three`）在 unseen 下反而更高。

## 仓库结构

```text
.
├── code/                训练与评测入口、配置、改动留证
│   ├── train.sh         训练入口（官方 launcher 的薄包装）
│   ├── eval.sh          评测入口（clean / randomized 双 setting）
│   ├── configs/         训练配置 / 模型配置 / robot config
│   ├── src/             上游 diff 与补丁、评测分析脚本
│   └── README.md        代码材料详细说明（含代码归属与合规自检）
├── checkpoint/          提交权重的配置文件与 tokenizer（不含 safetensors 分片）
│   ├── config.json
│   ├── lingbotvla_cli.yaml
│   └── tokenizer / preprocessor 配置
├── results/             评测结果（results.json + 评测报告）
└── docs/                复现与调优说明
    ├── reproduction_report.md                 阶段 2–4（LoRA 调优）
    └── reproduction_report_hf11000_history.md 阶段 1（11000 步全参数 SFT）
```

## 模型权重

权重文件（约 12 GB，3 个分片）超过 GitHub 单文件限制，发布在 Hugging Face：
**[`Dwyane/lingbot-vla-lora_tuned_1500`](https://huggingface.co/Dwyane/lingbot-vla-lora_tuned_1500)**

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 2.0 GB | `7bff0166a4cc6e5f66d2b52b144f863679c81ef890c719e1ced5c499cfe22cc4` |
| `model-00002-of-00003.safetensors` | 5.0 GB | `2ba12d8f780fa0df533d764e01b0a09528b1901e02c79c23af00ca96c5d89267` |
| `model-00003-of-00003.safetensors` | 4.6 GB | `495150c7f55d83f65554580179bffa2a33f078ce9d14073f60a57f3ee7957aac` |

```bash
hf download Dwyane/lingbot-vla-lora_tuned_1500 --local-dir ./checkpoint
```

下载后 `checkpoint/` 目录即包含 `config.json`、`model.safetensors.index.json`、
tokenizer 文件与三个权重分片，可直接作为 `MODEL_PATH` 传给 `code/eval.sh`。

> 三个分片均已上传，可直接从上述 HF 仓库下载。
> 若 HF 不可达，也可基于
> [`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
> 与 [`code/configs/lingbotvla_cli.lora1500.yaml`](code/configs/lingbotvla_cli.lora1500.yaml)
> 按 [`docs/reproduction_report.md`](docs/reproduction_report.md) 自行复现。

## 快速开始

环境：8 × AMD Radeon W7900D 48 GB（ROCm 7.2.1 + PyTorch 2.9.1），依赖
`lingbot-vla-v2`、`RoboTwin 2.0`、`LeRobot`、`MPLib`。详见
[`code/README.md`](code/README.md)。

```bash
# 阶段 1：11000 步全参数 SFT（官方配方，仅设定运行规模）
GPU_COUNT=8 MAX_STEPS=11000 SAVE_STEPS=1000 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_11k_8gpu \
bash code/train.sh

# 阶段 2–3：LoRA 微调（见 docs/reproduction_report.md）；
# 使用官方 launcher 并读取 code/configs/lingbotvla_cli.lora1500.yaml

# 评测
MODEL_PATH=<checkpoint-dir> GPU_COUNT=8 EPISODES=100 TASK_CONFIG=both bash code/eval.sh
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [`README.md`](README.md) | 英文版说明 |
| [`code/README.md`](code/README.md) | 环境依赖、两个必设环境变量、训练/评测入口、对官方代码的修改与归属说明 |
| [`docs/reproduction_report.md`](docs/reproduction_report.md) | 最终提交权重的训练超参、评测配置、四版权重 devset 对比、口径与已知限制 |
| [`docs/reproduction_report_hf11000_history.md`](docs/reproduction_report_hf11000_history.md) | 阶段 1：11000 步全参数 SFT —— 训练配方、实测速度、环境适配、评测口径分析、代码改动与归属 |
| [`results/eval_report_clean_randomized.md`](results/eval_report_clean_randomized.md) | 双 setting 逐任务成绩、分布观察、评测成本 |

## 合规与改动说明

- 训练数据仅使用 RoboTwin 2.0 中 Aloha-AgileX 的 **50 个 clean 任务**数据，未使用任何
  randomized 数据或外部数据；`train.sh` 内置自检（清单必须 50 条、不得含 `random`）。
- 对官方代码仅有 2 处改动（checkpoint 轮转、续训开关），均不改变默认训练数值，
  diff 与补丁见 [`code/src/`](code/src)，归属说明见 [`code/README.md`](code/README.md)。

## 引用

- LingBot-VLA 2.0：`robbyant/lingbot-vla-v2-6b`
- RoboTwin 2.0（上游 commit `266f3aa`）
