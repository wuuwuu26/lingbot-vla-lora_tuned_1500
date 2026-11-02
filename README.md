# LingBot-VLA 2.0 × RoboTwin 2.0 —— 微调与评测复现

RoboTwin 2.0（Cobot Magic）机器人操作基准上，对 **LingBot-VLA 2.0（6B）** 做 LoRA 微调，
并在 `clean`（seen）/ `randomized`（unseen）两个 setting 下评测全部 50 个任务的材料汇总。

- **团队**：
- **基础模型**：[`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
- **提交 checkpoint**：`lora_tuned_1500`（LoRA rank 64 / alpha 128，1500 步后 merge 回基座）
- **评测规划器**：cuRobo v0.7.8

## 结果

| setting | 任务数 | 成绩 | 成功率 |
|---|---:|---:|---:|
| clean2clean（seen） | 50 / 50 | **314 / 500** | **62.8%** |
| clean2random（unseen） | 50 / 50 | **185 / 500** | **37.0%** |

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
    ├── reproduction_report.md               最终提交的 lora_tuned_1500
    └── reproduction_report_hf11000_history.md  早期全参数 SFT 版本（环境适配、代码改动）
```

## 模型权重

权重文件（约 12 GB，3 个分片）体积超过 GitHub 单文件限制，发布在 Hugging Face：
**`Dwyane/lingbot-vla-lora_tuned_1500`**

| 文件 | 大小 |
|---|---:|
| `model-00001-of-00003.safetensors` | 2.0 GB |
| `model-00002-of-00003.safetensors` | 5.0 GB |
| `model-00003-of-00003.safetensors` | 4.6 GB |

```bash
huggingface-cli download Dwyane/lingbot-vla-lora_tuned_1500 --local-dir ./checkpoint
```

下载后 `checkpoint/` 目录即包含 `config.json`、`model.safetensors.index.json`、
tokenizer 文件与三个权重分片，可直接作为 `MODEL_PATH` 传给 `code/eval.sh`。

> **状态**：权重分片尚在上传中。在三个分片全部出现于上述 HF 仓库之前，下载只会得到配置文件。
> 若 HF 不可达，也可基于
> [`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
> 与 [`code/configs/lingbotvla_cli.lora1500.yaml`](code/configs/lingbotvla_cli.lora1500.yaml)
> 按 [`docs/reproduction_report.md`](docs/reproduction_report.md) 自行复现。

## 快速开始

环境：8 × AMD Radeon W7900D 48 GB（ROCm 7.2.1 + PyTorch 2.9.1），依赖
`lingbot-vla-v2`、`RoboTwin 2.0`、`LeRobot`、`MPLib`。详见
[`code/README.md`](code/README.md#环境依赖)。

```bash
# 训练（全参数 SFT 入口；最终提交使用 LoRA 配置，见 docs/reproduction_report.md）
GPU_COUNT=8 MAX_STEPS=11000 SAVE_STEPS=1000 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_11k_8gpu \
bash code/train.sh

# 评测
MODEL_PATH=<checkpoint-dir> GPU_COUNT=8 EPISODES=100 TASK_CONFIG=both bash code/eval.sh
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [`code/README.md`](code/README.md) | 环境依赖、两个必设环境变量、训练/评测入口、对官方代码的修改与归属说明 |
| [`docs/reproduction_report.md`](docs/reproduction_report.md) | 最终提交权重的训练超参、评测配置、四版权重 devset 对比、口径与已知限制 |
| [`docs/reproduction_report_hf11000_history.md`](docs/reproduction_report_hf11000_history.md) | 早期 11000 步全参数 SFT 版本的复现细节（环境适配、checkpoint 轮转、续训开关） |
| [`results/eval_report_clean_randomized.md`](results/eval_report_clean_randomized.md) | 双 setting 逐任务成绩、分布观察、评测成本 |

## 合规与改动说明

- 训练数据仅使用 RoboTwin 2.0 中 Aloha-AgileX 的 **50 个 clean 任务**数据，未使用任何
  randomized 数据或外部数据；`train.sh` 内置自检（清单必须 50 条、不得含 `random`）。
- 对官方代码仅有 2 处改动（checkpoint 轮转、续训开关），均不改变默认训练数值，
  diff 与补丁见 [`code/src/`](code/src)，归属说明见 [`code/README.md`](code/README.md#对官方代码的修改)。

## 引用

- LingBot-VLA 2.0：`robbyant/lingbot-vla-v2-6b`
- RoboTwin 2.0（上游 commit `266f3aa`）
