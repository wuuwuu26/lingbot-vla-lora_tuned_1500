# LingBot-VLA 2.0 × RoboTwin 2.0 — Fine-tuning and Evaluation

LoRA fine-tuning of **LingBot-VLA 2.0 (6B)** on the RoboTwin 2.0 (Cobot Magic) robotic
manipulation benchmark, evaluated on all 50 tasks under both `clean` (seen) and
`randomized` (unseen) settings.

- **Team**: 
- **Base model**: [`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
- **Submitted checkpoint**: `lora_tuned_1500` (LoRA rank 64 / alpha 128, 1500 steps, merged back into the base)
- **Motion planner used for evaluation**: cuRobo v0.7.8

> 中文版说明见 [`README.zh-CN.md`](README.zh-CN.md)。

## Results

| Setting | Tasks | Score | Success rate |
|---|---:|---:|---:|
| clean2clean (seen) | 50 / 50 | **314 / 500** | **62.8%** |
| clean2random (unseen) | 50 / 50 | **185 / 500** | **37.0%** |

10 episodes per task (the rules permit reporting under-measured results; official
re-evaluation takes precedence). Per-task details are in
[`results/results.json`](results/results.json) and the analysis in
[`results/eval_report_clean_randomized.md`](results/eval_report_clean_randomized.md).

> Task-family differences dominate setting differences: `place_*` (placing / alignment)
> tasks generally score 60–90%, while stacking / opening / sorting tasks generally score
> 0–40%. A few tasks (e.g. `stack_bowls_three`) actually score higher under unseen randomization.

## Repository layout

```text
.
├── code/                Training and evaluation entry points, configs, change evidence
│   ├── train.sh         Training entry (thin wrapper around the official launcher)
│   ├── eval.sh          Evaluation entry (clean / randomized)
│   ├── configs/         Training config / model config / robot config
│   ├── src/             Upstream diffs and patches, evaluation analysis scripts
│   └── README.md        Detailed notes (code ownership, compliance checks)
├── checkpoint/          Config and tokenizer for the submitted weights (no safetensors shards)
│   ├── config.json
│   ├── lingbotvla_cli.yaml
│   └── tokenizer / preprocessor configs
├── results/             Evaluation results (results.json + report)
└── docs/                Reproduction and tuning notes
    ├── reproduction_report.md                  Final submitted lora_tuned_1500
    └── reproduction_report_hf11000_history.md  Earlier full-parameter SFT run
```

## Model weights

The weights (~12 GB across 3 shards) exceed GitHub's file size limit and are published on
Hugging Face: **`Dwyane/lingbot-vla-robotwin-weights`**

| File | Size |
|---|---:|
| `model-00001-of-00003.safetensors` | 2.0 GB |
| `model-00002-of-00003.safetensors` | 5.0 GB |
| `model-00003-of-00003.safetensors` | 4.6 GB |

```bash
hf download Dwyane/lingbot-vla-robotwin-weights --local-dir ./checkpoint
```

After downloading, `checkpoint/` contains `config.json`, `model.safetensors.index.json`,
the tokenizer files and the three weight shards, ready to be passed as `MODEL_PATH` to
`code/eval.sh`.

> **Status**: the weight shards are still being uploaded. Until all three appear in the
> Hugging Face repository above, downloading returns only the config files. If HF is
> unreachable, you can reproduce the model from
> [`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b) and
> [`code/configs/lingbotvla_cli.lora1500.yaml`](code/configs/lingbotvla_cli.lora1500.yaml)
> following [`docs/reproduction_report.md`](docs/reproduction_report.md).

## Quick start

Environment: 8 × AMD Radeon W7900D 48 GB (ROCm 7.2.1 + PyTorch 2.9.1), depending on
`lingbot-vla-v2`, `RoboTwin 2.0`, `LeRobot` and `MPLib`. See
[`code/README.md`](code/README.md).

```bash
# Training (full-parameter SFT entry point; the submitted model uses the LoRA config,
# see docs/reproduction_report.md)
GPU_COUNT=8 MAX_STEPS=11000 SAVE_STEPS=1000 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_11k_8gpu \
bash code/train.sh

# Evaluation
MODEL_PATH=<checkpoint-dir> GPU_COUNT=8 EPISODES=100 TASK_CONFIG=both bash code/eval.sh
```

## Documentation

| Document | Contents |
|---|---|
| [`README.zh-CN.md`](README.zh-CN.md) | This README in Chinese |
| [`code/README.md`](code/README.md) | Environment, required env vars, train/eval entry points, modifications to official code and their ownership |
| [`docs/reproduction_report.md`](docs/reproduction_report.md) | Training hyper-parameters, evaluation setup, devset comparison of four checkpoints, caveats |
| [`docs/reproduction_report_hf11000_history.md`](docs/reproduction_report_hf11000_history.md) | Reproduction details for the earlier 11000-step full-parameter SFT run |
| [`results/eval_report_clean_randomized.md`](results/eval_report_clean_randomized.md) | Per-task scores for both settings, distribution observations, evaluation cost |

## Compliance and modifications

- Training uses only the **50 clean tasks** of RoboTwin 2.0 (Aloha-AgileX). No randomized
  data and no external data are used; `train.sh` enforces this (exactly 50 entries, no `random`).
- Only 2 changes are made to the official code (checkpoint rotation and a resume switch),
  neither of which alters default training numerics. Diffs and patches are in
  [`code/src/`](code/src); ownership notes are in [`code/README.md`](code/README.md).

## References

- LingBot-VLA 2.0: `robbyant/lingbot-vla-v2-6b`
- RoboTwin 2.0 (upstream commit `266f3aa`)
