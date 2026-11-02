# LingBot-VLA 2.0 × RoboTwin 2.0 — Fine-tuning and Evaluation

LoRA fine-tuning of **LingBot-VLA 2.0 (6B)** on the RoboTwin 2.0 (Cobot Magic) robotic
manipulation benchmark, evaluated on all 50 tasks under both `clean` (seen) and
`randomized` (unseen) settings.

- **Team**: 
- **Base model**: [`robbyant/lingbot-vla-v2-6b`](https://huggingface.co/robbyant/lingbot-vla-v2-6b)
- **Submitted checkpoint**: `lora_tuned_1500` (LoRA rank 64 / alpha 128, 1500 steps, merged back into the base)
- **Motion planner used for evaluation**: cuRobo v0.7.8

> 中文版说明见 [`README.zh-CN.md`](README.zh-CN.md)。

## Training progression

The submitted model is the last step of a four-stage path, not a single training run.
All stages share the same data, environment and compliance constraints; what changed is
the training recipe.

| Stage | What it is | Devset | Full evaluation |
|---|---|---:|---|
| **1. Full-parameter SFT, 11000 steps** (`hf_final_11000`) | Official RoboTwin recipe, no hyper-parameter changes, trained on 50 clean tasks | — | clean **48.55%** (670/1380), randomized **24.00%** (180/750) |
| **2. Continued SFT to 6000 steps** (`lora6000_base`) | Same recipe shortened/continued, lr 1e-4 | 27/50 = 54.0% | — |
| **3. LoRA 1500 steps on that base** (`lora_tuned_1500`) | LoRA rank 64 / alpha 128 on `action_expert`, lr 5e-5 | **31/50 = 62.0%** | clean **62.8%** (314/500), randomized **37.0%** (185/500) |
| 4. Alternatives rejected | v3: batch 8→16, lr 5e-5→7e-5, first-25-step supervision + temporal smoothing · v2: vision noise 0.02 / state jitter 0.01 | 29/50 = 58.0%<br>23/41 ≈ 56% | — |

Key points about stage 1, which the later stages build on:

- It follows the official `train_full_sft.sh` launcher verbatim — **no hyper-parameter
  tuning**, only run scale (8 GPUs, `max_steps=11000`, `save_steps=1000`) is set externally.
- Measured throughput 19.3–22.1 s/it on 8 × W7900D, i.e. roughly 59–68 hours for 11000 steps.
- It required only infrastructure-level changes to run on this platform (checkpoint
  rotation, a resume switch, two environment variables), none of which affects training numerics.
- Its full-evaluation numbers are lower than stage 3 because it used fewer steps and was
  not tuned; the devset comparison in [`docs/reproduction_report.md`](docs/reproduction_report.md)
  is what identified LoRA fine-tuning as the better direction.

Detailed reports: [`docs/reproduction_report_hf11000_history.md`](docs/reproduction_report_hf11000_history.md)
(stage 1, including environment adaptation and evaluation-methodology analysis) and
[`docs/reproduction_report.md`](docs/reproduction_report.md) (stages 2–4).


## Results

Final submitted model (`lora_tuned_1500`):

| Setting | Tasks | Score | Success rate |
|---|---:|---:|---:|
| clean2clean (seen) | 50 / 50 | **314 / 500** | **62.8%** |
| clean2random (unseen) | 50 / 50 | **185 / 500** | **37.0%** |

Earlier full-parameter SFT model (`hf_final_11000`):

| Setting | Tasks | Score | Success rate |
|---|---:|---:|---:|
| clean2clean (seen) | 50 / 50 | 670 / 1380 | 48.55% |
| clean2random (unseen) | 50 / 50 | 180 / 750 | 24.00% |

Both were measured with a different episode count per task (see the reports); 10 episodes
per task for the final model.

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
    ├── reproduction_report.md                  Stages 2–4 (LoRA tuning)
    └── reproduction_report_hf11000_history.md  Stage 1 (11000-step full-parameter SFT)
```

## Model weights

The weights (~12 GB across 3 shards) exceed GitHub's file size limit and are published on
Hugging Face: **`Dwyane/lingbot-vla-lora_tuned_1500`**

| File | Size |
|---|---:|
| `model-00001-of-00003.safetensors` | 2.0 GB |
| `model-00002-of-00003.safetensors` | 5.0 GB |
| `model-00003-of-00003.safetensors` | 4.6 GB |

```bash
hf download Dwyane/lingbot-vla-lora_tuned_1500 --local-dir ./checkpoint
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
# Stage 1: 11000-step full-parameter SFT (the official recipe, run scale only)
GPU_COUNT=8 MAX_STEPS=11000 SAVE_STEPS=1000 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_11k_8gpu \
bash code/train.sh

# Stages 2-3: LoRA fine-tuning (see docs/reproduction_report.md);
# runs through the official launcher with code/configs/lingbotvla_cli.lora1500.yaml

# Evaluation
MODEL_PATH=<checkpoint-dir> GPU_COUNT=8 EPISODES=100 TASK_CONFIG=both bash code/eval.sh
```

## Documentation

| Document | Contents |
|---|---|
| [`README.zh-CN.md`](README.zh-CN.md) | This README in Chinese |
| [`code/README.md`](code/README.md) | Environment, required env vars, train/eval entry points, modifications to official code and their ownership |
| [`docs/reproduction_report.md`](docs/reproduction_report.md) | Training hyper-parameters, evaluation setup, devset comparison of four checkpoints, caveats |
| [`docs/reproduction_report_hf11000_history.md`](docs/reproduction_report_hf11000_history.md) | Stage 1: 11000-step full-parameter SFT — recipe, throughput, environment adaptation, evaluation-methodology analysis, code changes and their ownership |
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
