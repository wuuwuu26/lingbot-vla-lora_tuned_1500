# 02_代码材料

本目录包含初赛评测结果对应的实现代码，用于主办方复现训练与评测流程。

> **⚠️ 提交模型说明（请先读）**：本次最终提交的 checkpoint 为 **`lora_tuned_1500`**
> （在 6000 步基座 `lora6000_base` 上做 **LoRA rank64/alpha128 微调 1500 步**，再 merge 回基座）。
> - 训练配置：`configs/lingbotvla_cli.lora1500.yaml`
> - 评测结果：`01_评测结果/results.json`（clean **62.8%** / randomized **37.0%**，各 50 任务 × 10 集）
> - 评测规划器：**cuRobo v0.7.8**
>
> 本 README 下文描述的 `train.sh` / **全参数 SFT 11000 步**流程，是团队早期版本
> `hf_final_11000` 的训练/评测流程，**保留作为代码材料与复现参考**。最终提交模型以
> `lora_tuned_1500` 为准，两者模型与评测口径（规划器）不同，请勿混用。

## 目录结构

```text
02_代码材料/
├── README.md          本文件
├── train.sh           训练入口（RoboTwin 2.0 Aloha-AgileX 50 任务 clean 数据）
├── eval.sh            评测入口（clean / randomized 双 setting，各 50 任务）
├── configs/           训练与评测配置（训练配置 + 模型配置 + robot config，见下）
└── src/               改动留证：上游 diff 与补丁脚本（见「对官方代码的修改」）
```

`configs/` 逐文件对应目录要求的三类配置：

| 文件 | 类别 | 内容 |
|---|---|---|
| `lingbotvla_cli.resolved.yaml` | **训练配置** | 本次训练实际生效的完整配置（含数据、并行、优化器等） |
| `lingbotvla_cli.official.yaml` | 训练配置（基线） | 官方原始配置，供与上面那份逐项比对 |
| `vla_robotwin.yaml` | **模型配置** | `model:` 段（`config_key: LingbotVLAV2Config`、MoE、tokenizer 等） |
| `robotwin.yaml` | **robot config** | 机器人状态/动作空间映射（Aloha-AgileX 双机械臂） |

## 环境依赖

- **硬件**：8 × AMD Radeon W7900D 48 GB（gfx1100）
- **软件栈（实测值）**：ROCm 7.2.1 + PyTorch **2.9.1+rocm7.2.1** + Triton 3.5.1 +
  flash-attn 2.8.4（Triton 回退实现，镜像内**无** `flash_attn_2_cuda` HIP 扩展）+
  Python **3.12.3**
- **运动规划器**：MPLib（本镜像**不含 cuRobo**，故走 MPLib 回退；其对本自测成功率
  的影响见 `../04_复现与调优说明/reproduction_report.md` §3）
- **主要依赖**：`lingbot-vla-v2`、`RoboTwin 2.0`、`LeRobot`、`MPLib`

### 训练必设的两个环境变量（`train.sh` 已内置）

| 变量 | 不设的后果 |
|---|---|
| `AITER_TRITON_ONLY=1` | 镜像内 triton 为 3.5.1，而 aiter 的 gluon 内核要求 ≥ 3.6.0；`import flash_attn` 经 Triton 回退路径导入 aiter 时抛 `RuntimeError`。`qwenvl_in_vla.py` 中的 `is_flash_attn_available()` 守卫会失效（transformers 仅检查包是否存在，不实际导入），故训练在建模阶段必崩。镜像自带的推理脚本已设此变量，训练脚本未设。 |
| `TMPDIR=/var/tmp/lingbot` | 首次 forward 时 AMD COMGR 为**每个 rank** 写一个 543 MB 的 `hipfatbin`，8 卡合计约 4.3 GB，超过 `/tmp` 的 4 GB tmpfs 上限，报 `LLVM ERROR: IO failure on output stream: No space left on device`。这是**确定性故障**，非偶发。 |

这两项都不改变训练数值，只是让官方配置能在本镜像上跑起来。

## 基础模型与数据

- **基础权重**：`robbyant/lingbot-vla-v2-6b`（赛题指定，训练必须从此初始化）
  - `train.sh` 内置合规自检：若基座路径疑似 `-robotwin` 后训练权重则直接退出
- **训练数据**：RoboTwin 2.0 中 Aloha-AgileX 的 50 个任务 clean 数据，每任务 50 条
  - 清单文件：`data/robotwin_demo_clean_joint_v30.txt`（50 行）
    - 该清单里的路径写作 `/workspace/RoboTwin/...`，官方 launcher
      （`training/train_full_sft.sh:105-108`）会自动 `sed` 归一化为 `/RoboTwin/...`，
      **无须手工修改**（详见 `../04_复现与调优说明/reproduction_report.md` §2）
  - **禁止**使用 randomized 数据参与训练，也**禁止**使用任何外部数据
- **动作空间**：55 维统一表示（14 臂关节 + 14 末端位姿 + 2 夹爪 + …）

## 训练

```bash
GPU_COUNT=8 MAX_STEPS=11000 SAVE_STEPS=1000 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_11k_8gpu \
bash train.sh
```

`train.sh` 是官方 launcher（`training/train_full_sft.sh`）的等价包装，**不追加任何
`--train.*` 参数**，训练超参完全由官方脚本与 `configs/` 决定。

关键环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPU_COUNT` | 8 | 训练使用的 GPU 数 |
| `MAX_STEPS` | 11000 | 训练步数 |
| `SAVE_STEPS` | 1000 | checkpoint 保存间隔 |
| `MICRO_BATCH_SIZE` | 16 | 单卡 micro batch |
| `GLOBAL_BATCH_SIZE` | 256 | = micro 16 × 8 卡 × 梯度累积 2 |
| `TEACHER_MODE` | full | 深度 + 未来帧 + DINO-video 双路教师蒸馏（官方推荐复现配置） |
| `OPTIMIZER` | adamw | 优化器 |
| `DATA_LIST` | clean 50 任务清单 | 训练数据列表 |
| `ENABLE_RESUME` | false | 续训开关（见「对官方代码的修改」第 2 条） |

`train.sh` 内置两项合规自检：清单必须恰好 50 条，且不得出现 `random` 字样，否则直接退出。

> **微调方式（本节指 hf_final_11000 历史版本）**：全参数 SFT。官方 launcher 内部硬编码
> `use_lora false` 与 `train_expert_only false`。**最终提交的 `lora_tuned_1500` 改用 LoRA**
> （`use_lora: true`，rank64/alpha128，scope `action_expert`，配置见
> `configs/lingbotvla_cli.lora1500.yaml`），详见顶部「提交模型说明」。

## 评测

```bash
MODEL_PATH=<checkpoint>/hf_final_11000 \
GPU_COUNT=8 EPISODES=100 TASK_CONFIG=both bash eval.sh
```

> 上例为**赛题标准规模**（每任务每 setting 100 次）。本次提交的 `results.json` 受算力
> 窗口限制，实际按 `EPISODES=30`（clean 的前 42 个任务）与 `EPISODES=15`（其余 58 个
> 任务，含全部 randomized）跑出；口径、原因与逐 setting 的成功率见
> `../04_复现与调优说明/reproduction_report.md` §3「评测规模」。《比赛要求》§4.2：
> 「未测满可提交，最终以官方复测为准」。

评测链路：每张 GPU 启动一个模型 server（端口从 `13400` 起），任务由动态队列分配，
每个任务完成后写入 `done/<task_config>__<task>.done`，中断后可用 `--resume` 续跑。

`eval.sh` 暴露两个推理参数（默认值与官方 deploy 脚本逐字一致）：
`INFER_PRECISION`（`bf16` | `fp32`）与 `USE_LENGTH`（动作 chunk 执行长度）。
其施加方式不修改任何官方文件（利用官方脚本第 7 行的 `LINGBOT_VLA_PYTHON` 覆盖点，
运行时生成薄包装器并预检），详见 `../04_复现与调优说明/reproduction_report.md` §3。

> **时间预算（实测，非估算）**：8 × W7900D 上，本次全量自测（100 个任务 × 混合
> 30/15 episodes = **2130 个请求 episode**）实测耗时 **81.4 GPU·小时 / 10.2 小时墙钟**，
> 即 **2.29 min / 请求 episode（GPU 时间）**。按此外推，赛题标准的
> 100 任务 × 100 episodes × 2 setting = **10000 episodes 约需 48 小时墙钟（8 卡）**，
> 请据此规划算力窗口 —— 这是**下限估计**，因为该值是在本次自测的成功率水平
> （clean 48.6% / random 24.0%）下测得的，而**失败 episode 必跑满步数上限、
> 成功率越低越慢**；策略越强，此值越小。务必用 `--resume` 避免重复计算。
> （实测依据：`<run-dir>/task_times.tsv` 的逐任务秒数求和。）

### 生成提交用 results.json

```bash
python tools/results_tool.py collect --run-dir <runtime>/outputs/<run-name> --team-id <团队ID>
python tools/results_tool.py validate 01_评测结果/results.json
```

`collect` 会解析 `<run-dir>/logs/eval_<task_config>__<task>.log` 中的
`Final success rate: N/M` 行；若任一 setting 缺任务则拒绝生成，避免提交不完整结果。

## 对官方代码的修改

### 归属说明（哪些是镜像自带、哪些是本方案改动）

本平台镜像是**预构建**的，其自带的两个源码仓库**相对上游官方 commit 已经有改动**
（AMD ROCm 移植、LoRA 支持等）。这些改动**不是本方案所为**。区分依据是**文件 mtime**：

| 批次 | mtime | 归属 |
|---|---|---|
| 镜像自带改动 | **2026-09-15 01:15 UTC**（与镜像内 `/RoboTwin/README.md`、`/opt/robotwin-env` 的构建时间落在同一分钟内） | 镜像 |
| 本方案改动 | **2026-10-12 04:31–04:35 UTC** | 本方案 |

镜像内另自带一份复现说明 `RoboTwin_ROCm_Reproduction.ipynb`（同日 05:18），其中逐条列出
`ROBOTWIN_DISABLE_CUROBO=1` / `ROBOTWIN_EE_PLANNER=mplib` / `PYOPENGL_PLATFORM=egl`
等环境钩子 —— 正是下述 RoboTwin 改动所实现的那一套，可交叉印证归属。

完整 diff 留证于 `src/`：

| 文件 | 内容 |
|---|---|
| `src/RoboTwin.upstream_diff.patch` | RoboTwin 相对上游 `266f3aa` 的**全部**改动（**均为镜像自带**） |
| `src/lingbot-vla-v2.upstream_diff.patch` | lingbot-vla-v2 相对上游 `951475a` 的**全部**改动（镜像自带 + 本方案 1 处） |
| `src/patch_checkpoint_rotation.py` | **本方案**对 checkpoint 轮转的补丁（幂等，原文件备份为 `.orig`） |

复现报告所引数字的分析脚本（均可对 `eval.sh` 的产物直接重跑）：

| 文件 | 生成的结论 |
|---|---|
| `src/eval_diagnostics.py` | 报告 §3「测量口径」与「失败形态」：请求/实际/跳过/成功/失败、两种口径成功率、失败 episode 的终止步数 |
| `src/expert_check_stats.py` | 报告 §3「证据二」：专家 `plan_success` / `task_success` 统计与跨批次翻转率 |
| `src/official_corr.py` | 报告 §3「证据三」：与官方 README 逐任务成功率的相关性 |

> `src/expert_check_stats.py` 读到的是 `episode_info` 里的 `task_success` 字段，
> 该字段**不是**评测器的判据（报告 §3「口径说明二」有实测反例），只用于观察
> 同字段的跨批次稳定性。

### 本方案的改动（共 2 处）

**1. `experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2/tasks/vla/train_lingbotvla.py`（+34 行）**

新增模块级辅助函数 `_rotate_checkpoints()`，并在 3 处 `Checkpointer.save(...)` 调用点
**之前**各插入一行调用。**未改动模型结构、超参、损失函数、数据管线或任何训练数值逻辑。**

*原因*：本平台工作区持久卷为 100 GB，而本配置下单份全量 DCP checkpoint 实测 **72 GB**
（`optimizer/` 48 GB + `model/` 24 GB）。官方 `Checkpointer.save` 只创建目录后写入，
**从不清理旧 checkpoint**，也没有 keep_last 之类的参数，因此按 `save_steps=1000` 训练时
**第二次保存必然写满磁盘**（剩余 28 GB < 需要 72 GB）并中断训练。

*策略*：写入新 checkpoint 前，保留最新一份旧 checkpoint 的 `model/`（24 GB）作兜底，
更早的整份删除 —— 峰值占用 24 + 72 = **96 GB ≤ 100 GB**（已在 step 2000 实测验证）。

**2. `experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh`（1 行）**

```diff
-  --train.enable_resume false \
+  --train.enable_resume "${ENABLE_RESUME:-false}" \
```

*原因*：官方 launcher 把该参数**硬编码**为 `false`，不暴露任何开关，无法从外部续训；
本平台单次会话受算力窗口限制，11000 步训练需跨会话续跑。

*影响*：**默认取值不变** —— 未设 `ENABLE_RESUME` 时展开为 `false`，与官方原命令行
**逐字等价**，故对该变量不作设置的复现者**零行为差异**。

### RoboTwin 官方评测代码

**本方案未修改 RoboTwin 的任何文件（0 个文件）。**

但须说明：**镜像自带**的 RoboTwin 相对上游已有 9 个文件的改动（+279 / −74），其中

| 文件 | 改动量 | 改了什么 |
|---|---:|---|
| `scripts/eval_policy_xpolicylab.py` | +191 | XPolicyLab 评测协议适配 |
| `envs/robot/planner.py` | +72 | 运动规划器 cuRobo → MPLib 回退 |
| `envs/robot/robot.py` | +42 | 同上，机器人侧接线 |
| `envs/_base_task.py` | +5 | **渲染兼容**：光线追踪降噪器 `oidn` → `none`，并压低 sapien 日志级别 |
| 其余（含 `assets/`、`data/process_stuck.py` 的删除） | — | 体积裁剪 |

这些正是该镜像的 **ROCm / MPLib 适配**（本平台不含 cuRobo），**非本方案所为**，
本方案对其**原样使用**，未做二次修改。完整 diff 见
`src/RoboTwin.upstream_diff.patch`，供评委核验。

> **须如实披露（不作掩饰）**：`envs/_base_task.py` 那 5 行把光线追踪降噪器由 `oidn`
> 改为 `none`，会改变相机渲染结果 —— 即**策略实际看到的图像与上游默认不同**，
> 不属于零副作用的纯适配。该改动由镜像预置（mtime `2026-09-15 01:15:31`，
> 早于本方案全部工作），本方案按镜像自带配方原样使用。
> 附表为报备而不是辩解：若评委认为该渲染口径需要还原，本方案可提供移除该改动后的复测。

## 复现说明

训练方式、参数配置、调优思路与复现步骤见
`../04_复现与调优说明/reproduction_report.md`。
