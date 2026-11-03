# 复现与调优说明（提交模型：lora_tuned_1500）

> 本文件对应**最终提交的 checkpoint `lora_tuned_1500`**。
> 早期版本 `hf_final_11000`（全参数 SFT）的复现细节见同目录
> `reproduction_report_hf11000_历史.md`（含环境适配、代码改动等通用说明，本文件不再重复）。

## 1. 基础信息

- **基座模型**：LingBot-VLA 2.0（`robbyant/lingbot-vla-v2-6b`）
- **提交 checkpoint**：`lora_tuned_1500`
  —— 在 6000 步基座（`lora6000_base`）上做 **LoRA 微调 1500 步**，再把 LoRA 权重 **merge 回基座**导出的 `hf_ckpt_merged`（约 12G，3 个 safetensors 分片 + config/tokenizer）。
- **代码版本 / Commit**（镜像自带仓库 HEAD，未切分支）：
  - lingbot-vla-v2：`951475ae1b1d87553e7dc47c97b53a3d695c0d13`
  - RoboTwin：`266f3aadf505a4f7fe9af0faa41a20f5f47cd123`

## 2. 训练复现

- **微调方式**：**LoRA**（`use_lora: true`），只训练 action_expert 的注意力投影层，基座其余参数冻结。
- **基座**：`lora6000_base`（官方 6B 基座经 6000 步 SFT 得到的基座权重）。
- **训练数据**：RoboTwin 2.0 的 **50 个 clean 任务**（`assets/training_data/robotwin.txt`）。
  **未使用任何 randomized 数据**（遵守比赛规则）。
- **训练配置**：`02_代码材料/configs/lingbotvla_cli.lora1500.yaml`（实际生效的 resolved 配置，与 checkpoint 目录内 `lingbotvla_cli.yaml` 一致）。
- **关键超参**：

| 项 | 值 |
|---|---|
| lora_rank / lora_alpha | **64 / 128** |
| lora_scope | `action_expert` |
| lora_target_modules | `q_proj,k_proj,v_proj,o_proj` |
| max_steps | **1500**（save_steps 1500） |
| 学习率 / 调度 | **5e-5** / cosine，min 2.5e-5，warmup 0 |
| global_batch_size / micro_batch | 8 / 8（梯度累积 1） |
| optimizer | adamw，weight_decay 0，max_grad_norm 1.0 |
| chunk_size | 50 |
| 图像增强 | `image_augment: true` |
| 混合精度 | 开启 |
| 参数分片 | FSDP2（shard_size 1 / replicate_size 1） |
| MoE | 开启（`use_moe: true`） |
| 随机种子 | 42 |
| 损失 | L1 flow-matching（`L1_fm`） |

- **训练入口**：使用官方训练 launcher，读取上述 `lingbotvla_cli.lora1500.yaml`
  （`use_lora: true`、`max_steps: 1500`、`output_dir` 指向 `lora_tuned_1500`）。
  训练日志/tensorboard 见该 `output_dir/runs/`。

## 3. 评测复现

- **评测环境**：
  - 仿真：NVIDIA RTX 4090D / RTX 6000D（SAPIEN 3.0.0b1 光追渲染 + OIDN 降噪）
  - 推理：NVIDIA RTX 6000D（4 个 WebSocket 策略服务，端口 13400–13403）
  - 软件栈：torch 2.7.0+cu128、cuRobo v0.7.8、Vulkan GPU 渲染
- **运动规划器**：**cuRobo v0.7.8**（非 MPLib；两者口径不可比，见 §6）
- **推理参数**：`--use_length 25 --use_bf16 True --use_fp32 False --use_compile False`
- **评测规模**：每个 setting **50 任务 × 10 集 = 500 集**
  （`clean` 与 `randomized` 各 500 集，全部 50 任务均已完成）
- **一键评测**：
  ```bash
  # clean（seen）
  WORKERS=4 PLANNER=curobo BASE_PORT=13400 bash run_eval_parallel.sh 10 demo_clean
  # randomized（unseen）
  WORKERS=4 PLANNER=curobo BASE_PORT=13400 bash run_eval_parallel.sh 10 demo_randomized
  ```
  （4 个仿真 worker 并行；策略服务用 `start_policy_servers.sh` 在推理机拉起）

## 4. 评测结果

| setting | 任务数 | 成绩 | 成功率 |
|---|---:|---:|---:|
| clean2clean（seen） | 50 / 50 | **314 / 500** | **62.8%** |
| clean2random（unseen） | 50 / 50 | **185 / 500** | **37.0%** |

- 逐任务明细见 `01_评测结果/results.json` 与 `robotwin_handoff/评测报告_clean与randomized.md`。
- **任务族差异 >> setting 差异**：`place_*` 摆放/对齐类普遍 60–90%，堆叠/开合/排序类普遍 0–40%；
  unseen 随机化整体下降约 26 个百分点，但个别任务（`stack_bowls_three`、`place_cans_plasticbox` 等）反而上升。

## 5. 调优说明（四版权重 devset 对比）

devset = lift_pot / place_shoe / hanging_mug / stack_bowls_three / open_microwave，每任务 10 集 = 50 集。

| 权重 | 训练配置 | devset 成绩 | 判定 |
|---|---|---|---|
| base6000 | 3000→6000 累计，lr 1e-4 | 27/50 = 54.0% | 基线 |
| **tuned1500（提交）** | 6000 基座 + LoRA 1500 步，lr 5e-5 | **31/50 = 62.0%** | ✅ **最优** |
| v3 | batch 8→16、lr 5e-5→7e-5、前 25 步监督 + 时序平滑 | 29/50 = 58.0% | 不如 tuned1500 |
| v2 | + 视觉噪声 0.02 / 状态扰动 0.01 | 23/41 ≈ 56% | 无收益，放弃 |

**结论**：`lora_tuned_1500` 是四版里最优（devset 62%，比基线高 8 个点）；视觉噪声增强无收益；v3 的 lr 提高（5e-5→7e-5）可能是其变差的主因。故提交 tuned1500。

## 6. 口径与已知限制

- **规划器口径**：本方案自测用 **cuRobo**。若官方复测使用平台镜像默认的 **MPLib**（平台镜像不含 cuRobo 时自动回退），
  同一权重的绝对成功率会偏低——这是**评测工具差异**，非策略差异。最终以**官方复测**为准（赛题 §4.2）。
- **每任务 10 集**：单任务统计噪声较大（10 集时约 ±30pp），整体均值较稳。赛题允许未测满 100 次如实上报。
- **未使用 randomized 数据训练**：合规（`demo_randomized` 仅用于评测，不参与训练）。
- **通用复现细节**（环境适配、checkpoint 轮转、续训开关等代码改动）见 `reproduction_report_hf11000_历史.md`。
