# 复现与调优说明

## 1. 基础信息

- **使用模型版本**：LingBot-VLA 2.0（`robbyant/lingbot-vla-v2-6b`）
- **代码版本 / Commit**（镜像自带仓库的 HEAD，本方案未切换分支或 commit）：
  - lingbot-vla-v2：`951475ae1b1d87553e7dc47c97b53a3d695c0d13`
  - RoboTwin：`266f3aadf505a4f7fe9af0faa41a20f5f47cd123`
- **提交 checkpoint 名称**：`hf_final_11000`（全参数 SFT，`max_steps=11000` 步导出的 HF 权重）

## 2. 训练复现

- **使用数据集**：RoboTwin 2.0 Aloha-AgileX 50 个任务 clean 数据，每任务 50 条
  （清单：`data/robotwin_demo_clean_joint_v30.txt`，共 50 行）
- **是否使用 randomized 数据训练**：否
- **一键训练脚本**：`02_代码材料/train.sh`
- **训练配置文件**：`02_代码材料/configs/lingbotvla_cli.resolved.yaml`
  （训练时**实际生效**的 resolved 配置，含已填充的 `align_params` 教师蒸馏分支）。
  官方 launcher 的原始模板同时附于 `configs/lingbotvla_cli.official.yaml`，便于逐项对照。
- **robot config**：`02_代码材料/configs/robotwin.yaml`（官方提供，未修改）
- **VLA 任务配置**：`02_代码材料/configs/vla_robotwin.yaml`（官方提供，未修改）

> 注：训练侧实际读取的路径位于
> `experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2/configs/` 下，
> 上表列的是提交包内与之逐字一致的副本。
>
> **复现提示（数据清单的前缀归一化）**：镜像自带的
> `data/robotwin_demo_clean_joint_v30.txt` 里 50 条路径写作
> `/workspace/RoboTwin/data/lerobot/<task>_joint_v30`，而本镜像的 RoboTwin 根实际在
> `/RoboTwin`（该目录经软链指向持久卷 `/models/robotwin-persistent/data/lerobot`），
> 故**清单里的字面路径并不存在**。这不是故障：官方 launcher
> `training/train_full_sft.sh:105-108` 会检测到 `/workspace/RoboTwin/` 前缀并
> `sed` 替换为 `/RoboTwin/` 后再用。因此**复现者无须修改清单，也无须手工建软链**；
> 但若把 RoboTwin 装在 `/RoboTwin` 以外的位置，该替换会失配（属官方 launcher 行为），
> 需自行把 `DATA_LIST` 指向一份已归一化的清单。

### 训练命令

实际使用官方 launcher `training/train_full_sft.sh`，并**必须**先设两个环境变量
（原因见 §4「环境适配」）：

```bash
cd /RoboTwin
export AITER_TRITON_ONLY=1
mkdir -p /var/tmp/lingbot && export TMPDIR=/var/tmp/lingbot

TEACHER_MODE=full GPU_COUNT=8 OPTIMIZER=adamw \
MICRO_BATCH_SIZE=16 GLOBAL_BATCH_SIZE=256 TIMEOUT_SECONDS=0 \
MAX_STEPS=11000 SAVE_STEPS=1000 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_11k_8gpu \
bash experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh
```

`02_代码材料/train.sh` 是该命令的等价包装（含上述环境变量与合规自检）。

### 训练超参

| 项 | 值 |
|---|---|
| 微调方式 | **全参数 SFT**（官方 launcher 硬编码 `use_lora false` / `train_expert_only false`） |
| lora_rank / lora_alpha | 不适用（非 LoRA） |
| lora_scope | 不适用 |
| lora_target_modules | 不适用 |
| 学习率 / 调度 | 1e-4 / cosine，min 5e-5 |
| global_batch_size | 256 |
| micro_batch_size | 16 |
| 梯度累积步数 | 2 |
| max_steps | 11000（save_steps 1000） |
| chunk_size | 50 |
| optimizer | adamw |
| 梯度检查点 | 开启 |
| 参数分片 | `enable_full_shard true`（FSDP2，shard_size=8 / replicate_size=1） |
| 教师蒸馏 | `TEACHER_MODE=full`（深度 + 未来帧 + DINO-video 双路蒸馏，官方推荐复现配置） |
| 随机种子 | 沿用官方配置默认值，未改动 |

> 训练速度实测：稳态 **19.3 – 22.1 s/it**（8× W7900D，4 次运行的区间）。
> 其中「从零开始」运行的两次为 19.3 / 20.7 s/it，「从 DCP checkpoint 续训」
> 的两次为 21.7 / 22.1 s/it，每次运行**内部极稳**（±0.5 s），差异来源未定位到
> 具体原因（已逐项排除 GPU 降频、CPU 抢占、数据饥饿、NFS 写延迟、教师前向耗时）。
> 11000 步对应约 **59 – 68 小时**。
> 官方复现指南的 sweep 配置实测为 24.682 s/it，本方案各次运行均快于该值。

## 3. 评测复现

- **一键评测脚本**：`02_代码材料/eval.sh`
- **checkpoint 加载方式**：通过 `MODEL_PATH` 指向导出的 `hf_ckpt` 目录，由
  `run_clean_benchmark.py` 拉起模型 server 后闭环评测
- **clean setting 评测命令**：

```bash
MODEL_PATH=<hf_ckpt> GPU_COUNT=8 EPISODES=30 TASK_CONFIG=demo_clean bash eval.sh
```

- **randomized setting 评测命令**：

```bash
MODEL_PATH=<hf_ckpt> GPU_COUNT=8 EPISODES=15 TASK_CONFIG=demo_randomized bash eval.sh
```

> 以上 `EPISODES` 为**本次实际执行值**，非赛题标准值（标准为 100，见下「评测规模」）。
> 本次两个 setting 实际由同一条 `TASK_CONFIG=both --episodes 15 --resume` 续跑完成：
> clean 的前 42 个任务在此之前已按 30 episodes 跑完并被 `--resume` 跳过，其余按 15 跑。
> 两个 setting 也可用 `TASK_CONFIG=both` 一次跑完。

- **评测环境**：
  - GPU：8 × AMD Radeon W7900D 48 GB（gfx1100）
  - 软件栈：ROCm 7.2.1 + PyTorch 2.9.1+rocm7.2.1 + Triton 3.5.1 + flash-attn 2.8.4
    （Triton 回退实现，无 `flash_attn_2_cuda` HIP 扩展，需 `AITER_TRITON_ONLY=1`）
  - 运动规划器：MPLib（`ROBOTWIN_DISABLE_CUROBO=1`、`ROBOTWIN_EE_PLANNER=mplib`）
  - 推理精度：**BF16**（镜像自带的 `launch_official_server.sh` 固定传入
    `--use_bf16 True --use_fp32 False`；本方案对此的处置见下）

> **关于推理精度（本节措辞已按《比赛要求》更正）**：官方发布验证使用 **FP32** 推理，
> 而本镜像自带的 `launch_official_server.sh` 第 42–43 行把精度**硬编码**为
> `--use_bf16 True --use_fp32 False`（`run_clean_benchmark.py` 仅暴露 `--use-compile`，
> **没有精度开关**）。关键点是：**BF16 不是镜像强加的限制，而是本方案当前的取值** ——
> 《比赛要求》§2.4（三）明文把「推理精度调整（bf16/fp32）」列为**允许的推理策略**，
> §4.2 又明确初赛仿真评测**不限制单次推理时间与显存占用**。故本方案既有能力、也有
> 依据改用 FP32 与官方同口径。
>
> **本方案的取值与依据**：`eval.sh` 最终取 **BF16**（`INFER_PRECISION` 默认 `bf16`）。
> 理由有二：① 这是镜像自带 `launch_official_server.sh` 的默认值，不引入任何与镜像既有
> 行为不同的因素；② 算力窗口内**未安排** BF16 / FP32 的对照实测 —— 一次对照需占用全部
> 8 卡重跑一轮完整评测，会直接挤占正式评测的窗口。既未实测，**本节即不对该取值作任何
> 优劣声明**：下列全部自测与正式提交数据，都是在 BF16 下产生的。
>
> `eval.sh` 仍保留 `INFER_PRECISION=fp32` 开关。若评委希望与官方发布口径（FP32）对齐，
> 设该变量即可 —— 《比赛要求》§2.4（三）明确允许推理精度调整，且 §4.2 规定初赛仿真
> 评测不限制单次推理时间与显存占用。
>
> **须特别注意（自包含要求）**：§4.2 规定「**最终测评的时候会使用选手自己 `eval.sh`
> 中的推理参数**」，即本节所列推理参数**就是评委复测所用参数**。无论评委是直接运行本
> 方案的 `eval.sh`，还是按 README 披露的参数复现，`eval.sh` 都必须是**自包含**的 ——
> 它会在评委的全新镜像实例上运行，任何对镜像内官方脚本（如
> `launch_official_server.sh`）的改动**不会随之带过去**，必须由 `eval.sh` 自身在运行时
> 施加并校验。

**推理参数的自包含施加方式（本方案实际实现）**：官方 `launch_official_server.sh` 把
推理参数**写死在命令行**里（`--use_length 25 --use_bf16 True --use_fp32 False`），
且不暴露任何开关。本方案**没有修改该文件**，而是利用了它第 7 行自带的解释器覆盖点：

```bash
python_bin=${LINGBOT_VLA_PYTHON:-/opt/robotwin-env/bin/python}   # 第 7 行
...
exec "$python_bin" -m deploy.lingbot_vla_v2_policy \
     --use_length 25 --use_bf16 True --use_fp32 False ...
```

`LINGBOT_VLA_PYTHON` 在整个镜像中**仅此一处引用、且默认未设置**。`eval.sh` 在运行时
生成一个薄包装器赋给该变量，由它重写上述三个参数后再 `exec` 真正的解释器。该包装器
另带一个 `--print-rewrite` 预检模式：`eval.sh` 在正式评测前，用官方脚本的真实命令行
喂给它，`grep -F` 校验重写结果符合预期，**不通过则直接退出**，不会静默用错参数。

这样做的好处是：① **零改动官方文件**，在评委全新镜像上同样成立（无持久化副作用）；
② **默认取值下重写结果与官方原命令行逐字等价**（已实测），故默认路径不引入任何行为
差异；③ 实际生效值还会以 `[eval-wrapper] ...` 打点写入 `server_gpuN.log`，**事后可核验**。

`eval.sh` 暴露的两个推理参数：

| 环境变量 | 取值 | 规则依据 |
|---|---|---|
| `INFER_PRECISION` | `bf16` \| `fp32`（默认 `bf16`） | §2.4（三）推理精度调整 |
| `USE_LENGTH` | 正整数（默认 `25`） | §2.4（一）动作 chunk 执行长度调整 |

与 §4.2 中 `LINGBOTVLA_TRAINING_CONFIG` 的处理同款：两者都由 `eval.sh` 在运行时自行
解析并校验，均不依赖镜像内的既有状态。

- **评测必做：显式指定训练配置**。本镜像的 `launch_official_server.sh` 总会
  `export LINGBOTVLA_TRAINING_CONFIG`；对**非官方默认路径**传入的 checkpoint，
  它会指向 repo 内的 `training/lingbotvla_cli.yaml`，而那份配置的 `align_params`
  为**空**。本方案经 `TEACHER_MODE=full` 蒸馏，checkpoint 含 depth/video align 分支
  参数（`depth_align_embs` / `future_depth_align_embs` / `current_video_align_embs` /
  `future_video_align_embs`，全模型共 1708 个张量），而 deploy 侧是以
  **`strict=True`** 加载权重（`deploy/lingbot_vla_v2_policy.py:369`），架构不符会
  **直接抛错、评测无法启动**。因此评测本方案 checkpoint 时必须把该变量指向与训练
  一致的 resolved 配置：

  ```bash
  export LINGBOTVLA_TRAINING_CONFIG=<训练输出目录>/lingbotvla_cli.yaml
  ```

  `02_代码材料/eval.sh` 已内置该变量的自动解析与校验：优先在 checkpoint 目录内查找
  `lingbotvla_cli.yaml`，其次查其上级目录；两者都没有则**报错退出**，不会静默使用
  错误配置。

### 评测规模

| setting | 任务数 | 每任务 episodes | 合计 |
|---|---:|---:|---:|
| clean2clean | 50 | **42 个任务 × 30 + 8 个任务 × 15** | **1380** |
| clean2random | 50 | 15 | **750** |

> 标准要求为每任务、每 setting 各 100 次；《比赛要求》§4.2 明确「未测满可提交，
> 最终以官方复测为准」，故上表按**实际完成量**填写（受算力窗口限制时如实记录）。
>
> **口径不一须说明**：clean 的前 42 个任务按 30 episodes 跑完后，因剩余算力需容纳
> 尚未开始的 50 个 randomized 任务，于 2026-10-15 07:40 UTC 将其余 58 个任务
> （clean 8 个 + randomized 50 个）降为 15 episodes 续跑。原因是
> `scripts/run_clean_benchmark.py:162` 的完成标记只按 `<setting>__<task>` 命名、
> **不含 episodes 数**，故无法"在已有成果上补跑更多轮"，只能二选一：整批重来，
> 或让剩余任务按更小的量跑完。本项目评测工具
> `tools/results_tool.py` 的 `collect` 是**全有或全无**（任一 setting 缺任务即拒绝
> 生成 `results.json`，见该文件 `:84-97`），没有部分提交的降级路径，因此选择后者。
> 该口径差异**不影响**下方成功率的分母定义（各任务均以自身实际请求数为分母），
> 但会增大 15-episode 任务的逐任务统计噪声。官方复测不受此影响。


### 自测结果的解读边界（重要）

本平台镜像**不含 cuRobo**（`import curobo` 抛 `ModuleNotFoundError`），镜像自带的
`scripts/run_clean_benchmark.py:251-252` 因此**硬编码**了 MPLib 回退
（`ROBOTWIN_DISABLE_CUROBO=1`、`ROBOTWIN_EE_PLANNER=mplib`）。这一环境差异会显著影响
自测成功率的**绝对水平**，故本方案的自测数字**只能作方向性参考**。下面三条依据均可查证。

**证据一：官方 93.52% 与本方案自测并非同一配方，故两者不可直接比较。**
官方 README（`reference/lingbot-vla-v2_README.md`）明确记载：

| 项 | 官方 93.52% 对应设置 | 本方案 |
|---|---|---|
| 权重 | `lingbot-vla-v2-6b-robotwin`（官方**已后训练**的权重） | 基座 `lingbot-vla-v2-6b` + 自行 SFT |
| 训练步数 | `global_step_50000`（README 部署命令所用路径） | `max_steps=11000` |
| 训练数据 | **clean 与 randomized 一起训**<br>（README 第 138 行原文：*trained with clean and randomized data together*） | 仅 `demo_clean` |

其中「训练数据」一项，本比赛《比赛要求》的增广黑白名单**禁止**使用
`demo_randomized` 参与训练，故该**配方差异在规则内无法消除**；训练步数则受算力窗口限制。
因此上表两个数字**并非同一配方下的对照结果**，直接相减不构成"调优增益"或"能力差距"。

> **关于评测环境**：官方 README **未记载** 93.52% 的评测硬件 / 规划器 / 推理精度。
> README 中唯一出现的 `RTX 4090D` 用于描述**单次推理延迟**（约 130 ms），与评测环境
> 无关；官方 `launch_official_server.sh:41-44` 的推理默认值为
> `--use_bf16 True --use_fp32 False`，与本方案 `eval.sh` 默认一致。
> 故**不可**据此推断官方在 FP32 或 cuRobo 环境下测得该数字。

**证据二（更关键）：本镜像上"专家演示本身"就跑不通。** 评测脚本的 `--expert-check`
会在每个 episode 前跑一遍 ground-truth 专家演示。统计本次评测产出的 `episode_info`
日志（即 `eval.sh` 运行后落在 `<run-dir>/episode_info/` 的那批；本次为
`both_100x100` / `both_100x50` / `both_100x30` 三次运行，按
`(setting, task, seed)` 去重后 **1926 个唯一 episode**）：

| 指标 | 实测 | 应有值 |
|---|---:|---:|
| 专家 `plan_success` | **765–770 / 1926 = 39.7%–40.0%** | ≈100% |
| 专家 `task_success` | **679–690 / 1926 = 35.3%–35.8%** | ≈100% |

**给出区间而非单点值，是因为该判定在本镜像上不可复现。** 同一 `(setting, task, seed)`
在两次运行中会被重新判定，实测 **190 个跨批次可比较的 key 中有 17 个 `plan_success`
翻转、20 个 `task_success` 翻转（8.9% / 10.5%）**。原因是 MPLib 的规划器使用随机采样
（RRT），同一确定性种子并不给出确定性规划结果。因此**任何单次统计的精确小数位都无
意义**，上表区间即"取每 key 首次观测"与"取末次观测"两种口径的包络。

按任务拆分（全量 50 个任务 × 2 个 setting = 100 个组合），其中 **35 个组合的专家
`plan_success` 为 0%**：**15 个 `demo_clean` 任务**（`adjust_bottle`、
`blocks_ranking_rgb`、`blocks_ranking_size`、`grab_roller`、`handover_block`、
`handover_mic`、`move_playingcard_away`、`place_bread_skillet`、`place_burger_fries`、
`place_dual_shoes`、`place_mouse_pad`、`place_phone_stand`、`put_object_cabinet`、
`stack_blocks_three`、`stack_blocks_two`）与 **20 个 `demo_randomized` 任务**；
其余组合为 20%–100%。**专家在 `demo_clean` 的确定性种子上本应接近
100%**，故可判定这是 MPLib 回退导致的规划失败，而非任务本身不可解。
（本次评测日志中反复出现 `IK Failed! Cannot find valid solution.` 与 `planning failed`；
在最终 run 的 100 个日志文件里，`IK Failed` 出现 **1174** 次、`planning failed`
**1291** 次、`RRTConnect Failed` **117** 次、
`Only 0 cluttered objects are placed on the table` **52** 次。复核：
`grep -c "IK Failed" <run-dir>/logs/eval_*.log`。）

> **口径说明一（n 的来历）**：上表 n = 三次运行**全部落盘后**按
> `(setting, task, seed)` 去重的唯一 episode 数（1926），已是最终值。它小于本次评测
> 的请求 episode 数（2130），差额来自 `UnStableError` 跳过（见 §3「测量口径」）与
> 前两次运行只跑了部分任务。
>
> **口径说明二（`task_success` 字段不是评测器的判据）**：`episode_info` 里的
> `task_success` 与本 harness 评定成功所用的判据**不是同一个**。实测
> `beat_block_hammer`(clean)：日志 `Final success rate: 22/30`，而该文件里
> `task_success` 为真的只有 16 条（`plan_success` 17 条，两者取与/取或均凑不出 22）；
> 全量上该字段只能解释 clean 的 36.7%，而日志口径是 51.0%。故上表 `task_success`
> **只用于观测"同一字段在不同批次间是否稳定"**，**不可**当作专家或策略的成功率来读。
> 本报告其余各处（§3「测量口径」「失败形态」、§5）的成功率一律取自评测日志，
> 不受此影响。
>
> 上表全部数字由 `02_代码材料/src/expert_check_stats.py` 生成，可复核：
>
> ```bash
> python src/expert_check_stats.py <run-dir-100> <run-dir-50> <run-dir-30>
> ```
>
> 去重键为 `(task_config, task_name, seed)`；传入多个 run-dir 时按命令行顺序视为
> 先后批次，翻转率即由此检测。

> **因此须特别注意三点**：
>
> 1. **专家成功率不构成策略成功率的"上限"。** 二者走的是 MPLib 的不同能力 ——
>    专家的 `plan_success` 来自**逆运动学 / 碰撞规划**（`robot.py:plan_path`，
>    本镜像无 cuRobo 时回退 MPLib，日志中的 `IK Failed` 即出自此处）；本方案策略
>    按 `action_type=joint` 输出**关节目标**，经 `_base_task.py:take_action` 的
>    `qpos` 分支**直接施加**，不做 IK 与碰撞规划，仅调用 MPLib 的 **`TOPP()`**
>    做时间最优参数化（失败时回退固定 50 步）。因此两条路径的失败模式并不同源。
>    数据亦印证：本方案策略在 `adjust_bottle`(clean) 上得 **19/30 = 63.3%**、在
>    `grab_roller`(clean) 上得 **30/30 = 100%**（取自评测日志，非 `task_success`
>    字段），而这两个任务的专家 `plan_success` 在上述 1926 个 episode 中均为 **0%**。
>    故**不能**用"策略成功率≈专家成功率"论证策略已触及环境上限。
> 2. **本自测成功率不可与 README 的 93.52% 直接比较**（训练配方、评测规模、
>    运行环境三者均不同，见证据一）。**特别地，不能把差距归因于规划器差异** ——
>    如第 1 点所述，策略执行不经 IK 规划，cuRobo→MPLib 的替换直接影响的是专家
>    检查而非策略本身。已查证的差异中，**训练配方（50000 步、clean+randomized）
>    是官方文档直接支持的一项**，且本方案受规则与算力限制无法对齐。
>    **本方案不声称已把差距逐项归因完毕**：推理精度（BF16 vs FP32）与评测规模
>    （每任务 30 次 vs 100 次）这两项的对照**未做**，其余环境差异亦未单独量化。
>    种子区间**不构成**差异：`st_seed = 100000 * (1 + seed)` 系官方默认取法
>    （`eval_policy_xpolicylab.py:302`，`seed=0` 时即 `100000`）。
>
>    **关于评测环境这一项，须特别声明**：《比赛要求》§4.2 本身即写明
>    「评测环境不限于 CUDA+cuRobo：AMD 赛事支持算力（ROCm+MPLib）产生的 json
>    可正常提交，须在 README 说明评测环境细节」，§4.3 亦把「评测环境细节
>    （含 ROCm+MPLib 等非 CUDA 环境）」列为必须如实说明的事项 —— 即主办方**明确
>    预期**非 CUDA 环境会产出**绝对数值水平不同**的结果。本方案 §4.0 还如实披露了
>    本镜像改变了相机渲染（降噪器 `oidn` → `none`），而官方 README **未记载**其
>    93.52% 是在何种硬件 / 规划器 / 渲染口径下测得。
>    **因此本方案对 §5 两行之差不作任何归因主张**：把它读作「能力差距」、
>    读作「本方案调优不足」，或反过来读作「本方案已与官方持平」，**均不被本报告支持**。
>    本方案能负责地陈述的只有一点：**本自测的绝对水平依赖本平台的评测环境，
>    该环境的贡献未被量化，故本自测成绩只能作方向性参考**。
> 3. **最终以赛事组在官方环境中的复测成绩为准**（《比赛要求》§4.2）。

**证据三（辅助）：两边的逐任务成绩"结构不同"，不是整体等比缩放。**
把本方案逐任务成功率与官方 README 的逐任务表按 50 个 clean 任务对齐，两者相关性
**很弱**：Pearson **0.32**、Spearman **0.30**（改用实际尝试口径为 0.35 / 0.34，
结论不变）。也就是说，本方案的成绩分布**不是**官方成绩分布的等比缩小 ——
双方各有明显偏强、偏弱的任务。两端例证：`grab_roller` 双方均为 100%；
而 `stack_blocks_three` 本方案 0% / 官方 100%；`hanging_mug` 本方案 6.7% /
官方 49%，后者恰是官方自身最弱的一项（官方 clean 逐任务最低值）。

> **本项只是一项观察，不含因果推断。** 造成两边结构不同的来源**不止一个**：
> 既可能是训练配方（11000 步 vs 50000 步、仅 `demo_clean` vs clean+randomized），
> 也可能是评测环境（本平台为 ROCm+MPLib，且 §4.0 已披露渲染口径与上游默认不同）。
> **本方案未做任何能把这两者分离开的实验**，故对本项**不作归因**：
> 它既不证明差距来自训练配方，也不证明来自评测环境。
> 上述相关系数由 `02_代码材料/src/official_corr.py` 生成，可复核：
>
> ```bash
> python src/official_corr.py <run-dir> <官方lingbot-vla-v2-6b-robotwin/README.md>
> ```
>
> 注：该脚本还会打印"逐任务均值的算术平均"（clean 为 47.67%），它与 §5 表中
> 加总口径的 48.55% 不同 —— 前者是 50 个任务各自成功率的**等权平均**，后者是
> **成功数加总 / 请求数加总**；因前 42 个任务为 30 次、后 8 个为 15 次，两者不相等。
> 本报告除本处外一律使用加总口径。

### 自测数字的测量口径（影响数值大小，非环境问题）

`results_tool.py` 抓取的 `Final success rate` 行（`eval_policy_xpolicylab.py:349`）
其**分母为请求的 episode 数**，而分子只统计成功数。当某次 `setup_demo` 抛出
`UnStableError`（初始场景不稳定）时，该 seed 会在 `eval_policy_xpolicylab.py:882-887`
被 `continue` 跳过 —— **策略从未被尝试，却仍计入分母**。同一次运行中滚动打印的
`Success rate`（`:1025` 的 `task_env.test_num`）用的是**实际尝试数**作分母，
故两行分母不同。本方案同时给出两种口径，以免误读。

实测（全量 100 个任务）：

| setting | 请求 | 实际执行 | 因场景不稳定跳过 | `Final` 口径 | 实际尝试口径 |
|---|---:|---:|---:|---:|---:|
| clean2clean | 1380 | 1313 | 67（**4.9%**） | 670/1380 = **48.55%** | 670/1313 = **51.03%** |
| clean2random | 750 | 640 | 110（**14.7%**） | 180/750 = **24.00%** | 180/640 = **28.12%** |

**randomized 的跳过率是 clean 的三倍**，这与日志中反复出现的场景生成失败一致
（`left arm planning failed (RRTConnect Failed)`、
`Only 0 cluttered objects are placed on the table`）——即随机化场景有相当比例
本身即不可解，**与策略能力无关**。单这一项就使 clean2random 的 `Final` 口径
比实际尝试口径低 **4.1 个百分点**。

个别任务受口径影响极大（下表均为 clean2clean）：

| 任务 | 请求 | 实际尝试 | `Final` 报出 | 实际尝试口径 |
|---|---:|---:|---:|---:|
| `adjust_bottle` | 30 | 19 | 19/30 = 63.3% | 19/19 = **100%** |
| `shake_bottle` | 30 | 17 | 17/30 = 56.7% | 17/17 = **100%** |
| `shake_bottle_horizontally` | 30 | 17 | 17/30 = 56.7% | 17/17 = **100%** |
| `dump_bin_bigbin` | 30 | 18 | 11/30 = 36.7% | 11/18 = 61.1% |
| `pick_diverse_bottles` | 30 | 15 | 6/30 = 20.0% | 6/15 = 40.0% |

本方案**不改动该口径** —— 官方评测代码不应被改写，此处仅作如实说明。
提交的 `results.json` 与官方复测均按 `Final` 口径，故两种口径的差异
**不影响提交数值与官方数值之间的可比性**；列出实际尝试口径只为说明
「绝对分数被场景不稳定额外压低」这一事实。

### 失败形态：全部为「步数耗尽」，无中途判负

全量 100 个任务共 **1103 个失败 episode**：**1103/1103（100%）终止于步数上限**
（400 / 500 / 600 / 700 / 800 / 900 / 1200 / 1500 / 1700，取自官方
`env_cfg/task_config/_eval_step_limit.yml`，本方案未修改该文件），
**无任一 episode 在步数未耗尽时被判负**；作为对照，850 个成功 episode 的
步数中位数为 **147** 步（最短 26 步，最长 1133 步）。即失败的**物理表现**是
「动作被执行、但任务未在限步内完成」，而非「动作被判为非法或中途判负」。

> **本项只描述失败的形态，不含归因。** 造成「跑满步数」的原因可能有多个
> （策略未完成动作序列、环境执行/规划延迟、任务本身的步数上限设置等），
> **本方案未做能把该形态归因于其中任一项的实验**，故不作归因。本项的价值仅在于
> 说明：自测的失败**不是**由评测器中途判负造成的，因而**成功率没有被评测口径额外压低**。

> 以上两节数字由 `02_代码材料/src/eval_diagnostics.py` 生成，可复核：
>
> ```bash
> python src/eval_diagnostics.py <run-dir>
> ```
>
> 以上为**全量 100 个任务**的最终数值（评测已于 2026-10-15 13:28 UTC 全部完成）。

## 4. 代码改动说明

### 4.0 改动归属的界定（请先读本节）

本平台镜像是**预构建**的，其自带的两个源码仓库**相对上游官方 commit 就已经有改动**
（AMD ROCm 移植、LoRA 支持、数据处理适配等）。这些改动**不是本方案所为**，属于镜像的
既有状态。为避免"哪些是官方代码、哪些是选手改动"产生歧义，本方案按下列方式界定并留证：

| 证据文件（在 `02_代码材料/src/`） | 内容 |
|---|---|
| `RoboTwin.upstream_diff.patch` | RoboTwin 相对上游 `266f3aa` 的**全部**改动 |
| `lingbot-vla-v2.upstream_diff.patch` | lingbot-vla-v2 相对上游 `951475a` 的**全部**改动 |
| `patch_checkpoint_rotation.py` | **本方案**对 checkpoint 轮转的补丁（幂等，原文件备份为 `.orig`） |

界定依据是**文件 mtime**，两批改动的时间戳截然分开：

| 批次 | mtime | 归属 |
|---|---|---|
| 镜像自带改动 | **2026-09-15 01:15 UTC**（与镜像内 `/RoboTwin/README.md`、`/opt/robotwin-env` 的构建时间落在同一分钟内） | 镜像 |
| 本方案改动 | **2026-10-12 04:31–04:35 UTC** | 本方案 |

按此标准，镜像自带改动 vs 本方案改动的量化如下：

| 仓库 | 镜像自带 | 本方案 |
|---|---|---|
| RoboTwin | 9 个文件，+279 / −74 | **0 个文件** |
| lingbot-vla-v2 | 7 个文件，+175 / −8，**其中 141 行为镜像自带** | **1 个文件，+34 行** |

镜像自带改动中体量最大的是 RoboTwin 的 `scripts/eval_policy_xpolicylab.py`（+191）、
`envs/robot/planner.py`（+72）、`envs/robot/robot.py`（+42）—— 这正是该镜像的
ROCm / MPLib 适配；lingbot-vla-v2 的 `lingbotvla/utils/lora_utils.py`（+20）与
`tasks/vla/train_lingbotvla.py` 中的 LoRA 相关部分（40 行）是镜像加入的 LoRA 支持。
**本方案未触碰以上任何一行。**

镜像内另自带一份复现说明 `RoboTwin_ROCm_Reproduction.ipynb`（mtime 同日 05:18），
其中逐条列出 `ROBOTWIN_DISABLE_CUROBO=1` / `ROBOTWIN_EE_PLANNER=mplib` /
`PYOPENGL_PLATFORM=egl` 等环境钩子 —— 正是上述 RoboTwin 改动所实现的那一套，
可**交叉印证**这批改动的归属。

> **其中一处须如实披露，不作掩饰**：`envs/_base_task.py` 那 5 行改动把光线追踪降噪器
> 由 `oidn` 改为 `none`，会改变相机渲染结果 —— 即**策略实际看到的图像与上游默认不同**，
> 不属于零副作用的纯适配。该改动由镜像预置（mtime `2026-09-15 01:15:31`，早于本方案
> 全部工作），本方案按镜像自带配方**原样使用、未做二次修改**。本节不含任何
> "该改动无影响"的断言。
>
> **该改动的影响边界须说清**：本方案**没有**任何实验能把这项渲染差异的影响单独
> 分离出来 —— 本报告内所有数字（§3 的全部自测值、§5 的两行对比）都是在**这个渲染
> 口径之下**取得的，提交权重的成绩与"上游默认渲染"下的成绩**不保证可比**。
> 若要测定其影响，需要额外跑一组"恢复 `oidn`"的对照，**本方案未做**，
> 因此不声称其影响为正或为负，也不把它列为分数差距的解释之一。

- **是否修改 LingBot-VLA 2.0 官方代码**：**是**，本方案改 **2 处**（§4.1、§4.2），
  均为 IO / 启动参数的基础设施改动，**不涉及模型结构、损失函数、数据管线与训练数值**。
- **是否修改 RoboTwin 官方评测代码**：**本方案未修改**（0 个文件）。但如 §4.0 所述，
  **镜像自带**的 RoboTwin 相对上游已有改动；本方案对此的态度是**原样使用、不做任何
  额外修改**，并在提交包内附上完整 diff 供评委核验。

### 4.1 本方案改动 1/2：checkpoint 轮转

**文件**：`tasks/vla/train_lingbotvla.py`
**改动**：新增一个模块级辅助函数 `_rotate_checkpoints()`，并在 3 处
`Checkpointer.save(...)` 调用点**之前**各插入一行调用。**未改动模型结构、超参、
损失函数、数据管线或任何训练数值逻辑。**

```text
+ import shutil
+ def _rotate_checkpoints(save_root: str) -> None: ...   # 约 30 行
  ...
  + _rotate_checkpoints(args.train.save_checkpoint_path)
    Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
```

**原因**：本平台工作区持久卷为 **100 GB**，而本配置下单份全量 checkpoint 实测
**72 GB**（`optimizer/` 48 GB + `model/` 24 GB + `extra_state/` 64 KB）。
官方 `Checkpointer.save` 只创建目录后写入，**从不清理旧 checkpoint**，也没有
keep_last 之类的参数，因此按 `save_steps=1000` 训练时**第二次保存必然写满磁盘**
（剩余 28 GB < 需要 72 GB）并中断训练。

轮转策略（在写入新 checkpoint 之前执行，此时最新的一份旧 checkpoint 记为
`entries[0]`）：

| 对象 | 处理 | 磁盘占用 |
|---|---|---|
| `entries[0]`（上一份） | 删除 `optimizer/`，保留 `model/` 作为兜底 | 24 GB |
| `entries[1:]`（更早） | 整份删除 | 0 |

写入新 checkpoint 时的占用峰值为 24 + 72 = **96 GB ≤ 100 GB**（已在 step 2000
实测验证：`global_step_2000` 完整 72 GB、`global_step_1000` 仅剩 24 GB 的 `model/`，
磁盘实际占用 96 GB，峰值不随步数增长）。
保留的 `model/` 可直接用于导出 HF 权重提交；仅 `optimizer/` 被丢弃，
即该兜底不用于带优化器状态续训。**未删除任何训练产物中的权重本体。**

> 补丁脚本：`02_代码材料/src/patch_checkpoint_rotation.py`（幂等，原文件自动备份为 `.orig`）。

### 4.2 本方案改动 2/2：暴露续训开关

**文件**：`experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh`
**改动**：1 行。**未改动任何训练数值逻辑。**

```diff
-  --train.enable_resume false \
+  --train.enable_resume "${ENABLE_RESUME:-false}" \
```

**原因**：官方 launcher 把 `--train.enable_resume` **硬编码**为 `false`，不暴露任何开关，
无法从外部续训。本平台单次会话受算力窗口限制，11000 步训练需跨会话续跑，故把该值改为
可被环境变量覆盖。

**默认取值不变**：未设 `ENABLE_RESUME` 时展开为 `false`，与官方原命令行**逐字等价**，
故对该变量不作设置的复现者**零行为差异**。`02_代码材料/train.sh` 透传该变量，并在
注释中标注"默认 false，即官方行为"。

### 4.3 环境适配（非代码改动，但复现必做）

官方复现指南中的 8 卡训练命令在本平台镜像上**直接执行会失败**，需补两个环境变量：

| # | 变量 | 不设的后果 |
|---|---|---|
| 1 | `AITER_TRITON_ONLY=1` | 镜像内 triton 为 3.5.1，而 aiter 的 gluon 内核要求 ≥3.6.0；`import flash_attn` 经 Triton 回退路径导入 aiter 时抛 `RuntimeError`。`qwenvl_in_vla.py` 中的 `is_flash_attn_available()` 守卫**失效**（transformers 仅检查包是否存在，不实际导入），故训练在建模阶段必崩。镜像自带的推理脚本 `launch_official_server.sh` 已设此变量，训练脚本未设。 |
| 2 | `TMPDIR=/var/tmp/lingbot` | 首次 forward 时 AMD COMGR 为**每个 rank** 写一个 543 MB 的 `hipfatbin`，8 卡合计约 4.3 GB，超过 `/tmp` 的 4 GB tmpfs 上限，报 `LLVM ERROR: IO failure on output stream: No space left on device`。这是**确定性故障**，非偶发。 |

这两项均**不改变训练数值**，只是让官方配置能在本镜像上跑起来。

## 5. 调优说明

本方案**未对官方配方做任何超参调优** —— 训练参数、数据处理方式、模型结构
三项均与 LingBot-VLA 2.0 官方 RoboTwin 复现配方逐项一致（超参见 §2）。
这是**有意为之**：本赛题对数据来源与推理策略的边界有明确约束，而官方配方
本身即在这些约束内经过完整 sweep 得到的配置；在其之外做未经充分验证的
改动，收益不确定而风险明确。

- **训练参数调整**：**无**。lr 1e-4（cosine 衰减至 5e-5）、`global_batch_size`
  256（micro 16 × 8 卡 × grad_accum 2）、optimizer AdamW、50 步 action chunk、
  gradient checkpointing、`enable_full_shard=true`、`TEACHER_MODE=full`
  （含 depth / future-depth / future-video 三路教师蒸馏）**全部沿用官方配置**。
  官方 launcher 内部硬编码 `use_lora=false` 与 `train_expert_only=false`，
  即**官方 RoboTwin 配方本身就是全参数训练**，本方案未做改动。
  由本方案设定的只有**运行规模**（卡数、`max_steps`、`save_steps`），
  官方脚本将这几项留给使用者指定，不属于超参调整。

- **数据处理方式**：**无额外处理**。训练数据**仅使用官方 `demo_clean` 的
  50 个任务，未使用任何 randomized 数据**。未重新生成数据、未做额外
  simulator rollout、未使用官方 randomization 重生成 observation、未引入
  任何外部数据。数据增强完全依赖官方数据管线自带的 visual/state
  augmentation 与加噪，未额外添加或修改。

- **模型结构改动**：**无**。除 §4.1（checkpoint 轮转）与 §4.2（续训开关）外，
  本方案未改动任何代码。这两处都属 IO / 启动参数的基础设施改动，**不参与前向或反向
  计算**，不影响模型结构、损失函数与训练数值；且 §4.2 的默认取值与原命令行逐字等价。

- **推理策略**：**单份权重**，不按 task ID / instruction 做任何确定性路由。

  按《比赛要求》§2.4，推理侧**被明确允许**调整的参数与策略包括：动作 chunk 执行长度
  及执行部分 chunk 后于 rollout 内重新推理（replanning）、动作序列平滑
  （action smoothing）、推理精度（bf16/fp32）、计算图编译开关及其它不引入额外权重、
  不改变监督来源的推理时参数。本方案把其中**推理精度**与**动作 chunk 执行长度**
  （`--use_length`）暴露为 `eval.sh` 的环境变量，**默认取值与官方 deploy 脚本逐字一致**
  （`bf16` / `25`）。本方案**未对这两项做任何寻优实测**（原因见 §3），故**不声称**由此
  带来任何提升。

  **明确不采用**的手段：多 checkpoint / 多 model ensemble、多 seed 投票、测试时适应
  （TTA）。temporal ensemble 与 confidence gating 依规则书自身表述「存在解释空间，
  以主办方书面答复为准；取得书面确认前，不视为已明确允许」，本方案在取得书面确认前
  **不采用**；「失败检测后重置 episode 重新执行」同属存疑项，**不采用**。

- **主要提升点**：前三项（训练参数、数据处理方式、模型结构）均为「无」，即本方案
  **未对官方训练配方做任何改动**；推理策略一项仅把官方 deploy 脚本本身已有的两个参数
  （推理精度、动作 chunk 执行长度）暴露为环境变量，**默认值与官方逐字一致**，且未做
  寻优实测。因此本提交与官方基线之间若存在差异，主要来源是**训练步数与随机性**，
  而非调优。**本方案不声称任何来自调优的增益。**
  本方案的实际工作量集中在**让官方配方能在本平台跑完** ——
  即 §4.1 的 checkpoint 轮转、§4.2 的续训开关与 §4.3 的两项环境适配：官方 8 卡训练
  命令在本平台镜像上直接执行必崩，且按 `save_steps=1000` 训练时第二次保存必然写满
  100 GB 持久卷而中断。这些改动不改变任何训练数值，但决定训练能否完成。

### 与官方基线的对比

| 方法 | clean2clean | clean2random |
|---|---:|---:|
| LingBot-VLA 2.0 官方 RoboTwin 后训练 checkpoint<br>（官方 README 数值；评测环境 README 未记载） | 93.52% | 92.80% |
| **本题提交 checkpoint**（基座 + 自训 11000 步、仅 clean 数据） | **48.55%**<br>670/1380 | **24.00%**<br>180/750 |

> 上表官方数据引自 LingBot-VLA 2.0 仓库 README，仅作对照参考。
> 两行**并非同一配方**：官方为 50000 步且 clean 与 randomized 混训，本方案为
> 基座 + 自训 11000 步、仅 `demo_clean`（`demo_randomized` 被比赛规则禁止用于训练，
> 见证据一）。因此两行相减**不构成"调优增益"或"能力差距"**。
>
> **本方案未以官方后版权重做任何评测对照**：《比赛要求》4.1/4.2 虽只将该权重限制在
> 「训练起点」「教师」两个角色上，但为避免任何歧义，本方案**不引入、不使用、也不
> 报告**该权重下的任何评测结果；差距的归因完全依赖 §3 的证据一 / 二 / 三
> （均由本方案自身的评测产物与官方 README 的公开数值得出）。
>
> 列名采用《比赛要求》附录 A.5 的官方口径：`clean2clean` = clean 训练 + clean 测试 +
> **seen** 指令（对应 `TASK_CONFIG=demo_clean`）；`clean2random` = clean 训练 +
> randomized 测试 + **unseen** 指令（对应 `TASK_CONFIG=demo_randomized`）。
> 本方案为**单份权重 cotrain**（附录 A.3：在全部 50 个任务上混合训练同一份权重），
> 两类 setting 共用同一 checkpoint，不做任何按任务切换。
