#!/usr/bin/env bash
# 全参数 SFT 一键训练脚本 —— 官方 launcher 的薄包装。
#
# 设计原则
# --------
# 本脚本**不自己拼训练参数**，只做两件官方脚本没做的事，然后原样调用官方 launcher：
#
#   1. 补齐本平台镜像上缺失的两个环境变量（不加会导致训练**必崩**，见下）
#   2. 数据与基座权重的合规自检（赛题红线，见 04_复现与调优说明 §4.2）
#
# 训练参数全部由官方 launcher 决定，因此与官方复现配方**逐参数等价**。
# 官方 launcher 负责：梯度累积推导、teacher 依赖校验、global batch 一致性校验、
# 陈旧数据路径归一化、FSDP2 分片配置等。
#
# 没有自己拼 `--train.*` 的原因：官方 launcher 里有 grad accum 推导与多处
# 一致性断言，手抄一份必然产生细微出入且难以察觉。
#
# 两个环境变量为什么必须加
# ------------------------
#   AITER_TRITON_ONLY=1
#     镜像内 triton 为 3.5.1，而 aiter 的 gluon 内核要求 >= 3.6.0。flash_attn
#     缺 flash_attn_2_cuda HIP 扩展会回退到 Triton 实现，导入 aiter 时抛
#     RuntimeError。qwenvl_in_vla.py 的 is_flash_attn_available() 守卫失效
#     （transformers 只检查包是否存在、不实际导入），故训练在建模阶段必崩。
#     镜像自带的推理脚本 launch_official_server.sh 已设此变量，训练脚本漏了。
#
#   TMPDIR
#     首次 forward 时 AMD COMGR 为每个 rank 写一个 543 MB 的 hipfatbin，
#     8 卡合计约 4.3 GB，超过 /tmp 的 4 GB tmpfs 上限，报
#     "LLVM ERROR: IO failure on output stream: No space left on device"。
#     这是确定性故障，非偶发。必须指到 overlay（/，2.7 TB）。
#
# 这两项都**不改变训练数值**。
#
# 用法
# ----
#   GPU_COUNT=8 MAX_STEPS=11000 SAVE_STEPS=1000 \
#   OUTPUT_DIR=/workspace/runtime/outputs/full_sft_11k_8gpu \
#   bash train.sh
#
# 其余可透传的官方变量：MICRO_BATCH_SIZE、GLOBAL_BATCH_SIZE、OPTIMIZER、
# TEACHER_MODE、NUM_WORKERS、PREFETCH_FACTOR、ENABLE_FULL_SHARD、
# TIMEOUT_SECONDS、DATA_LIST、CONFIG、LOG_FILE、TMPDIR_ROOT。
#
# 续训：在已有 checkpoint 的 OUTPUT_DIR 上以 ENABLE_RESUME=true 重跑即可。
# （框架默认 resume_dataloader_state=true，会一并还原数据顺序、LR 调度与 RNG 状态。）

set -euo pipefail

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/RoboTwin}"
LAUNCHER="${LAUNCHER:-${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh}"

# ---- 训练规模（官方 launcher 的默认值是 4 卡 / 100 步，此处按正式训练设定）----
GPU_COUNT="${GPU_COUNT:-8}"
MAX_STEPS="${MAX_STEPS:-11000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
TEACHER_MODE="${TEACHER_MODE:-full}"
OPTIMIZER="${OPTIMIZER:-adamw}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-0}"

DATA_LIST="${DATA_LIST:-${ROBOTWIN_ROOT}/data/robotwin_demo_clean_joint_v30.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runtime/outputs/full_sft_${GPU_COUNT}gpu_${MAX_STEPS}steps}"

# 注意：官方 launcher 内 `MODEL_ROOT` / `BASE_MODEL` 是**硬编码赋值**（形如
# BASE_MODEL="${MODEL_ROOT}/..."，而非 ${VAR:-default}），从外部 export 无法覆盖。
# 这里按同样规则推导出 launcher 实际会用的路径，**仅用于合规自检，不向下传递**。
LAUNCHER_BASE_MODEL="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b"
TMPDIR_ROOT="${TMPDIR_ROOT:-/var/tmp/lingbot}"
# ---------------------------------------------------------------------------

die() { echo "错误: $*" >&2; exit 1; }

# ===== 合规自检（赛题红线）=================================================

[[ -f "${LAUNCHER}" ]] || die "找不到官方 launcher: ${LAUNCHER}"
[[ -f "${DATA_LIST}" ]] || die "找不到数据清单: ${DATA_LIST}"

# 1) 数据清单必须恰好 50 个 clean 任务
manifest_lines="$(grep -cve '^[[:space:]]*$' "${DATA_LIST}")"
[[ "${manifest_lines}" -eq 50 ]] \
  || die "${DATA_LIST} 含 ${manifest_lines} 条记录，应为 50 条 clean 任务"

# 2) 禁止 randomized 数据训练
if grep -qi 'random' "${DATA_LIST}"; then
  die "数据清单出现 randomized 字样，赛题禁止使用 randomized 数据训练"
fi

# 3) 基座必须是官方 lingbot-vla-v2-6b，而非 -robotwin 后训练权重
[[ -d "${LAUNCHER_BASE_MODEL}" ]] || die "找不到基座权重目录: ${LAUNCHER_BASE_MODEL}"
# 只看目录名：完整路径里必然含 robotwin（上层目录名为 lingbot_vla_v2_6b_robotwin），
# 匹配整个路径会误报。
if [[ "$(basename "${LAUNCHER_BASE_MODEL}")" == *robotwin* ]]; then
  die "基座疑似 -robotwin 后训练权重（${LAUNCHER_BASE_MODEL}）；赛题要求 robbyant/lingbot-vla-v2-6b"
fi

# ===== 环境修补（不做则训练必崩）===========================================

# 坑 1：aiter gluon 内核要求 triton>=3.6.0，镜像只有 3.5.1
export AITER_TRITON_ONLY=1

# 坑 2：8 卡 COMGR 编译产物约 4.3 GB，撑爆 4 GB 的 /tmp tmpfs
mkdir -p "${TMPDIR_ROOT}"
export TMPDIR="${TMPDIR_ROOT}"

echo "== 训练配置 =="
echo "  GPU_COUNT=${GPU_COUNT}  MAX_STEPS=${MAX_STEPS}  SAVE_STEPS=${SAVE_STEPS}"
echo "  TEACHER_MODE=${TEACHER_MODE}  OPTIMIZER=${OPTIMIZER}"
echo "  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "  基座=${LAUNCHER_BASE_MODEL}（由官方 launcher 硬编码指定，已通过合规自检）"
echo "  数据=${DATA_LIST}（${manifest_lines} 个 clean 任务，无 randomized）"
echo "  输出=${OUTPUT_DIR}"
echo "  环境修补: AITER_TRITON_ONLY=1  TMPDIR=${TMPDIR_ROOT}"
echo

# ===== 调用官方 launcher ===================================================
# 训练参数完全交给官方脚本，本脚本不追加任何 --train.* 参数。
export GPU_COUNT MAX_STEPS SAVE_STEPS TEACHER_MODE OPTIMIZER
export MICRO_BATCH_SIZE GLOBAL_BATCH_SIZE TIMEOUT_SECONDS
export DATA_LIST OUTPUT_DIR
# ENABLE_RESUME 由调用方按需设置（默认 false，即官方行为）

exec bash "${LAUNCHER}"
