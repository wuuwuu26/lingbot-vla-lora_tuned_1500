#!/usr/bin/env bash
# 加载提交的 checkpoint，在 clean / randomized 两个 setting 下评测全部 50 个任务。
#
# 每个任务每 setting 评测 100 次（赛题要求），共 100 task-config × 100 = 10000 episodes。
# 底层调用 RoboTwin 的多卡动态队列评测脚本 run_clean_benchmark.py。
#
# 用法:
#   MODEL_PATH=/path/to/hf_ckpt GPU_COUNT=8 EPISODES=100 bash eval.sh
#
# 建议先用小子集验证链路，再跑全量:
#   EPISODES=1 GPU_COUNT=8 TASK_CONFIG=demo_clean bash eval.sh
#
# 推理策略（《比赛要求》§4.2 规定官方复测使用本文件的取值）:
#   INFER_PRECISION  bf16 | fp32   推理精度，§2.4(三) 明确允许
#   USE_LENGTH       动作 chunk 执行长度，§2.4(一) 明确允许
# 两项默认值与官方 deploy 脚本一致（bf16 / 25），默认路径不改变任何官方行为。
#
set -euo pipefail

# ---- 可覆盖配置 ----------------------------------------------------------
GPU_COUNT="${GPU_COUNT:-8}"
EPISODES="${EPISODES:-100}"
TASK_CONFIG="${TASK_CONFIG:-both}"          # both | demo_clean | demo_randomized
MODEL_PATH="${MODEL_PATH:-}"                 # 必填: 导出的 hf_ckpt 目录
RUN_NAME="${RUN_NAME:-}"
RUNTIME_DIR="${RUNTIME_DIR:-/workspace/runtime}"
BASE_PORT="${BASE_PORT:-13400}"
USE_COMPILE="${USE_COMPILE:-false}"

INFER_PRECISION="${INFER_PRECISION:-bf16}"   # bf16 | fp32
USE_LENGTH="${USE_LENGTH:-25}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/RoboTwin}"
PYTHON="${PYTHON:-/opt/robotwin-env/bin/python}"
SCRIPT="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py"
SERVER_SCRIPT="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh"
WRAPPER_DIR="${WRAPPER_DIR:-/var/tmp/lingbot_eval}"
# --------------------------------------------------------------------------

fail() { echo "错误: $*" >&2; exit 1; }

if [[ -z "${MODEL_PATH}" ]]; then
  fail "必须通过 MODEL_PATH 指定 checkpoint 目录（导出的 hf_ckpt）"
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  fail "checkpoint 目录不存在: ${MODEL_PATH}"
fi
if [[ ! -f "${SERVER_SCRIPT}" ]]; then
  fail "找不到官方服务端启动脚本: ${SERVER_SCRIPT}"
fi

case "${INFER_PRECISION}" in
  bf16|fp32) ;;
  *) fail "INFER_PRECISION 只能是 bf16 或 fp32（当前 ${INFER_PRECISION}）" ;;
esac
if ! [[ "${USE_LENGTH}" =~ ^[1-9][0-9]*$ ]]; then
  fail "USE_LENGTH 必须是正整数（当前 ${USE_LENGTH}）"
fi
if [[ "${GPU_COUNT}" != "4" && "${GPU_COUNT}" != "8" ]]; then
  fail "run_clean_benchmark.py 仅支持 --gpu-count 4 或 8（当前 ${GPU_COUNT}）"
fi

if [[ "${INFER_PRECISION}" == "fp32" ]]; then
  _bf16=False; _fp32=True
else
  _bf16=True;  _fp32=False
fi

# ---- 训练配置解析（关键，不可省）--------------------------------------------
# 本镜像的 launch_official_server.sh 总会 export LINGBOTVLA_TRAINING_CONFIG，
# 对「非官方默认路径」传入的 checkpoint，它会指向 repo 内的
# training/lingbotvla_cli.yaml —— 那份配置的 align_params 为空。
# 而本方案经 TEACHER_MODE=full 蒸馏的 checkpoint 含 depth/video align 分支参数
# （depth_align_embs / future_depth_align_embs / current_video_align_embs /
#   future_video_align_embs 等），deploy 侧是 strict=True 加载，
# 架构不符会直接抛错、评测无法启动。
# 故必须显式把该变量指向与训练一致的 resolved 配置。
if [[ -z "${LINGBOTVLA_TRAINING_CONFIG:-}" ]]; then
  for cand in "${MODEL_PATH}/lingbotvla_cli.yaml" \
              "$(dirname "$(realpath "${MODEL_PATH}")")/lingbotvla_cli.yaml"; do
    if [[ -f "${cand}" ]]; then
      LINGBOTVLA_TRAINING_CONFIG="${cand}"
      break
    fi
  done
fi
if [[ -z "${LINGBOTVLA_TRAINING_CONFIG:-}" || ! -f "${LINGBOTVLA_TRAINING_CONFIG}" ]]; then
  echo "错误: 未找到与训练一致的 lingbotvla_cli.yaml，评测会因架构不符而失败。" >&2
  echo "  请把它放入 checkpoint 目录内，或显式指定：" >&2
  echo "  LINGBOTVLA_TRAINING_CONFIG=<训练输出目录>/lingbotvla_cli.yaml bash eval.sh" >&2
  exit 1
fi
export LINGBOTVLA_TRAINING_CONFIG

# ---- 推理参数注入（自包含，见 README「推理参数的自包含施加」）------------------
# 官方 launch_official_server.sh 把推理参数写死在命令行里：
#   exec "$python_bin" -m deploy.lingbot_vla_v2_policy \
#        --use_length 25 --use_bf16 True --use_fp32 False ...
# 其中 python_bin=${LINGBOT_VLA_PYTHON:-/opt/robotwin-env/bin/python}（该脚本第 7 行）。
# LINGBOT_VLA_PYTHON 是官方留出的解释器覆盖点，全镜像仅此一处引用、默认未设。
#
# 因此本脚本不改动任何官方文件，而是把 LINGBOT_VLA_PYTHON 指向一个运行时生成的
# 薄包装器，由它重写上述三个参数后再 exec 真正的解释器。这样在评委的全新镜像实例上
# 同样成立（无持久化副作用），且默认取值下重写结果与原命令行逐字等价。
mkdir -p "${WRAPPER_DIR}"
WRAPPER="${WRAPPER_DIR}/server_python_wrapper.sh"

cat > "${WRAPPER}" <<'WRAPPER_EOF'
#!/usr/bin/env bash
# 由 eval.sh 运行时生成。重写服务端推理参数后 exec 真正的解释器。
# 用法（由 launch_official_server.sh 调用）：
#   <本脚本> -m deploy.lingbot_vla_v2_policy --use_length 25 --use_bf16 True --use_fp32 False ...
set -euo pipefail
REAL_PYTHON='__REAL_PYTHON__'
WANT_USELEN='__USE_LENGTH__'
WANT_BF16='__BF16__'
WANT_FP32='__FP32__'

if [[ "${1:-}" == "--print-rewrite" ]]; then
  # 预检模式：打印重写后的参数，不执行。供 eval.sh 在正式评测前校验。
  shift
  PRINT_ONLY=1
else
  PRINT_ONLY=0
fi

args=(); seen_len=0; seen_bf16=0; seen_fp32=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --use_length) args+=("$1" "${WANT_USELEN}"); seen_len=1;  shift 2 ;;
    --use_bf16)   args+=("$1" "${WANT_BF16}");   seen_bf16=1; shift 2 ;;
    --use_fp32)   args+=("$1" "${WANT_FP32}");   seen_fp32=1; shift 2 ;;
    *)            args+=("$1"); shift ;;
  esac
done
[[ "${seen_len}"  -eq 0 ]] && args+=("--use_length" "${WANT_USELEN}")
[[ "${seen_bf16}" -eq 0 ]] && args+=("--use_bf16"   "${WANT_BF16}")
[[ "${seen_fp32}" -eq 0 ]] && args+=("--use_fp32"   "${WANT_FP32}")

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  # 单行空格分隔输出，供 eval.sh 用 grep -F 校验
  [[ ${#args[@]} -gt 0 ]] && printf '%s\n' "${args[*]}"
  exit 0
fi
# 启动时打点：该行随 stdout 落入 server_gpuN.log，便于事后核验实际生效值。
printf '[eval-wrapper] real_python=%s use_length=%s use_bf16=%s use_fp32=%s\n' \
  "${REAL_PYTHON}" "${WANT_USELEN}" "${WANT_BF16}" "${WANT_FP32}"
exec "${REAL_PYTHON}" "${args[@]}"
WRAPPER_EOF

# 占位符替换（用 | 作分隔符，避免路径中的 / 干扰）
sed -i \
  -e "s|__REAL_PYTHON__|${PYTHON}|g" \
  -e "s|__USE_LENGTH__|${USE_LENGTH}|g" \
  -e "s|__BF16__|${_bf16}|g" \
  -e "s|__FP32__|${_fp32}|g" \
  "${WRAPPER}"
chmod +x "${WRAPPER}"
bash -n "${WRAPPER}" || fail "生成的包装器语法有误: ${WRAPPER}"

# 预检：用官方 launch_official_server.sh 的真实命令行喂进去，确认重写结果正确。
probe="$("${WRAPPER}" --print-rewrite -m deploy.lingbot_vla_v2_policy \
  --model_path /probe --use_length 25 --use_bf16 True --use_fp32 False \
  --use_compile False --port 1 2>&1)" || fail "包装器预检执行失败"
for want in "--use_length ${USE_LENGTH}" "--use_bf16 ${_bf16}" "--use_fp32 ${_fp32}"; do
  # shellcheck disable=SC2076
  if ! grep -qF -- "${want}" <<<"${probe}"; then
    echo "${probe}" >&2
    fail "包装器预检未产出期望参数: ${want}"
  fi
done
export LINGBOT_VLA_PYTHON="${WRAPPER}"

# 端口占用预检
for ((i = 0; i < GPU_COUNT; i++)); do
  port=$((BASE_PORT + i))
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -q ":${port} "; then
    fail "端口 ${port} 已被占用，先停止旧的模型 server"
  fi
done

if [[ -z "${RUN_NAME}" ]]; then
  size="100"; [[ "${TASK_CONFIG}" == "both" ]] || size="50"
  # 把 checkpoint 名并进 RUN_NAME：run_clean_benchmark.py 的 --resume 依据
  # run_dir/done/*.done 跳过已完成任务，而 run_dir 只由 RUN_NAME 决定。
  # 若两个不同 checkpoint 共用同一 RUN_NAME，第二次会静默复用第一次的结果。
  ckpt_tag="$(basename "$(realpath "${MODEL_PATH}")")"
  RUN_NAME="${TASK_CONFIG}_${size}x${EPISODES}_${GPU_COUNT}gpu_${ckpt_tag}"
fi

# 评测环境（ROCm 路径使用 MPLib 运动规划器替代 cuRobo）
export ROBOTWIN_DISABLE_CUROBO=1
export ROBOTWIN_EE_PLANNER=mplib
export PYOPENGL_PLATFORM=egl

echo "MODEL_PATH=${MODEL_PATH}"
echo "LINGBOTVLA_TRAINING_CONFIG=${LINGBOTVLA_TRAINING_CONFIG}"
echo "LINGBOT_VLA_PYTHON=${LINGBOT_VLA_PYTHON}"
echo "推理参数: use_length=${USE_LENGTH} precision=${INFER_PRECISION} (use_bf16=${_bf16} use_fp32=${_fp32})"
echo "GPU_COUNT=${GPU_COUNT} EPISODES=${EPISODES} TASK_CONFIG=${TASK_CONFIG}"
echo "RUN_NAME=${RUN_NAME}"
echo "预计规模: ${TASK_CONFIG} × ${EPISODES} episodes/任务"

cd "${ROBOTWIN_ROOT}"
"${PYTHON}" "${SCRIPT}" \
  --gpu-count "${GPU_COUNT}" \
  --episodes "${EPISODES}" \
  --task-config "${TASK_CONFIG}" \
  --model-path "${MODEL_PATH}" \
  --run-name "${RUN_NAME}" \
  --runtime-dir "${RUNTIME_DIR}" \
  --base-port "${BASE_PORT}" \
  --expert-check \
  --accept-expert-info-on-failure \
  --no-video \
  $( [[ "${USE_COMPILE}" == "true" ]] && echo --use-compile ) \
  --resume

RUN_DIR="${RUNTIME_DIR}/outputs/${RUN_NAME}"
echo
echo "评测完成。结果目录: ${RUN_DIR}"
echo "每个任务的实际耗时（用于核算评测窗口）: ${RUN_DIR}/task_times.tsv"
echo "汇总为提交用 results.json:"
echo "  python tools/results_tool.py collect --run-dir '${RUN_DIR}' --team-id <你的团队ID>"
