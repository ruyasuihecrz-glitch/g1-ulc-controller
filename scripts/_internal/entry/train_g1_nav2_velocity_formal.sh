#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-Unitree-G1-29dof-StableVelocity}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50000}"
SEED="${SEED:-42}"
HEADLESS="${HEADLESS:-true}"
RUN_NAME="${RUN_NAME:-g1_nav2_officialbase_antistall_env8192_ckpt2000}"
RESUME="${RESUME:-false}"
LOAD_RUN="${LOAD_RUN:-}"
CHECKPOINT="${CHECKPOINT:-}"
SESSION_NAME="${SESSION_NAME:-g1_nav2_velocity_train}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/g1_nav2_velocity_formal_${TIMESTAMP}.log"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/.runtime}"
PYTHON_SHIM_DIR="${PYTHON_SHIM_DIR:-${RUNTIME_DIR}/bin}"

mkdir -p "${LOG_DIR}" "${PYTHON_SHIM_DIR}"
export PYTHONPATH="${REPO_ROOT}/source/unitree_rl_lab${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" "${PYTHON_SHIM_DIR}/python"
  export PATH="${PYTHON_SHIM_DIR}:${PATH}"
fi

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

CMD="/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task ${TASK_NAME} \
  --num_envs ${NUM_ENVS} \
  --max_iterations ${MAX_ITERATIONS} \
  --seed ${SEED} \
  --run_name ${RUN_NAME}"

if is_true "${HEADLESS}"; then
  CMD="${CMD} --headless"
fi

if is_true "${RESUME}"; then
  if [[ -z "${LOAD_RUN}" || -z "${CHECKPOINT}" ]]; then
    echo "RESUME=true requires explicit LOAD_RUN and CHECKPOINT. Refusing to resume implicitly."
    exit 1
  fi
  CMD="${CMD} --resume --load_run ${LOAD_RUN} --checkpoint ${CHECKPOINT}"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session ${SESSION_NAME} already exists. Attach with:"
  echo "  tmux attach -t ${SESSION_NAME}"
  exit 1
fi

echo "task: ${TASK_NAME}"
echo "run_name: ${RUN_NAME}"
echo "resume: ${RESUME}"
echo "load_run: ${LOAD_RUN:-<none>}"
echo "checkpoint: ${CHECKPOINT:-<none>}"
echo "num_envs: ${NUM_ENVS}"
echo "max_iterations: ${MAX_ITERATIONS}"
echo "seed: ${SEED}"
echo "pythonpath: ${PYTHONPATH}"
echo "save_interval: 2000 (from G1StableVelocityPPORunnerCfg)"
echo "oom_fallback: rerun with NUM_ENVS=6144 if 8192 exceeds available GPU memory"
echo "log_file: ${LOG_FILE}"
echo "command:"
echo "  ${CMD}"

tmux new-session -d -s "${SESSION_NAME}" "cd ${REPO_ROOT} && export PYTHONPATH='${PYTHONPATH}' PATH='${PATH}' && { echo '[INFO] GPU memory before launch:'; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv || true; ${CMD}; } 2>&1 | tee ${LOG_FILE}"

echo "Training launched in tmux session: ${SESSION_NAME}"
echo "Attach:"
echo "  tmux attach -t ${SESSION_NAME}"
echo "Tail log:"
echo "  tail -f ${LOG_FILE}"
echo "TensorBoard:"
echo "  tensorboard --logdir ${REPO_ROOT}/logs/rsl_rl"
