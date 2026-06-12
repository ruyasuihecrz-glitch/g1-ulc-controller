#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-Unitree-G1-29dof-UpperBodyStabilityFineTune}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
SEED="${SEED:-42}"
HEADLESS="${HEADLESS:-true}"
RUN_NAME="${RUN_NAME:-g1_29dof_arm_disturbance_base_from_15999}"
RESUME="${RESUME:-true}"
LOAD_RUN="${LOAD_RUN:-2026-05-29_09-14-43_g1_29dof_turn_finetune_from_14000}"
CHECKPOINT="${CHECKPOINT:-model_15999.pt}"
SESSION_NAME="${SESSION_NAME:-g1_upper_body_stability_ft}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="${REPO_ROOT}/logs"
LOG_FILE="${LOG_DIR}/g1_29dof_upper_body_stability_ft_${TIMESTAMP}.log"
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
echo "upper_body_ft: arm-only disturbance + low-speed tracking + stand stability rewards"
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
echo "  tensorboard --logdir ${REPO_ROOT}/logs/rsl_rl/unitree_g1_29dof_velocity"
