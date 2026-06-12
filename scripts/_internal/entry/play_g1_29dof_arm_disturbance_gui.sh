#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

TASK_NAME="${TASK_NAME:-Unitree-G1-29dof-UpperBodyStabilityFineTune}"
NUM_ENVS="${NUM_ENVS:-4}"
CHECKPOINT="${CHECKPOINT:-logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-08_09-00-27_g1_29dof_arm_disturbance_base_from_15999/model_16000.pt}"

export TERM="${TERM:-xterm}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi
export PYTHONPATH="${REPO_ROOT}/source/unitree_rl_lab${PYTHONPATH:+:${PYTHONPATH}}"

exec /workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task "${TASK_NAME}" \
  --num_envs "${NUM_ENVS}" \
  --checkpoint "${CHECKPOINT}"
