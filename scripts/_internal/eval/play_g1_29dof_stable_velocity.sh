#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

MODE="${1:-play}"
CHECKPOINT="${2:-}"
TASK_NAME="Unitree-G1-29dof-StableVelocity"
HEADLESS="${HEADLESS:-0}"
VX="${VX:-}"
VY="${VY:-}"
WZ="${WZ:-}"

if [[ "${MODE}" == "eval" ]]; then
  CMD=(/workspace/isaaclab/isaaclab.sh -p scripts/_internal/eval/eval_g1_29dof_stable_velocity_commands.py
    --task "${TASK_NAME}"
    --num_envs 1
    --steps 1000)
else
  CMD=(/workspace/isaaclab/isaaclab.sh -p scripts/rsl_rl/play.py
    --task "${TASK_NAME}"
    --num_envs 1)
fi

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -n "${VX}" ]]; then
  CMD+=(--vx "${VX}")
fi
if [[ -n "${VY}" ]]; then
  CMD+=(--vy "${VY}")
fi
if [[ -n "${WZ}" ]]; then
  CMD+=(--wz "${WZ}")
fi

if [[ "${HEADLESS}" != "0" ]]; then
  CMD+=(--headless)
fi

echo "Running mode=${MODE}"
if [[ -n "${VX}" || -n "${VY}" || -n "${WZ}" ]]; then
  echo "Fixed command: vx=${VX:-0.0} vy=${VY:-0.0} wz=${WZ:-0.0}"
fi
"${CMD[@]}"
