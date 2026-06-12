#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

CHECKPOINT="${1:-}"

CMD=(/workspace/isaaclab/isaaclab.sh -p scripts/_internal/export/export_g1_29dof_stable_velocity.py
  --task Unitree-G1-29dof-StableVelocity
  --num_envs 1
  --headless)

if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=(--checkpoint "${CHECKPOINT}")
fi

"${CMD[@]}"

echo "Exported stable policy to:"
echo "  deploy/robots/g1_29dof/config/policy/stable_velocity/v0/exported/policy.onnx"
echo "  deploy/robots/g1_29dof/config/policy/stable_velocity/v0/params/deploy.yaml"
