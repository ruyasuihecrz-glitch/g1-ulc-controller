#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
CKPT_ROOT="${REPO_ROOT}/logs/rsl_rl/unitree_g1_29dof_stablevelocity"

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "Checkpoint root does not exist: ${CKPT_ROOT}"
  exit 1
fi

echo "StableVelocity checkpoints under:"
echo "  ${CKPT_ROOT}"
echo

find "${CKPT_ROOT}" -type f -name 'model_*.pt' -printf '%TY-%Tm-%Td %TH:%TM:%TS|%s|%p\n' \
  | sort -r \
  | while IFS='|' read -r mtime size_bytes path; do
      size_mb="$(awk "BEGIN {printf \"%.2f\", ${size_bytes}/1024/1024}")"
      echo "${mtime}  ${size_mb} MB  ${path}"
    done

echo
echo "Recommendation:"
echo "  Resume from a checkpoint before the critic explosion, for example model_8000.pt or model_8500.pt."
echo "  Do not resume from checkpoints written after value_function loss jumped to 1e17/inf."
