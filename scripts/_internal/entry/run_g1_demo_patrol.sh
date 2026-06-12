#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

LOOPS="${LOOPS:-0}"
PAUSE="${PAUSE:-0.2}"
POINTS="${POINTS:-1.0,0.0,0.0;8.0,2.0,0.0;8.0,6.0,0.0;1.0,4.0,0.0}"

env -u PYTHONHOME -u PYTHONPATH -u PYTHONEXECUTABLE bash --noprofile --norc -lc \
  'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 scripts/_internal/tasks/patrol_g1_nav2.py --points "$1" --loops "$2" --pause "$3" --wait_timeout 60 --continue_on_failure' \
  bash "${POINTS}" "${LOOPS}" "${PAUSE}"
