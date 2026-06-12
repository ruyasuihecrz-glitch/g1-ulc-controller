#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

MAP_OUTPUT="$(python3 scripts/_internal/maps/create_substation_nav2_map.py)"
SCENE_JSON="$(printf '%s\n' "${MAP_OUTPUT}" | sed -n '2p')"

SCENE_JSON="${SCENE_JSON}" bash scripts/_internal/entry/run_g1_policy_cmdvel_ros2.sh
