#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

MAP_OUTPUT="$(python3 scripts/_internal/maps/create_substation_nav2_map.py)"
MAP_YAML="$(printf '%s\n' "${MAP_OUTPUT}" | sed -n '1p')"
SCENE_JSON="$(printf '%s\n' "${MAP_OUTPUT}" | sed -n '2p')"

PARAMS_FILE="${PARAMS_FILE:-${REPO_ROOT}/configs/nav2/g1_nav2_substation_params.yaml}" \
MAP_YAML="${MAP_YAML}" \
bash scripts/_internal/entry/run_g1_nav2_empty_map_demo.sh
