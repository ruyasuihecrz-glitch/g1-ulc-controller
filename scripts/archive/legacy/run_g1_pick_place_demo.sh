#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

PRINT_ONLY="false"
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --print-only|--dry-run)
      PRINT_ONLY="true"
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      ;;
  esac
done

MAP_OUTPUT="$(python3 scripts/create_substation_nav2_map.py)"
SCENE_JSON="$(printf '%s\n' "${MAP_OUTPUT}" | sed -n '2p')"

echo "scene_json: ${SCENE_JSON}"

BASE_CMD=(/usr/bin/python3 scripts/g1_nav_pick_place_state_machine.py
  --scene_json "${SCENE_JSON}"
  --wait_timeout "${WAIT_TIMEOUT:-90}"
  --nav_retries "${NAV_RETRIES:-1}"
  --loops "${LOOPS:-1}"
  --return_x "${RETURN_X:-0.8}"
  --return_y "${RETURN_Y:-0.0}"
  --return_yaw "${RETURN_YAW:-0.0}"
  --manip_udp_host "${MANIP_UDP_HOST:-127.0.0.1}"
  --manip_udp_port "${MANIP_UDP_PORT:-15002}"
  --manip_timeout "${MANIP_TIMEOUT:-20}"
  --manip_duration_scale "${MANIP_DURATION_SCALE:-1.0}"
)

if [[ "${PRINT_ONLY}" == "true" ]]; then
  BASE_CMD+=(--print_only)
fi

BASE_CMD+=("${EXTRA_ARGS[@]}")

env -u PYTHONHOME -u PYTHONPATH -u PYTHONEXECUTABLE bash --noprofile --norc -lc \
  'source /opt/ros/jazzy/setup.bash && "$@"' \
  bash "${BASE_CMD[@]}"
