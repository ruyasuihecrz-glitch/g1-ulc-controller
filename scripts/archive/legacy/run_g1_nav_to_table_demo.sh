#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

PRINT_ONLY="false"
if [[ "${1:-}" == "--print-only" || "${1:-}" == "--dry-run" ]]; then
  PRINT_ONLY="true"
fi

MAP_OUTPUT="$(python3 scripts/create_substation_nav2_map.py)"
SCENE_JSON="$(printf '%s\n' "${MAP_OUTPUT}" | sed -n '2p')"

POINTS="$(
  /usr/bin/python3 - "${SCENE_JSON}" <<'PY'
import json
import sys

scene = json.load(open(sys.argv[1], encoding="utf-8"))
table = scene["manipulation"]["table"]
approach = table["approach_goal"]
front = table["front_goal"]
print(";".join(",".join(f"{value:.4f}" for value in pose) for pose in [approach, front]))
PY
)"

echo "scene_json: ${SCENE_JSON}"
echo "table_points: ${POINTS}"

if [[ "${PRINT_ONLY}" == "true" ]]; then
  exit 0
fi

env -u PYTHONHOME -u PYTHONPATH -u PYTHONEXECUTABLE bash --noprofile --norc -lc \
  'source /opt/ros/jazzy/setup.bash && /usr/bin/python3 scripts/patrol_g1_nav2.py --points "$1" --loops 1 --pause 0.2 --wait_timeout 90' \
  bash "${POINTS}"
