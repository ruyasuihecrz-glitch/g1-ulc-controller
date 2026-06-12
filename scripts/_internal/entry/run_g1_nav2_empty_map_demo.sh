#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
PARAMS_FILE="${PARAMS_FILE:-${REPO_ROOT}/configs/nav2/g1_nav2_empty_params.yaml}"
MAP_YAML="${MAP_YAML:-}"
RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/.runtime}"
ROS_HOME_DIR="${ROS_HOME_DIR:-${RUNTIME_DIR}/ros}"
ROS_LOG_DIR="${ROS_LOG_DIR:-${REPO_ROOT}/logs/ros2}"
USE_SIM_TIME="${USE_SIM_TIME:-true}"
USE_COMPOSITION="${USE_COMPOSITION:-false}"
LOG_LEVEL="${LOG_LEVEL:-info}"
WAIT_FOR_BASE_TF="${WAIT_FOR_BASE_TF:-true}"
WAIT_FOR_BASE_TF_TIMEOUT="${WAIT_FOR_BASE_TF_TIMEOUT:-120}"

mkdir -p "${ROS_HOME_DIR}" "${ROS_LOG_DIR}"
export ROS_HOME="${ROS_HOME:-${ROS_HOME_DIR}}"
export ROS_LOG_DIR

if [[ -f "${ROS_SETUP}" ]]; then
  # ROS setup scripts may read unset variables.
  # shellcheck disable=SC1090
  set +u
  source "${ROS_SETUP}"
  set -u
else
  echo "[ERROR] ROS setup file not found: ${ROS_SETUP}" >&2
  exit 1
fi

if [[ -z "${MAP_YAML}" ]]; then
  MAP_YAML="$(python3 scripts/_internal/maps/create_empty_nav2_map.py)"
fi

ros_bool() {
  case "${1,,}" in
    true|1|yes|on) echo "True" ;;
    false|0|no|off) echo "False" ;;
    *)
      echo "[ERROR] Invalid boolean value: $1" >&2
      exit 1
      ;;
  esac
}

USE_SIM_TIME_ARG="$(ros_bool "${USE_SIM_TIME}")"
USE_COMPOSITION_ARG="$(ros_bool "${USE_COMPOSITION}")"

cleanup() {
  for pid in ${PIDS:-}; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

echo "map: ${MAP_YAML}"
echo "params: ${PARAMS_FILE}"
echo "ros log dir: ${ROS_LOG_DIR}"
echo "use_sim_time: ${USE_SIM_TIME_ARG}"
echo "use_composition: ${USE_COMPOSITION_ARG}"
echo "wait_for_base_tf: ${WAIT_FOR_BASE_TF} timeout=${WAIT_FOR_BASE_TF_TIMEOUT}s"

ros2 run tf2_ros static_transform_publisher \
  --x 0.0 --y 0.0 --z 0.0 \
  --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id map --child-frame-id odom &
PIDS="${PIDS:-} $!"

if [[ "${WAIT_FOR_BASE_TF,,}" == "true" || "${WAIT_FOR_BASE_TF}" == "1" ]]; then
  echo "[INFO] Waiting for G1 bridge TF odom -> base_link before launching Nav2..."
  deadline=$((SECONDS + WAIT_FOR_BASE_TF_TIMEOUT))
  while true; do
    if timeout 2s ros2 topic echo /tf --once 2>/dev/null | grep -q "child_frame_id: base_link"; then
      echo "[INFO] Found TF odom -> base_link."
      break
    fi
    if (( SECONDS >= deadline )); then
      echo "[ERROR] Timed out waiting for TF odom -> base_link." >&2
      echo "[ERROR] Start the G1 bridge first in another terminal:" >&2
      echo "        bash scripts/demo.sh policy-empty" >&2
      exit 1
    fi
    sleep 1.0
  done
fi

ros2 launch "${REPO_ROOT}/launch/g1_nav2_empty_launch.py" \
  use_sim_time:="${USE_SIM_TIME_ARG}" \
  map:="${MAP_YAML}" \
  params_file:="${PARAMS_FILE}" \
  autostart:=True \
  log_level:="${LOG_LEVEL}" &
PIDS="${PIDS:-} $!"

echo
echo "[INFO] Nav2 empty-map demo is starting."
echo "[INFO] Keep the G1 bridge running in another terminal:"
echo "       bash scripts/demo.sh policy-empty"
echo "[INFO] Send a goal with:"
echo "       python3 scripts/_internal/tasks/send_g1_nav2_goal.py --x 2.0 --y 0.0 --yaw 0.0"
echo

wait
