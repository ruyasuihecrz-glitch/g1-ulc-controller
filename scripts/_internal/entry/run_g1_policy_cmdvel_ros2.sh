#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

if [[ "${TERM:-}" == "dumb" || -z "${TERM:-}" ]]; then
  export TERM=xterm-256color
fi

RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/.runtime}"
ROS_HOME_DIR="${ROS_HOME_DIR:-${RUNTIME_DIR}/ros}"
ROS_LOG_DIR_LOCAL="${ROS_LOG_DIR_LOCAL:-${REPO_ROOT}/logs/ros2}"
PYTHON_SHIM_DIR="${PYTHON_SHIM_DIR:-${RUNTIME_DIR}/bin}"

mkdir -p "${ROS_HOME_DIR}" "${ROS_LOG_DIR_LOCAL}" "${PYTHON_SHIM_DIR}"
export ROS_HOME="${ROS_HOME:-${ROS_HOME_DIR}}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${ROS_LOG_DIR_LOCAL}}"
export PYTHONPATH="${REPO_ROOT}/source/unitree_rl_lab${PYTHONPATH:+:${PYTHONPATH}}"

if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" "${PYTHON_SHIM_DIR}/python"
  export PATH="${PYTHON_SHIM_DIR}:${PATH}"
fi

ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
TASK_NAME="${TASK_NAME:-Unitree-G1-29dof-TurnFineTune}"
CHECKPOINT="${CHECKPOINT:-logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-29_09-14-43_g1_29dof_turn_finetune_from_14000/model_15999.pt}"
HEADLESS="${HEADLESS:-false}"
CMD_VEL_TOPIC="${CMD_VEL_TOPIC:-/cmd_vel}"
ODOM_TOPIC="${ODOM_TOPIC:-/odom}"
MAX_VX="${MAX_VX:-0.55}"
MIN_VX="${MIN_VX:--0.10}"
MAX_ABS_VY="${MAX_ABS_VY:-0.08}"
MAX_ABS_WZ="${MAX_ABS_WZ:-0.35}"
COMMAND_TIMEOUT="${COMMAND_TIMEOUT:-2.0}"
DISABLE_COMMAND_TIMEOUT="${DISABLE_COMMAND_TIMEOUT:-false}"
CMD_UDP_PORT="${CMD_UDP_PORT:-15000}"
ODOM_UDP_PORT="${ODOM_UDP_PORT:-15001}"
MANIP_UDP_PORT="${MANIP_UDP_PORT:-15002}"
ARM_UDP_PORT="${ARM_UDP_PORT:-15003}"
ARM_MAX_DELTA_PER_STEP="${ARM_MAX_DELTA_PER_STEP:-0.003}"
BRIDGE_RATE="${BRIDGE_RATE:-50.0}"
SCENE_JSON="${SCENE_JSON:-}"

if [[ -f "${ROS_SETUP}" ]]; then
  # ROS setup scripts may read unset variables, so temporarily disable nounset.
  # shellcheck disable=SC1090
  set +u
  source "${ROS_SETUP}"
  set -u
else
  echo "[ERROR] ROS setup file not found: ${ROS_SETUP}" >&2
  exit 1
fi

ROS_BRIDGE_CMD=(python3 scripts/_internal/bridge/ros2_g1_udp_bridge.py
  --cmd_vel_topic "${CMD_VEL_TOPIC}"
  --odom_topic "${ODOM_TOPIC}"
  --cmd_udp_port "${CMD_UDP_PORT}"
  --odom_udp_port "${ODOM_UDP_PORT}"
  --rate "${BRIDGE_RATE}"
  --command_timeout "${COMMAND_TIMEOUT}"
)

ISAAC_CMD=(/workspace/isaaclab/isaaclab.sh -p scripts/_internal/bridge/play_g1_policy_cmdvel_udp.py
  --task "${TASK_NAME}"
  --num_envs 1
  --checkpoint "${CHECKPOINT}"
  --cmd_udp_port "${CMD_UDP_PORT}"
  --odom_udp_host "127.0.0.1"
  --odom_udp_port "${ODOM_UDP_PORT}"
  --manip_udp_port "${MANIP_UDP_PORT}"
  --arm_udp_port "${ARM_UDP_PORT}"
  --arm_max_delta_per_step "${ARM_MAX_DELTA_PER_STEP}"
  --min_vx "${MIN_VX}"
  --max_vx "${MAX_VX}"
  --max_abs_vy "${MAX_ABS_VY}"
  --max_abs_wz "${MAX_ABS_WZ}"
  --command_timeout "${COMMAND_TIMEOUT}"
)

if [[ -n "${SCENE_JSON}" ]]; then
  ISAAC_CMD+=(--scene_json "${SCENE_JSON}")
fi

if [[ "${DISABLE_COMMAND_TIMEOUT,,}" == "true" || "${DISABLE_COMMAND_TIMEOUT}" == "1" ]]; then
  ROS_BRIDGE_CMD+=(--disable_command_timeout)
  ISAAC_CMD+=(--disable_command_timeout)
fi

if [[ "${HEADLESS,,}" == "true" || "${HEADLESS}" == "1" ]]; then
  ISAAC_CMD+=(--headless)
  ISAAC_HEADLESS_ENV=1
else
  ISAAC_HEADLESS_ENV=0
fi

cleanup() {
  if [[ -n "${ROS_BRIDGE_PID:-}" ]] && kill -0 "${ROS_BRIDGE_PID}" 2>/dev/null; then
    kill "${ROS_BRIDGE_PID}" 2>/dev/null || true
    wait "${ROS_BRIDGE_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "task: ${TASK_NAME}"
echo "checkpoint: ${CHECKPOINT}"
echo "cmd_vel_topic: ${CMD_VEL_TOPIC}"
echo "odom_topic: ${ODOM_TOPIC}"
echo "udp: cmd_port=${CMD_UDP_PORT} odom_port=${ODOM_UDP_PORT} manip_port=${MANIP_UDP_PORT} arm_port=${ARM_UDP_PORT}"
echo "ros_home: ${ROS_HOME}"
echo "ros_log_dir: ${ROS_LOG_DIR}"
echo "pythonpath: ${PYTHONPATH}"
echo "command clamps: vx=[${MIN_VX}, ${MAX_VX}], |vy|<=${MAX_ABS_VY}, |wz|<=${MAX_ABS_WZ}"
echo "command_timeout: ${COMMAND_TIMEOUT} disable_command_timeout=${DISABLE_COMMAND_TIMEOUT}"
echo "scene_json: ${SCENE_JSON:-<none>}"
echo "ros bridge command:"
printf ' %q' "${ROS_BRIDGE_CMD[@]}"
echo
echo "isaac command:"
printf ' %q' "${ISAAC_CMD[@]}"
echo

"${ROS_BRIDGE_CMD[@]}" &
ROS_BRIDGE_PID=$!
sleep 1.0

HEADLESS="${ISAAC_HEADLESS_ENV}" "${ISAAC_CMD[@]}"
