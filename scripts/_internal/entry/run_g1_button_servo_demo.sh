#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

PRINT_ONLY="${PRINT_ONLY:-false}"
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

MAP_OUTPUT="$(python3 scripts/_internal/maps/create_substation_nav2_map.py)"
SCENE_JSON="$(printf '%s\n' "${MAP_OUTPUT}" | sed -n '2p')"

CMD=(/usr/bin/python3 scripts/_internal/tasks/g1_button_servo_state_machine.py
  --scene_json "${SCENE_JSON}"
  --loops "${LOOPS:-1}"
  --max_buttons "${MAX_BUTTONS:-0}"
  --wait_timeout "${WAIT_TIMEOUT:-90}"
  --nav_retries "${NAV_RETRIES:-1}"
  --servo_mode "${SERVO_MODE:-ik_twist}"
  --press_speed "${PRESS_SPEED:-0.045}"
  --press_time "${PRESS_TIME:-1.0}"
  --retreat_speed "${RETREAT_SPEED:-0.040}"
  --retreat_time "${RETREAT_TIME:-0.8}"
  --pause "${PAUSE:-1.0}"
  --joint_press_speed "${JOINT_PRESS_SPEED:-0.035}"
  --joint_press_time "${JOINT_PRESS_TIME:-0.45}"
  --work_base_distance "${WORK_BASE_DISTANCE:-0.42}"
  --work_xy_tolerance "${WORK_XY_TOLERANCE:-0.08}"
  --work_yaw_tolerance "${WORK_YAW_TOLERANCE:-0.10}"
  --work_timeout "${WORK_TIMEOUT:-18.0}"
  --max_fine_vx "${MAX_FINE_VX:-0.16}"
  --max_fine_wz "${MAX_FINE_WZ:-0.32}"
  --press_local_x_min "${PRESS_LOCAL_X_MIN:-0.48}"
  --press_local_x_max "${PRESS_LOCAL_X_MAX:-0.78}"
  --press_local_y_target "${PRESS_LOCAL_Y_TARGET:--0.20}"
  --press_local_y_tolerance "${PRESS_LOCAL_Y_TOLERANCE:-0.12}"
  --pose_hold_time "${POSE_HOLD_TIME:-1.2}"
  --ik_timeout "${IK_TIMEOUT:-90}"
  --ik_tolerance "${IK_TOLERANCE:-0.035}"
  --ik_contact_tolerance "${IK_CONTACT_TOLERANCE:-0.028}"
  --ik_joint_tolerance "${IK_JOINT_TOLERANCE:-0.16}"
  --ik_max_linear_speed "${IK_MAX_LINEAR_SPEED:-0.018}"
  --ik_gain "${IK_GAIN:-0.25}"
  --ik_lateral_deadband "${IK_LATERAL_DEADBAND:-0.08}"
  --ik_lateral_gain "${IK_LATERAL_GAIN:-0.12}"
  --base_drift_limit "${BASE_DRIFT_LIMIT:-0.16}"
  --base_yaw_drift_limit "${BASE_YAW_DRIFT_LIMIT:-0.25}"
  --stable_base_hold "${STABLE_BASE_HOLD:-2.0}"
  --prepress_retreat "${PREPRESS_RETREAT:-0.02}"
  --press_overshoot "${PRESS_OVERSHOOT:-0.025}"
  --disable_tf_point_transform
  --continue_on_button_nav_failure
  --arm_udp_host "${ARM_UDP_HOST:-127.0.0.1}"
  --arm_udp_port "${ARM_UDP_PORT:-15003}"
)

if [[ "${PRINT_ONLY}" == "true" ]]; then
  CMD+=(--print_only)
fi
if [[ "${SKIP_NAV:-false}" == "true" ]]; then
  CMD+=(--skip_nav)
fi

CMD+=("${EXTRA_ARGS[@]}")

env -u PYTHONHOME -u PYTHONPATH -u PYTHONEXECUTABLE bash --noprofile --norc -lc \
  'source /opt/ros/jazzy/setup.bash && "$@"' \
  bash "${CMD[@]}"
