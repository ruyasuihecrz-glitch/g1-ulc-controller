#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/data/repos/unitree_rl_lab"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Unitree G1 demo single entry.

Usage:
  bash scripts/demo.sh <command>

展示主线：
  policy-substation    启动 Isaac policy + ROS bridge + 配电站 3D 场景
  nav2-substation      启动 Nav2 配电站静态地图
  patrol-substation    启动配电站蛇形巡逻
  servo                启动 MoveIt Servo 右臂，用于 IK/Servo 操作链路
  moveit-ik            启动 MoveIt move_group，用于直接 /compute_ik 右臂操作
  button               启动按钮操作状态机，默认 ik_twist 闭环驱动右臂按第一个按钮

辅助：
  policy-empty         启动普通/空地图 policy
  nav2-empty           启动空地图 Nav2
  patrol-empty         启动空地图巡逻
  train-turn           从 14000 ckpt 继续 turn-finetune
  train-upper-body     从 15999 ckpt 继续训上半身质心变化下的站姿/步态稳定，默认 2000 轮
  train-stable         stable velocity 正式训练
  train-nav2           Nav2 formal 训练入口

常用示例：
  HEADLESS=false bash scripts/demo.sh policy-substation
  bash scripts/demo.sh nav2-substation
  bash scripts/demo.sh patrol-substation
  bash scripts/demo.sh servo
  bash scripts/demo.sh moveit-ik
  MAX_BUTTONS=1 WORK_TIMEOUT=60 bash scripts/demo.sh button
  SERVO_MODE=moveit_ik bash scripts/demo.sh button
  SERVO_MODE=direct_joint bash scripts/demo.sh button
EOF
}

command="${1:-}"
if [[ -z "${command}" || "${command}" == "-h" || "${command}" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

case "${command}" in
  policy-substation)
    exec bash scripts/_internal/entry/run_g1_policy_substation_ros2.sh "$@"
    ;;
  nav2-substation)
    exec bash scripts/_internal/entry/run_g1_nav2_substation_demo.sh "$@"
    ;;
  patrol-substation)
    exec bash scripts/_internal/entry/run_g1_substation_patrol.sh "$@"
    ;;
  servo)
    exec env -u PYTHONHOME -u PYTHONPATH -u PYTHONEXECUTABLE bash --noprofile --norc -lc \
      'source /opt/ros/jazzy/setup.bash && ros2 launch launch/g1_moveit_servo.launch.py'
    ;;
  moveit-ik)
    exec env -u PYTHONHOME -u PYTHONPATH -u PYTHONEXECUTABLE bash --noprofile --norc -lc \
      'source /opt/ros/jazzy/setup.bash && ros2 launch launch/g1_moveit_ik.launch.py'
    ;;
  button)
    export MAX_BUTTONS="${MAX_BUTTONS:-1}"
    export WORK_TIMEOUT="${WORK_TIMEOUT:-60}"
    export MAX_FINE_VX="${MAX_FINE_VX:-0.22}"
    export MAX_FINE_WZ="${MAX_FINE_WZ:-0.35}"
    export POSE_HOLD_TIME="${POSE_HOLD_TIME:-1.2}"
    export PREPRESS_RETREAT="${PREPRESS_RETREAT:-0.02}"
    export PRESS_OVERSHOOT="${PRESS_OVERSHOOT:-0.025}"
    export WORK_BASE_DISTANCE="${WORK_BASE_DISTANCE:-0.42}"
    export PRESS_LOCAL_X_MIN="${PRESS_LOCAL_X_MIN:-0.48}"
    export PRESS_LOCAL_X_MAX="${PRESS_LOCAL_X_MAX:-0.78}"
    export PRESS_LOCAL_Y_TARGET="${PRESS_LOCAL_Y_TARGET:--0.20}"
    export PRESS_LOCAL_Y_TOLERANCE="${PRESS_LOCAL_Y_TOLERANCE:-0.12}"
    export IK_TIMEOUT="${IK_TIMEOUT:-90}"
    export IK_MAX_LINEAR_SPEED="${IK_MAX_LINEAR_SPEED:-0.018}"
    export IK_GAIN="${IK_GAIN:-0.25}"
    export IK_LATERAL_DEADBAND="${IK_LATERAL_DEADBAND:-0.08}"
    export IK_LATERAL_GAIN="${IK_LATERAL_GAIN:-0.12}"
    export BASE_DRIFT_LIMIT="${BASE_DRIFT_LIMIT:-0.16}"
    export BASE_YAW_DRIFT_LIMIT="${BASE_YAW_DRIFT_LIMIT:-0.25}"
    export STABLE_BASE_HOLD="${STABLE_BASE_HOLD:-2.0}"
    exec bash scripts/_internal/entry/run_g1_button_servo_demo.sh "$@"
    ;;
  policy-empty)
    exec bash scripts/_internal/entry/run_g1_policy_cmdvel_ros2.sh "$@"
    ;;
  nav2-empty)
    exec bash scripts/_internal/entry/run_g1_nav2_empty_map_demo.sh "$@"
    ;;
  patrol-empty)
    exec bash scripts/_internal/entry/run_g1_demo_patrol.sh "$@"
    ;;
  train-turn)
    exec bash scripts/_internal/entry/train_g1_29dof_turn_finetune.sh "$@"
    ;;
  train-upper-body)
    exec bash scripts/_internal/entry/train_g1_29dof_upper_body_stability_finetune.sh "$@"
    ;;
  train-stable)
    exec bash scripts/_internal/entry/train_g1_29dof_stable_velocity.sh "$@"
    ;;
  train-nav2)
    exec bash scripts/_internal/entry/train_g1_nav2_velocity_formal.sh "$@"
    ;;
  *)
    echo "[ERROR] Unknown command: ${command}" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
