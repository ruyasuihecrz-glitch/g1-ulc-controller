#!/usr/bin/env python3
"""Nav2 + MoveIt Servo button pressing state machine for the G1 demo."""

from __future__ import annotations

import os
import sys


def _ensure_system_python_for_ros2() -> None:
    system_python = "/usr/bin/python3"
    if os.environ.get("G1_BUTTON_SERVO_SYSTEM_PYTHON_REEXEC") == "1":
        return
    has_python_env_override = any(
        name in os.environ for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
    )
    if sys.executable != system_python or sys.version_info[:2] != (3, 12) or has_python_env_override:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
        }
        env["G1_BUTTON_SERVO_SYSTEM_PYTHON_REEXEC"] = "1"
        os.execve(system_python, [system_python, *sys.argv], env)


_ensure_system_python_for_ros2()

import argparse
import importlib.util
import json
import math
import pathlib
import socket
import time
from dataclasses import dataclass


def _ensure_ros2_python_environment() -> None:
    if importlib.util.find_spec("rclpy") is not None:
        return
    if os.environ.get("G1_BUTTON_SERVO_ROS_REEXEC") == "1":
        return

    ros_setup = os.environ.get("ROS_SETUP", "/opt/ros/jazzy/setup.bash")
    if not os.path.isfile(ros_setup):
        return

    env = os.environ.copy()
    env["G1_BUTTON_SERVO_ROS_REEXEC"] = "1"
    os.execve(
        "/bin/bash",
        ["bash", "-c", 'source "$1"; shift; exec "$@"', "bash", ros_setup, sys.executable, *sys.argv],
        env,
    )


_ensure_ros2_python_environment()


def _configure_ros_runtime_dirs() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    runtime_dir = pathlib.Path(os.environ.get("RUNTIME_DIR", repo_root / ".runtime"))
    ros_home = pathlib.Path(os.environ.get("ROS_HOME", runtime_dir / "ros"))
    ros_log_dir = pathlib.Path(os.environ.get("ROS_LOG_DIR", repo_root / "logs" / "ros2"))
    ros_home.mkdir(parents=True, exist_ok=True)
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_HOME", str(ros_home))
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))


_configure_ros_runtime_dirs()

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.msg import JointJog
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
try:
    from moveit_msgs.msg import MoveItErrorCodes, RobotState
    from moveit_msgs.srv import GetPositionIK, ServoCommandType
except ModuleNotFoundError:
    MoveItErrorCodes = None
    RobotState = None
    GetPositionIK = None
    ServoCommandType = None
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate_point_by_quat(point: list[float], quat_xyzw: tuple[float, float, float, float]) -> list[float]:
    x, y, z, w = quat_xyzw
    px, py, pz = point
    # q * p * q^-1，展开写避免额外依赖 tf_transformations。
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    return [
        px + w * tx + (y * tz - z * ty),
        py + w * ty + (z * tx - x * tz),
        pz + w * tz + (x * ty - y * tx),
    ]


def pose_from_list(values: list[float]) -> Pose2D:
    return Pose2D(float(values[0]), float(values[1]), float(values[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Navigate to cabinet buttons and press them with MoveIt Servo.")
    parser.add_argument("--scene_json", type=pathlib.Path, default=pathlib.Path("maps/nav2_substation/substation_room_scene.json"))
    parser.add_argument("--frame", default="map")
    parser.add_argument("--action_name", default="navigate_to_pose")
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel")
    parser.add_argument("--nav_cmd_vel_topic", default="/cmd_vel_nav")
    parser.add_argument("--odom_topic", default="/odom")
    parser.add_argument("--servo_twist_topic", default="/g1_servo/delta_twist_cmds")
    parser.add_argument("--servo_joint_topic", default="/g1_servo/delta_joint_cmds")
    parser.add_argument("--servo_pose_topic", default="/g1_servo/pose_target_cmds")
    parser.add_argument(
        "--servo_mode",
        choices=("ik_twist", "moveit_ik", "direct_joint", "pose_target", "joint_jog", "twist"),
        default="ik_twist",
    )
    parser.add_argument("--compute_ik_service", default="/compute_ik")
    parser.add_argument("--servo_node_name", default="/g1_moveit_servo")
    parser.add_argument("--servo_bridge_node_name", default="g1_moveit_servo_udp_bridge")
    parser.add_argument("--skip_servo_bridge_check", action="store_true")
    parser.add_argument("--servo_frame", default="torso_link")
    parser.add_argument("--ee_frame", default="right_rubber_hand")
    parser.add_argument("--disable_tf_point_transform", action="store_true")
    parser.add_argument("--wait_timeout", type=float, default=90.0)
    parser.add_argument("--nav_retries", type=int, default=1)
    parser.add_argument("--skip_nav", action="store_true", help="调试操作段时跳过 Nav2 action，直接进入精定位和按钮按压。")
    parser.add_argument(
        "--continue_on_button_nav_failure",
        action="store_true",
        help="If a button approach Nav2 goal aborts near the cabinet, continue with slow fine approach.",
    )
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--max_buttons", type=int, default=0, help="0 means all buttons.")
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--press_speed", type=float, default=0.045, help="Forward Servo speed in the EE frame, m/s.")
    parser.add_argument("--press_time", type=float, default=1.0)
    parser.add_argument("--retreat_speed", type=float, default=0.040)
    parser.add_argument("--retreat_time", type=float, default=0.8)
    parser.add_argument("--joint_press_speed", type=float, default=0.035)
    parser.add_argument("--joint_press_time", type=float, default=0.45)
    parser.add_argument(
        "--work_base_distance",
        type=float,
        default=0.62,
        help="Deprecated fallback distance along the button normal. The fine approach now primarily uses torso local x/y targets.",
    )
    parser.add_argument("--work_xy_tolerance", type=float, default=0.08)
    parser.add_argument("--work_yaw_tolerance", type=float, default=0.10)
    parser.add_argument("--work_timeout", type=float, default=18.0)
    parser.add_argument("--max_fine_vx", type=float, default=0.16)
    parser.add_argument("--max_fine_vy", type=float, default=0.04)
    parser.add_argument("--max_fine_wz", type=float, default=0.32)
    parser.add_argument(
        "--min_fine_vx",
        type=float,
        default=0.12,
        help="Minimum non-zero forward speed during close base alignment; helps overcome policy deadband.",
    )
    parser.add_argument(
        "--press_local_x_min",
        type=float,
        default=0.22,
        help="Button press_pose lower bound in torso_link x after base fine approach.",
    )
    parser.add_argument(
        "--press_local_x_max",
        type=float,
        default=0.38,
        help="Button press_pose upper bound in torso_link x after base fine approach.",
    )
    parser.add_argument("--press_local_y_target", type=float, default=-0.16)
    parser.add_argument("--press_local_y_tolerance", type=float, default=0.24)
    parser.add_argument(
        "--press_local_z_min",
        type=float,
        default=0.18,
        help="Button press_pose lower bound in torso_link z after base fine approach.",
    )
    parser.add_argument(
        "--press_local_z_max",
        type=float,
        default=0.46,
        help="Button press_pose upper bound in torso_link z after base fine approach.",
    )
    parser.add_argument("--torso_z_offset", type=float, default=0.044)
    parser.add_argument("--press_overshoot", type=float, default=0.04)
    parser.add_argument("--prepress_retreat", type=float, default=0.16)
    parser.add_argument("--pose_hold_time", type=float, default=1.2)
    parser.add_argument("--pose_rate", type=float, default=30.0)
    parser.add_argument("--ik_rate", type=float, default=35.0)
    parser.add_argument("--ik_timeout", type=float, default=5.0)
    parser.add_argument("--ik_tolerance", type=float, default=0.035)
    parser.add_argument("--ik_contact_tolerance", type=float, default=0.028)
    parser.add_argument(
        "--contact_validate_tolerance",
        type=float,
        default=0.07,
        help="Maximum EE-to-contact-target distance before emitting the Isaac button-pressed event.",
    )
    parser.add_argument(
        "--ik_joint_tolerance",
        type=float,
        default=0.16,
        help="MoveIt 直接 IK 模式下，右臂关节反馈和 IK 目标的最大允许误差(rad)。这个值用于执行闭环，不依赖末端 TF。",
    )
    parser.add_argument("--ik_max_linear_speed", type=float, default=0.075)
    parser.add_argument("--ik_gain", type=float, default=0.85)
    parser.add_argument(
        "--ik_lateral_deadband",
        type=float,
        default=0.08,
        help="按钮按压 IK 的侧向死区。按钮有宽度，侧向误差在死区内时不强追，避免 KDL Servo 把右臂带向坏姿态。",
    )
    parser.add_argument(
        "--ik_lateral_gain",
        type=float,
        default=0.12,
        help="侧向误差超过死区后的控制权重。越大越努力修正左右偏差，但越容易扰动底盘。",
    )
    parser.add_argument(
        "--base_drift_limit",
        type=float,
        default=0.16,
        help="操作阶段允许的底盘平面漂移上限，超过后立刻停止手臂，避免边按边挪导致失稳。",
    )
    parser.add_argument(
        "--base_yaw_drift_limit",
        type=float,
        default=0.25,
        help="操作阶段允许的底盘 yaw 漂移上限，超过后立刻停止手臂。",
    )
    parser.add_argument("--stable_base_hold", type=float, default=1.0)
    parser.add_argument("--manip_udp_host", default="127.0.0.1")
    parser.add_argument("--manip_udp_port", type=int, default=15002)
    parser.add_argument("--arm_udp_host", default="127.0.0.1")
    parser.add_argument("--arm_udp_port", type=int, default=15003)
    parser.add_argument("--print_only", action="store_true")
    return parser.parse_args()


class ButtonServoStateMachine:
    RIGHT_ARM_JOINTS = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    def __init__(self, args: argparse.Namespace, buttons: list[dict], home_goal: Pose2D | None):
        self.args = args
        self.buttons = buttons
        self.home_goal = home_goal
        self.node = rclpy.create_node("g1_button_servo_state_machine")
        self.nav_client = ActionClient(self.node, NavigateToPose, args.action_name)
        self.cmd_pub = self.node.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.nav_cmd_pub = self.node.create_publisher(Twist, args.nav_cmd_vel_topic, 10)
        self.servo_pub = self.node.create_publisher(TwistStamped, args.servo_twist_topic, 10)
        self.joint_servo_pub = self.node.create_publisher(JointJog, args.servo_joint_topic, 10)
        self.pose_servo_pub = self.node.create_publisher(PoseStamped, args.servo_pose_topic, 10)
        self.latest_odom: Odometry | None = None
        self.latest_joint_state: JointState | None = None
        self.node.create_subscription(Odometry, args.odom_topic, self._odom_cb, 10)
        self.node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self._tf_fallback_warned = False
        self.manip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.manip_target = (args.manip_udp_host, args.manip_udp_port)
        self.arm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.arm_target = (args.arm_udp_host, args.arm_udp_port)
        self.pause_client = self.node.create_client(SetBool, f"{args.servo_node_name}/pause_servo")
        self.command_type_client = (
            self.node.create_client(ServoCommandType, f"{args.servo_node_name}/switch_command_type")
            if ServoCommandType is not None
            else None
        )
        self.compute_ik_client = (
            self.node.create_client(GetPositionIK, args.compute_ik_service)
            if GetPositionIK is not None
            else None
        )

    def close(self) -> None:
        self.node.destroy_node()

    def run(self) -> None:
        if self.args.skip_nav:
            self.node.get_logger().warn("--skip_nav 已启用，跳过 Nav2 action server 等待")
        else:
            self.node.get_logger().info(f"等待 Nav2 action server: {self.args.action_name}")
            if not self.nav_client.wait_for_server(timeout_sec=self.args.wait_timeout):
                raise SystemExit(f"NavigateToPose action server not available after {self.args.wait_timeout}s")
        self._wait_for_odom()
        self._configure_servo()

        loop_index = 0
        while self.args.loops == 0 or loop_index < self.args.loops:
            loop_index += 1
            for button_index, button in enumerate(self.buttons, start=1):
                pose = pose_from_list(button["approach_goal"])
                if self.args.skip_nav:
                    nav_ok = True
                    self.node.get_logger().warn(
                        f"[BUTTON_{button_index}/{button['name']}] SKIP_NAV=true，跳过 Nav2 action，直接进入精定位"
                    )
                else:
                    nav_ok = self._nav_with_retry(f"BUTTON_{button_index}/{button['name']}", pose)
                if not nav_ok:
                    if not self.args.continue_on_button_nav_failure:
                        raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] 导航多次失败")
                    self.node.get_logger().warn(
                        f"[BUTTON_{button_index}/{button['name']}] Nav2 approach 失败，"
                        "继续使用低速精定位兜底"
                    )
                self._stop_base(0.8)
                if not self._drive_to_button_work_pose(button_index, button):
                    raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] 按钮仍在右臂可达范围外，停止演示")
                self._stop_base(0.5)
                self._press_button(button_index, button)
                self._drive_to_pose(f"BUTTON_{button_index}/BACK_OUT", pose, timeout_s=self.args.work_timeout)
                if self.args.pause > 0.0:
                    self._stop_base(self.args.pause)
            if self.home_goal is not None and not self.args.skip_nav:
                if not self._nav_with_retry("RETURN_HOME", self.home_goal):
                    raise SystemExit("[RETURN_HOME] 导航多次失败")
            self.node.get_logger().info("按钮 Servo 状态机循环完成")

    def _nav_with_retry(self, label: str, pose: Pose2D) -> bool:
        total_attempts = max(1, self.args.nav_retries + 1)
        if self.args.nav_retries < 0:
            self.node.get_logger().warn(
                f"[{label}] nav_retries={self.args.nav_retries} 会被按 1 次导航处理；"
                "需要无限重试时请用外层脚本循环，避免现场卡死。"
            )
        for attempt in range(total_attempts):
            self.node.get_logger().info(
                f"[{label}] 导航第 {attempt + 1}/{total_attempts} 次: "
                f"x={pose.x:.2f} y={pose.y:.2f} yaw={pose.yaw:.2f}"
            )
            status = self._send_nav_goal(pose)
            if status == GoalStatus.STATUS_SUCCEEDED:
                return True
            self.node.get_logger().warn(f"[{label}] 导航失败 status={status}")
            self._stop_base(0.6)
        return False

    def _send_nav_goal(self, pose2d: Pose2D) -> int:
        pose = PoseStamped()
        pose.header.frame_id = self.args.frame
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = pose2d.x
        pose.pose.position.y = pose2d.y
        qx, qy, qz, qw = yaw_to_quat(pose2d.yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            return GoalStatus.STATUS_ABORTED
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        return int(result_future.result().status)

    def _stop_base(self, duration_s: float) -> None:
        msg = Twist()
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            self._publish_base_stop()
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.04)

    def _publish_base_stop(self) -> None:
        msg = Twist()
        self._publish_base_cmd(msg)

    def _publish_base_cmd(self, msg: Twist) -> None:
        self.cmd_pub.publish(msg)
        self.nav_cmd_pub.publish(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _joint_state_cb(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def _wait_for_odom(self) -> None:
        deadline = time.monotonic() + self.args.wait_timeout
        while self.latest_odom is None and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.latest_odom is None:
            raise SystemExit(f"Odometry topic not available after {self.args.wait_timeout}s: {self.args.odom_topic}")

    def _wait_for_joint_state(self) -> None:
        deadline = time.monotonic() + self.args.wait_timeout
        while self.latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.latest_joint_state is None:
            raise SystemExit(f"JointState topic not available after {self.args.wait_timeout}s: /joint_states")

    def _configure_servo(self) -> None:
        if self.args.servo_mode == "direct_joint":
            self.node.get_logger().info(
                f"右臂操作使用 direct_joint UDP 模式 -> {self.arm_target}，绕过 MoveIt IK 奇异点缩放"
            )
            return
        if self.args.servo_mode == "moveit_ik":
            if GetPositionIK is None or RobotState is None or MoveItErrorCodes is None:
                raise SystemExit("moveit_msgs is not available; cannot use --servo_mode moveit_ik")
            self._wait_for_joint_state()
            assert self.compute_ik_client is not None
            if not self.compute_ik_client.wait_for_service(timeout_sec=8.0):
                raise SystemExit(
                    f"MoveIt compute IK service not available: {self.args.compute_ik_service}. "
                    "请先运行 `bash scripts/demo.sh moveit-ik`。"
                )
            self.node.get_logger().info(
                f"右臂操作使用 MoveIt 直接 IK 服务 {self.args.compute_ik_service} -> UDP {self.arm_target}"
            )
            return
        command_type_by_mode = {"joint_jog": 0, "twist": 1, "ik_twist": 1, "pose_target": 2}
        if ServoCommandType is None:
            raise SystemExit(f"moveit_msgs is not available; cannot use --servo_mode {self.args.servo_mode}")
        command_type = command_type_by_mode[self.args.servo_mode]
        assert self.command_type_client is not None
        if not self.pause_client.wait_for_service(timeout_sec=5.0):
            raise SystemExit(f"MoveIt Servo pause service not available: {self.pause_client.srv_name}")
        if not self.command_type_client.wait_for_service(timeout_sec=5.0):
            raise SystemExit(f"MoveIt Servo command type service not available: {self.command_type_client.srv_name}")
        if not self.args.skip_servo_bridge_check:
            self._wait_for_servo_udp_bridge()

        pause_req = SetBool.Request()
        pause_req.data = False
        pause_future = self.pause_client.call_async(pause_req)
        rclpy.spin_until_future_complete(self.node, pause_future)

        type_req = ServoCommandType.Request()
        type_req.command_type = command_type
        type_future = self.command_type_client.call_async(type_req)
        rclpy.spin_until_future_complete(self.node, type_future)
        self.node.get_logger().info(f"MoveIt Servo 已切到 {self.args.servo_mode} 模式")

    def _wait_for_servo_udp_bridge(self) -> None:
        expected = self.args.servo_bridge_node_name.lstrip("/")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            node_names = {name.lstrip("/") for name in self.node.get_node_names()}
            if expected in node_names:
                self.node.get_logger().info(f"MoveIt Servo UDP bridge 已连接: /{expected}")
                return
            rclpy.spin_once(self.node, timeout_sec=0.1)
        raise SystemExit(
            f"MoveIt Servo UDP bridge node not found: /{expected}. "
            "请先运行 `bash scripts/demo.sh servo`，并确认它启动的是 "
            "scripts/_internal/bridge/moveit_servo_udp_bridge.py"
        )

    def _press_button(self, button_index: int, button: dict) -> None:
        self.node.get_logger().info(
            f"[SERVO_PRESS/{button_index}] {button['name']} press_pose={button['press_pose']}"
        )
        if self.args.servo_mode == "pose_target":
            self._servo_press_pose_target(button_index, button)
            return
        if self.args.servo_mode == "moveit_ik":
            self._moveit_ik_press(button_index, button)
            return
        if self.args.servo_mode == "ik_twist":
            self._servo_press_ik_twist(button_index, button)
            return
        if self.args.servo_mode == "direct_joint":
            self._direct_joint_press(button_index, button)
            return
        if self.args.servo_mode == "twist":
            self._servo_x(self.args.press_speed, self.args.press_time)
            self._servo_x(0.0, 0.25)
            self._servo_x(-self.args.retreat_speed, self.args.retreat_time)
            self._servo_x(0.0, 0.25)
            return

        speed = self.args.joint_press_speed
        duration = self.args.joint_press_time
        # 这个动作是演示优先的 Servo 动作：抬右臂、肘部前伸、手腕点按、再收回。
        # 如果按钮高度或机器人站位变化，优先调 joint_press_speed 和 joint_press_time。
        self._servo_joints(
            {
                "right_shoulder_pitch_joint": speed,
                "right_elbow_joint": -speed,
                "right_wrist_pitch_joint": 0.45 * speed,
            },
            duration,
        )
        self._servo_joints({"right_wrist_pitch_joint": -0.5 * speed}, 0.25)
        self._servo_joints({"right_wrist_pitch_joint": 0.5 * speed}, 0.25)
        self._servo_joints(
            {
                "right_shoulder_pitch_joint": -speed,
                "right_elbow_joint": speed,
                "right_wrist_pitch_joint": -0.45 * speed,
            },
            duration,
        )

    def _drive_to_button_work_pose(self, button_index: int, button: dict) -> bool:
        press_x, press_y, _ = [float(v) for v in button["press_pose"]]
        _, _, safe_yaw = [float(v) for v in button["approach_goal"]]
        target_local_x = 0.5 * (self.args.press_local_x_min + self.args.press_local_x_max)
        target_local_y = self.args.press_local_y_target
        cos_yaw = math.cos(safe_yaw)
        sin_yaw = math.sin(safe_yaw)
        # Choose the base pose that places the button at the desired torso-local
        # x/y. This is the critical handoff: base moves first, then freezes for IK.
        work = Pose2D(
            press_x - (cos_yaw * target_local_x - sin_yaw * target_local_y),
            press_y - (sin_yaw * target_local_x + cos_yaw * target_local_y),
            safe_yaw,
        )
        self.node.get_logger().info(
            f"[BUTTON_{button_index}/FINE_APPROACH] 工作站位 x={work.x:.2f} y={work.y:.2f} yaw={work.yaw:.2f} "
            f"(目标 torso_press=[{target_local_x:.2f}, {target_local_y:.2f}])"
        )
        return self._drive_until_button_reachable(button_index, button, work)

    def _drive_until_button_reachable(self, button_index: int, button: dict, work: Pose2D) -> bool:
        label = f"BUTTON_{button_index}/FINE_APPROACH"
        deadline = time.monotonic() + self.args.work_timeout
        target_local_x = 0.5 * (self.args.press_local_x_min + self.args.press_local_x_max)
        last_log_time = 0.0
        while time.monotonic() < deadline:
            current = self._current_pose2d()
            press_local = self._map_point_to_torso([float(v) for v in button["press_pose"]])
            dx = work.x - current.x
            dy = work.y - current.y
            distance_to_work = math.hypot(dx, dy)
            yaw_error = wrap_to_pi(work.yaw - current.yaw)
            x_ok = self.args.press_local_x_min <= press_local[0] <= self.args.press_local_x_max
            y_error = press_local[1] - self.args.press_local_y_target
            y_ok = abs(y_error) <= self.args.press_local_y_tolerance
            z_ok = self.args.press_local_z_min <= press_local[2] <= self.args.press_local_z_max
            yaw_ok = abs(yaw_error) <= self.args.work_yaw_tolerance
            if x_ok and y_ok and z_ok and yaw_ok:
                self._stop_base(0.25)
                self.node.get_logger().info(
                    f"[{label}] 按钮进入右臂可达窗口 torso={press_local} "
                    f"x范围=[{self.args.press_local_x_min:.2f},{self.args.press_local_x_max:.2f}] "
                    f"y目标={self.args.press_local_y_target:.2f} "
                    f"z范围=[{self.args.press_local_z_min:.2f},{self.args.press_local_z_max:.2f}]"
                )
                return True

            now = time.monotonic()
            if now - last_log_time > 2.0:
                last_log_time = now
                self.node.get_logger().info(
                    f"[{label}] 调整中 torso_press=[{press_local[0]:.3f}, {press_local[1]:.3f}, {press_local[2]:.3f}] "
                    f"odom=[{current.x:.2f}, {current.y:.2f}, {current.yaw:.2f}] "
                    f"work_dist={distance_to_work:.2f} yaw_err={yaw_error:.2f}"
                )

            msg = Twist()
            if abs(yaw_error) > self.args.work_yaw_tolerance:
                msg.angular.z = max(-self.args.max_fine_wz, min(self.args.max_fine_wz, 1.35 * yaw_error))
            elif press_local[0] > self.args.press_local_x_max:
                forward = 0.85 * (press_local[0] - target_local_x)
                msg.linear.x = max(self.args.min_fine_vx, min(self.args.max_fine_vx, forward))
            elif press_local[0] < self.args.press_local_x_min:
                backward = 0.65 * (press_local[0] - target_local_x)
                msg.linear.x = max(-self.args.max_fine_vx, min(-self.args.min_fine_vx, backward))
            elif not y_ok:
                # The policy has little reliable lateral authority. Once x is in
                # range, use yaw-only nudges, then re-check x after the next step.
                y_correction = max(0.08, min(0.18, 1.20 * abs(y_error)))
                msg.angular.z = math.copysign(y_correction, y_error)
            elif distance_to_work > self.args.work_xy_tolerance:
                msg = self._unicycle_cmd_to_xy(current, work.x, work.y)
            else:
                msg.linear.x = max(
                    -self.args.max_fine_vx,
                    min(self.args.max_fine_vx, 0.75 * (press_local[0] - target_local_x)),
                )
            self._publish_base_cmd(msg)
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.04)

        self._stop_base(0.5)
        current = self._current_pose2d()
        press_local = self._map_point_to_torso([float(v) for v in button["press_pose"]])
        self.node.get_logger().warn(
            f"[{label}] 精定位超时，当前 x={current.x:.2f} y={current.y:.2f} yaw={current.yaw:.2f} "
            f"press_torso=[{press_local[0]:.3f}, {press_local[1]:.3f}, {press_local[2]:.3f}]"
        )
        return False

    def _drive_to_pose(self, label: str, target: Pose2D, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            current = self._current_pose2d()
            dx = target.x - current.x
            dy = target.y - current.y
            yaw_error = wrap_to_pi(target.yaw - current.yaw)
            cos_yaw = math.cos(current.yaw)
            sin_yaw = math.sin(current.yaw)
            err_x_b = cos_yaw * dx + sin_yaw * dy
            err_y_b = -sin_yaw * dx + cos_yaw * dy
            distance = math.hypot(dx, dy)
            if distance <= self.args.work_xy_tolerance and abs(yaw_error) <= self.args.work_yaw_tolerance:
                self._stop_base(0.25)
                return True

            if distance > self.args.work_xy_tolerance:
                msg = self._unicycle_cmd_to_xy(current, target.x, target.y)
            else:
                msg = Twist()
                msg.angular.z = max(-self.args.max_fine_wz, min(self.args.max_fine_wz, 1.35 * yaw_error))
            self._publish_base_cmd(msg)
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.04)

        self._stop_base(0.5)
        current = self._current_pose2d()
        self.node.get_logger().warn(
            f"[{label}] 精定位超时，当前 x={current.x:.2f} y={current.y:.2f} yaw={current.yaw:.2f}"
        )
        return False

    def _unicycle_cmd_to_xy(self, current: Pose2D, target_x: float, target_y: float) -> Twist:
        dx = target_x - current.x
        dy = target_y - current.y
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_to_pi(desired_yaw - current.yaw)
        msg = Twist()
        if abs(yaw_error) > 0.28:
            msg.angular.z = max(-self.args.max_fine_wz, min(self.args.max_fine_wz, 1.20 * yaw_error))
            return msg
        msg.linear.x = max(0.0, min(self.args.max_fine_vx, 0.55 * distance))
        msg.angular.z = max(-0.18, min(0.18, 0.80 * yaw_error))
        return msg

    def _current_pose2d(self) -> Pose2D:
        if self.latest_odom is None:
            self._wait_for_odom()
        assert self.latest_odom is not None
        pose = self.latest_odom.pose.pose
        return Pose2D(
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quat_xyzw(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )

    def _servo_press_ik_twist(self, button_index: int, button: dict) -> None:
        self._stop_base(self.args.stable_base_hold)
        press_local = self._map_point_to_torso([float(v) for v in button["press_pose"]])
        ee_local = self._current_ee_in_servo_frame()
        self.node.get_logger().info(
            f"[BUTTON_{button_index}/IK_TWIST] button_torso={press_local} "
            f"ee_torso={ee_local} base冻结={self.args.stable_base_hold:.1f}s"
        )

        normal = [float(v) for v in button.get("normal", [0.0, 0.0, 0.0])]
        press_pose = [float(v) for v in button["press_pose"]]
        pre_pose = [
            press_pose[0] + normal[0] * self.args.prepress_retreat,
            press_pose[1] + normal[1] * self.args.prepress_retreat,
            press_pose[2],
        ]
        contact_pose = [
            press_pose[0] - normal[0] * self.args.press_overshoot,
            press_pose[1] - normal[1] * self.args.press_overshoot,
            press_pose[2],
        ]
        pre_torso = self._map_point_to_torso(pre_pose)
        contact_torso = self._map_point_to_torso(contact_pose)
        self.node.get_logger().info(
            f"[BUTTON_{button_index}/IK_LOCK] 按压目标已锁定在 {self.args.servo_frame}: "
            f"pre={ [round(v, 3) for v in pre_torso] } contact={ [round(v, 3) for v in contact_torso] }"
        )

        reached_prepress = self._servo_ik_to_torso_point(
            f"BUTTON_{button_index}/IK_PREPRESS",
            pre_torso,
            timeout_s=self.args.ik_timeout,
            tolerance=self.args.ik_tolerance,
        )
        if not reached_prepress:
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] IK 未能到达按钮预接触点")

        reached_contact = self._servo_ik_to_torso_point(
            f"BUTTON_{button_index}/IK_PRESS",
            contact_torso,
            timeout_s=max(1.5, 0.6 * self.args.ik_timeout),
            tolerance=self.args.ik_contact_tolerance,
        )
        if not reached_contact:
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] IK 未能到达按钮按压点")

        self._stop_arm_servo(0.25)
        self._validate_contact_before_notify(f"BUTTON_{button_index}/IK_PRESS_VALIDATE", contact_torso)
        self._notify_button_pressed(button)
        self._servo_ik_to_torso_point(
            f"BUTTON_{button_index}/IK_RELEASE",
            pre_torso,
            timeout_s=2.5,
            tolerance=self.args.ik_tolerance,
        )
        self._stop_arm_servo(0.2)
        self._stop_base(0.4)

    def _servo_ik_to_map_point(self, label: str, target_map: list[float], timeout_s: float, tolerance: float) -> bool:
        return self._servo_ik_to_torso_point(label, self._map_point_to_torso(target_map), timeout_s, tolerance)

    def _servo_ik_to_torso_point(self, label: str, target_torso: list[float], timeout_s: float, tolerance: float) -> bool:
        deadline = time.monotonic() + timeout_s
        period = 1.0 / max(1.0, self.args.ik_rate)
        last_log_time = 0.0
        base_anchor = self._current_pose2d()
        while time.monotonic() < deadline:
            current = self._current_ee_in_servo_frame()
            error = [target_torso[i] - current[i] for i in range(3)]
            lateral_overflow = max(0.0, abs(error[1]) - self.args.ik_lateral_deadband)
            distance = math.sqrt(error[0] * error[0] + error[2] * error[2] + lateral_overflow * lateral_overflow)
            if distance <= tolerance:
                self._stop_arm_servo(0.15)
                self.node.get_logger().info(
                    f"[{label}] 到达 IK 目标 distance={distance:.3f} "
                    f"target_torso={[round(v, 3) for v in target_torso]} ee_torso={[round(v, 3) for v in current]}"
                )
                return True

            base_now = self._current_pose2d()
            base_drift = math.hypot(base_now.x - base_anchor.x, base_now.y - base_anchor.y)
            yaw_drift = abs(wrap_to_pi(base_now.yaw - base_anchor.yaw))
            if base_drift > self.args.base_drift_limit or yaw_drift > self.args.base_yaw_drift_limit:
                self._stop_arm_servo(0.35)
                self._stop_base(0.5)
                self.node.get_logger().warn(
                    f"[{label}] 底盘漂移过大，停止 IK: drift={base_drift:.3f} yaw_drift={yaw_drift:.3f} "
                    f"anchor=[{base_anchor.x:.2f}, {base_anchor.y:.2f}, {base_anchor.yaw:.2f}] "
                    f"now=[{base_now.x:.2f}, {base_now.y:.2f}, {base_now.yaw:.2f}]"
                )
                return False

            now = time.monotonic()
            if now - last_log_time > 0.5:
                last_log_time = now
                self.node.get_logger().info(
                    f"[{label}] IK 闭环中 distance={distance:.3f} "
                    f"target_torso={[round(v, 3) for v in target_torso]} ee_torso={[round(v, 3) for v in current]} "
                    f"err={[round(v, 3) for v in error]}"
                )

            speed = min(self.args.ik_max_linear_speed, self.args.ik_gain * distance)
            controlled_error = [error[0], 0.0, error[2]]
            if abs(error[1]) > self.args.ik_lateral_deadband:
                controlled_error[1] = self.args.ik_lateral_gain * (
                    error[1] - math.copysign(self.args.ik_lateral_deadband, error[1])
                )
            controlled_norm = math.sqrt(sum(value * value for value in controlled_error))
            if controlled_norm < 1.0e-6:
                self._stop_arm_servo(0.15)
                return False
            scale = speed / controlled_norm
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.args.servo_frame
            msg.twist.linear.x = controlled_error[0] * scale
            msg.twist.linear.y = controlled_error[1] * scale
            msg.twist.linear.z = controlled_error[2] * scale
            self.servo_pub.publish(msg)
            self._publish_base_stop()
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(period)

        self._stop_arm_servo(0.2)
        current = self._current_ee_in_servo_frame()
        distance = math.sqrt(sum((target_torso[i] - current[i]) ** 2 for i in range(3)))
        self.node.get_logger().warn(
            f"[{label}] IK 超时 distance={distance:.3f} "
            f"target_torso={[round(v, 3) for v in target_torso]} ee_torso={[round(v, 3) for v in current]}"
        )
        return False

    def _current_ee_in_servo_frame(self) -> list[float]:
        deadline = time.monotonic() + 2.0
        last_exc: TransformException | None = None
        while time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.servo_frame,
                    self.args.ee_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.15),
                )
                t = transform.transform.translation
                return [float(t.x), float(t.y), float(t.z)]
            except TransformException as exc:
                last_exc = exc
                rclpy.spin_once(self.node, timeout_sec=0.05)
                time.sleep(0.03)
        if last_exc is not None:
            raise SystemExit(
                f"无法读取末端 TF {self.args.servo_frame}->{self.args.ee_frame}: {last_exc}. "
                "请确认 policy bridge 发布 base_link->pelvis，且 robot_state_publisher 正在发布 URDF TF。"
            ) from last_exc
        raise SystemExit(f"无法读取末端 TF {self.args.servo_frame}->{self.args.ee_frame}: timeout")

    def _stop_arm_servo(self, duration_s: float) -> None:
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.args.servo_frame
            self.servo_pub.publish(msg)
            self._publish_base_stop()
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.03)

    def _direct_joint_press(self, button_index: int, button: dict) -> None:
        press_local = self._map_point_to_torso([float(v) for v in button["press_pose"]])
        self.node.get_logger().info(
            f"[BUTTON_{button_index}/DIRECT_JOINT] 使用右臂关节目标按压，当前按钮 torso={press_local}"
        )

        # 这些角度是给演示用的稳定按压姿态：先伸到按钮前，再继续前探一点，最后收回到默认垂臂附近。
        # 调参时优先改 shoulder_roll/elbow/wrist_pitch：roll 越负越往机器人右侧伸，elbow 越大越前探。
        prepress = {
            "right_shoulder_pitch_joint": 0.78,
            "right_shoulder_roll_joint": -0.58,
            "right_shoulder_yaw_joint": 0.25,
            "right_elbow_joint": 1.28,
            "right_wrist_roll_joint": -0.22,
            "right_wrist_pitch_joint": -0.32,
            "right_wrist_yaw_joint": 0.18,
        }
        press = {
            "right_shoulder_pitch_joint": 1.08,
            "right_shoulder_roll_joint": -0.88,
            "right_shoulder_yaw_joint": 0.52,
            "right_elbow_joint": 1.72,
            "right_wrist_roll_joint": -0.62,
            "right_wrist_pitch_joint": -0.74,
            "right_wrist_yaw_joint": 0.42,
        }
        release = {
            "right_shoulder_pitch_joint": 0.30,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.97,
            "right_wrist_roll_joint": -0.15,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

        self._send_direct_arm_target(f"BUTTON_{button_index}/PREPRESS_ARM", prepress, 0.9)
        self._send_direct_arm_target(f"BUTTON_{button_index}/PRESS_ARM", press, max(0.9, self.args.pose_hold_time))
        self.node.get_logger().warn(
            f"[BUTTON_{button_index}/DIRECT_JOINT] direct_joint 不再自动发送 BUTTON_PRESSED；"
            "它没有 IK/末端几何闭环，容易产生假成功。"
        )
        self._send_direct_arm_target(f"BUTTON_{button_index}/RELEASE_ARM", prepress, 0.55)
        self._send_direct_arm_target(f"BUTTON_{button_index}/STOW_ARM", release, 0.9)

    def _moveit_ik_press(self, button_index: int, button: dict) -> None:
        self._stop_base(self.args.stable_base_hold)
        press_local = self._map_point_to_torso([float(v) for v in button["press_pose"]])
        self.node.get_logger().info(
            f"[BUTTON_{button_index}/MOVEIT_IK] button_torso={press_local} base冻结={self.args.stable_base_hold:.1f}s"
        )

        normal = [float(v) for v in button.get("normal", [0.0, 0.0, 0.0])]
        press_pose = [float(v) for v in button["press_pose"]]
        pre_pose = [
            press_pose[0] + normal[0] * self.args.prepress_retreat,
            press_pose[1] + normal[1] * self.args.prepress_retreat,
            press_pose[2],
        ]
        contact_pose = [
            press_pose[0] - normal[0] * self.args.press_overshoot,
            press_pose[1] - normal[1] * self.args.press_overshoot,
            press_pose[2],
        ]
        pre_torso = self._map_point_to_torso(pre_pose)
        contact_torso = self._map_point_to_torso(contact_pose)
        self.node.get_logger().info(
            f"[BUTTON_{button_index}/MOVEIT_IK_LOCK] pre={ [round(v, 3) for v in pre_torso] } "
            f"contact={ [round(v, 3) for v in contact_torso] }"
        )

        pre_joints = self._compute_moveit_ik(f"BUTTON_{button_index}/IK_PREPRESS", pre_torso)
        if pre_joints is None:
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] MoveIt IK 未找到预接触解")
        if not self._send_ik_arm_target_until_near(
            f"BUTTON_{button_index}/IK_PREPRESS_EXEC",
            pre_joints,
            pre_torso,
            timeout_s=self.args.ik_timeout,
            tolerance=self.args.ik_tolerance,
        ):
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] 右臂未能执行到预接触点")

        contact_joints = self._compute_moveit_ik(f"BUTTON_{button_index}/IK_PRESS", contact_torso)
        if contact_joints is None:
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] MoveIt IK 未找到按压解")
        if not self._send_ik_arm_target_until_near(
            f"BUTTON_{button_index}/IK_PRESS_EXEC",
            contact_joints,
            contact_torso,
            timeout_s=max(2.0, 0.6 * self.args.ik_timeout),
            tolerance=self.args.ik_contact_tolerance,
        ):
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] 右臂未能执行到按钮按压点")

        self._validate_contact_before_notify(f"BUTTON_{button_index}/IK_PRESS_VALIDATE", contact_torso)
        self._notify_button_pressed(button)
        self._send_ik_arm_target_until_near(
            f"BUTTON_{button_index}/IK_RELEASE_EXEC",
            pre_joints,
            pre_torso,
            timeout_s=3.0,
            tolerance=self.args.ik_tolerance,
        )
        self._stop_base(0.4)

    def _compute_moveit_ik(self, label: str, target_torso: list[float]) -> dict[str, float] | None:
        self._wait_for_joint_state()
        assert self.latest_joint_state is not None
        current_joints = {
            name: float(position)
            for name, position in zip(self.latest_joint_state.name, self.latest_joint_state.position, strict=False)
        }
        orientations = [
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.7071068, 0.7071068),
            (0.0, 0.0, -0.7071068, 0.7071068),
            (0.7071068, 0.0, 0.0, 0.7071068),
            (0.0, -0.7071068, 0.0, 0.7071068),
        ]
        candidates: list[tuple[float, float, dict[str, float], tuple[float, float, float, float]]] = []
        failures: list[int | str] = []
        for orientation in orientations:
            req = GetPositionIK.Request()
            req.ik_request.group_name = "right_arm"
            req.ik_request.ik_link_name = self.args.ee_frame
            req.ik_request.avoid_collisions = False
            req.ik_request.robot_state = RobotState()
            req.ik_request.robot_state.joint_state = self.latest_joint_state
            req.ik_request.pose_stamped.header.frame_id = self.args.servo_frame
            req.ik_request.pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
            req.ik_request.pose_stamped.pose.position.x = float(target_torso[0])
            req.ik_request.pose_stamped.pose.position.y = float(target_torso[1])
            req.ik_request.pose_stamped.pose.position.z = float(target_torso[2])
            req.ik_request.pose_stamped.pose.orientation.x = orientation[0]
            req.ik_request.pose_stamped.pose.orientation.y = orientation[1]
            req.ik_request.pose_stamped.pose.orientation.z = orientation[2]
            req.ik_request.pose_stamped.pose.orientation.w = orientation[3]
            req.ik_request.timeout = Duration(sec=0, nanosec=int(0.25 * 1.0e9))

            future = self.compute_ik_client.call_async(req)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
            if not future.done() or future.result() is None:
                failures.append("timeout")
                continue
            result = future.result()
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                failures.append(int(result.error_code.val))
                continue
            names = list(result.solution.joint_state.name)
            positions = list(result.solution.joint_state.position)
            joints = {
                name: float(positions[names.index(name)])
                for name in self.RIGHT_ARM_JOINTS
                if name in names and names.index(name) < len(positions)
            }
            if len(joints) != len(self.RIGHT_ARM_JOINTS):
                self.node.get_logger().warn(f"[{label}] IK 解缺少右臂关节: {joints}")
                continue
            errors = [abs(joints[name] - current_joints.get(name, joints[name])) for name in self.RIGHT_ARM_JOINTS]
            candidates.append((max(errors), sum(errors), joints, orientation))

        if not candidates:
            self.node.get_logger().warn(f"[{label}] MoveIt IK 失败 codes={failures}")
            return None
        _, _, joints, orientation = min(candidates, key=lambda item: (item[0], item[1]))
        self.node.get_logger().info(
            f"[{label}] MoveIt IK 解 orientation={tuple(round(v, 3) for v in orientation)} "
            f"解={ {name: round(value, 3) for name, value in joints.items()} } "
            f"candidates={len(candidates)} failures={failures}"
        )
        return joints

    def _send_ik_arm_target_until_near(
        self,
        label: str,
        joints: dict[str, float],
        target_torso: list[float],
        timeout_s: float,
        tolerance: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        last_log_time = 0.0
        base_anchor = self._current_pose2d()
        payload = {
            "stamp": time.time(),
            "arm_joint_positions": joints,
            "hold_timeout": 0.5,
            "source": "button_state_machine_moveit_ik",
        }
        sent_count = 0
        while time.monotonic() < deadline:
            payload["stamp"] = time.time()
            self.arm_sock.sendto(json.dumps(payload).encode("utf-8"), self.arm_target)
            sent_count += 1
            rclpy.spin_once(self.node, timeout_sec=0.01)
            joint_error = self._right_arm_joint_error(joints)
            if joint_error <= self.args.ik_joint_tolerance:
                self.node.get_logger().info(
                    f"[{label}] 右臂已跟踪 MoveIt IK 关节目标 joint_error={joint_error:.3f} "
                    f"target_torso={[round(v, 3) for v in target_torso]} arm_udp_sent={sent_count}"
                )
                return True

            base_now = self._current_pose2d()
            base_drift = math.hypot(base_now.x - base_anchor.x, base_now.y - base_anchor.y)
            yaw_drift = abs(wrap_to_pi(base_now.yaw - base_anchor.yaw))
            if base_drift > self.args.base_drift_limit or yaw_drift > self.args.base_yaw_drift_limit:
                self._stop_base(0.5)
                self.node.get_logger().warn(
                    f"[{label}] 底盘漂移过大，停止 IK 执行: drift={base_drift:.3f} yaw_drift={yaw_drift:.3f}"
                )
                return False

            self._publish_base_stop()
            now = time.monotonic()
            if now - last_log_time > 0.7:
                last_log_time = now
                self.node.get_logger().info(
                    f"[{label}] 执行中 joint_error={joint_error:.3f} "
                    f"target_torso={[round(v, 3) for v in target_torso]} "
                    f"arm_udp_sent={sent_count}"
                )
            time.sleep(0.03)

        joint_error = self._right_arm_joint_error(joints)
        self.node.get_logger().warn(
            f"[{label}] 执行超时 joint_error={joint_error:.3f} "
            f"target_torso={[round(v, 3) for v in target_torso]} arm_udp_sent={sent_count}"
        )
        return False

    def _right_arm_joint_error(self, target_joints: dict[str, float]) -> float:
        self._wait_for_joint_state()
        assert self.latest_joint_state is not None
        current = {
            name: float(position)
            for name, position in zip(self.latest_joint_state.name, self.latest_joint_state.position, strict=False)
        }
        errors = [
            abs(float(target) - current[name])
            for name, target in target_joints.items()
            if name in current
        ]
        if len(errors) != len(target_joints):
            return float("inf")
        return max(errors)

    def _send_direct_arm_target(self, label: str, joints: dict[str, float], duration_s: float) -> None:
        payload = {
            "stamp": time.time(),
            "arm_joint_positions": joints,
            "hold_timeout": 0.35,
            "source": "button_state_machine_direct_joint",
        }
        end = time.monotonic() + duration_s
        count = 0
        while time.monotonic() < end:
            payload["stamp"] = time.time()
            self.arm_sock.sendto(json.dumps(payload).encode("utf-8"), self.arm_target)
            count += 1
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.02)
        self.node.get_logger().info(f"[{label}] sent direct arm target packets={count} joints={joints}")

    def _servo_press_pose_target(self, button_index: int, button: dict) -> None:
        self._stop_base(self.args.stable_base_hold)
        press_local = self._map_point_to_torso([float(v) for v in button["press_pose"]])
        if not (
            self.args.press_local_x_min <= press_local[0] <= self.args.press_local_x_max
            and abs(press_local[1] - self.args.press_local_y_target) <= self.args.press_local_y_tolerance
        ):
            raise SystemExit(
                f"[BUTTON_{button_index}/{button['name']}] press_pose 不可达: torso={press_local}, "
                f"x范围=[{self.args.press_local_x_min:.2f},{self.args.press_local_x_max:.2f}]"
            )
        normal = [float(v) for v in button.get("normal", [0.0, 0.0, 0.0])]
        press_pose = [float(v) for v in button["press_pose"]]
        pre_pose = [
            press_pose[0] + normal[0] * self.args.prepress_retreat,
            press_pose[1] + normal[1] * self.args.prepress_retreat,
            press_pose[2],
        ]
        contact_pose = [
            press_pose[0] - normal[0] * self.args.press_overshoot,
            press_pose[1] - normal[1] * self.args.press_overshoot,
            press_pose[2],
        ]
        pre_torso = self._map_point_to_torso(pre_pose)
        contact_torso = self._map_point_to_torso(contact_pose)
        self.node.get_logger().info(
            f"[BUTTON_{button_index}/POSE_LOCK] pose target 已锁定在 {self.args.servo_frame}: "
            f"pre={ [round(v, 3) for v in pre_torso] } contact={ [round(v, 3) for v in contact_torso] }"
        )
        reached_prepress = self._publish_pose_target_until_near(
            f"BUTTON_{button_index}/PREPRESS",
            pre_torso,
            timeout_s=self.args.ik_timeout,
            tolerance=self.args.ik_tolerance,
        )
        if not reached_prepress:
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] pose target 未能到达按钮预接触点")
        reached_contact = self._publish_pose_target_until_near(
            f"BUTTON_{button_index}/PRESS",
            contact_torso,
            timeout_s=max(2.0, 0.6 * self.args.ik_timeout),
            tolerance=self.args.ik_contact_tolerance,
        )
        if not reached_contact:
            raise SystemExit(f"[BUTTON_{button_index}/{button['name']}] pose target 未能到达按钮按压点")
        self._validate_contact_before_notify(f"BUTTON_{button_index}/POSE_PRESS_VALIDATE", contact_torso)
        self._notify_button_pressed(button)
        self._publish_pose_target_sequence(f"BUTTON_{button_index}/RELEASE", pre_torso, 0.7)

    def _validate_contact_before_notify(self, label: str, contact_torso: list[float]) -> None:
        ee_torso = self._current_ee_in_servo_frame()
        distance = math.sqrt(sum((contact_torso[i] - ee_torso[i]) ** 2 for i in range(3)))
        if distance > self.args.contact_validate_tolerance:
            raise SystemExit(
                f"[{label}] 末端没有真实到达按压点，拒绝发送 BUTTON_PRESSED: "
                f"distance={distance:.3f} tolerance={self.args.contact_validate_tolerance:.3f} "
                f"contact_torso={[round(v, 3) for v in contact_torso]} ee_torso={[round(v, 3) for v in ee_torso]}"
            )
        self.node.get_logger().info(
            f"[{label}] 接触几何验证通过 distance={distance:.3f} "
            f"contact_torso={[round(v, 3) for v in contact_torso]} ee_torso={[round(v, 3) for v in ee_torso]}"
        )

    def _notify_button_pressed(self, button: dict) -> None:
        payload = {
            "action": "press_button",
            "button": button["name"],
            "request_id": f"press_{button['name']}_{int(time.time() * 1000)}",
        }
        self.manip_sock.sendto(json.dumps(payload).encode("utf-8"), self.manip_target)
        self.node.get_logger().info(f"[BUTTON_PRESSED] {button['name']} 已发送 Isaac 按钮反馈事件")

    def _publish_pose_target_sequence(self, label: str, target_torso: list[float], duration_s: float) -> None:
        end = time.monotonic() + duration_s
        period = 1.0 / max(1.0, self.args.pose_rate)
        while time.monotonic() < end:
            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.args.servo_frame
            msg.pose.position.x = target_torso[0]
            msg.pose.position.y = target_torso[1]
            msg.pose.position.z = target_torso[2]
            msg.pose.orientation.w = 1.0
            self.pose_servo_pub.publish(msg)
            self._publish_base_stop()
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(period)
        self.node.get_logger().info(f"[{label}] pose_target torso={target_torso}")

    def _publish_pose_target_until_near(
        self,
        label: str,
        target_torso: list[float],
        timeout_s: float,
        tolerance: float,
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        period = 1.0 / max(1.0, self.args.pose_rate)
        last_log_time = 0.0
        base_anchor = self._current_pose2d()
        while time.monotonic() < deadline:
            current = self._current_ee_in_servo_frame()
            lateral_overflow = max(0.0, abs(target_torso[1] - current[1]) - self.args.ik_lateral_deadband)
            distance = math.sqrt(
                (target_torso[0] - current[0]) ** 2
                + (target_torso[2] - current[2]) ** 2
                + lateral_overflow * lateral_overflow
            )
            if distance <= tolerance:
                self.node.get_logger().info(
                    f"[{label}] pose target 到达 distance={distance:.3f} "
                    f"target_torso={[round(v, 3) for v in target_torso]} ee_torso={[round(v, 3) for v in current]}"
                )
                return True

            base_now = self._current_pose2d()
            base_drift = math.hypot(base_now.x - base_anchor.x, base_now.y - base_anchor.y)
            yaw_drift = abs(wrap_to_pi(base_now.yaw - base_anchor.yaw))
            if base_drift > self.args.base_drift_limit or yaw_drift > self.args.base_yaw_drift_limit:
                self._stop_arm_servo(0.35)
                self._stop_base(0.5)
                self.node.get_logger().warn(
                    f"[{label}] 底盘漂移过大，停止 pose target: drift={base_drift:.3f} yaw_drift={yaw_drift:.3f}"
                )
                return False

            now = time.monotonic()
            if now - last_log_time > 0.7:
                last_log_time = now
                self.node.get_logger().info(
                    f"[{label}] pose target 闭环中 distance={distance:.3f} "
                    f"target_torso={[round(v, 3) for v in target_torso]} ee_torso={[round(v, 3) for v in current]}"
                )

            msg = PoseStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.args.servo_frame
            msg.pose.position.x = target_torso[0]
            msg.pose.position.y = target_torso[1]
            msg.pose.position.z = target_torso[2]
            msg.pose.orientation.w = 1.0
            self.pose_servo_pub.publish(msg)
            self._publish_base_stop()
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(period)

        current = self._current_ee_in_servo_frame()
        distance = math.sqrt(sum((target_torso[i] - current[i]) ** 2 for i in range(3)))
        self.node.get_logger().warn(
            f"[{label}] pose target 超时 distance={distance:.3f} "
            f"target_torso={[round(v, 3) for v in target_torso]} ee_torso={[round(v, 3) for v in current]}"
        )
        return False

    def _map_point_to_torso(self, point_map: list[float]) -> list[float]:
        if not self.args.disable_tf_point_transform:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.servo_frame,
                    self.args.frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
                t = transform.transform.translation
                q = transform.transform.rotation
                rotated = rotate_point_by_quat(
                    [float(point_map[0]), float(point_map[1]), float(point_map[2])],
                    (float(q.x), float(q.y), float(q.z), float(q.w)),
                )
                return [rotated[0] + float(t.x), rotated[1] + float(t.y), rotated[2] + float(t.z)]
            except TransformException as exc:
                if not self._tf_fallback_warned:
                    self._tf_fallback_warned = True
                    self.node.get_logger().warn(
                        f"TF2 暂时无法计算 {self.args.frame}->{self.args.servo_frame}: {exc}; "
                        "临时退回 odom 平面手算。请确认 TF 树为 map->odom->base_link->pelvis->torso_link。"
                    )

        if self.latest_odom is None:
            self._wait_for_odom()
        assert self.latest_odom is not None
        pose = self.latest_odom.pose.pose
        base_x = float(pose.position.x)
        base_y = float(pose.position.y)
        base_z = float(pose.position.z)
        base_yaw = yaw_from_quat_xyzw(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        dx = float(point_map[0]) - base_x
        dy = float(point_map[1]) - base_y
        cos_yaw = math.cos(base_yaw)
        sin_yaw = math.sin(base_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        local_z = float(point_map[2]) - base_z - self.args.torso_z_offset
        return [local_x, local_y, local_z]

    def _servo_x(self, linear_x: float, duration_s: float) -> None:
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            msg = TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.args.servo_frame
            msg.twist.linear.x = float(linear_x)
            self.servo_pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.02)

    def _servo_joints(self, velocities_by_joint: dict[str, float], duration_s: float) -> None:
        names = list(velocities_by_joint)
        velocities = [float(velocities_by_joint[name]) for name in names]
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            msg = JointJog()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = self.args.servo_frame
            msg.joint_names = names
            msg.velocities = velocities
            msg.duration = 0.1
            self.joint_servo_pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.01)
            time.sleep(0.02)


def main() -> None:
    args = parse_args()
    scene = json.loads(args.scene_json.read_text(encoding="utf-8"))
    buttons = scene["button_demo"]["buttons"]
    if args.max_buttons > 0:
        buttons = buttons[: args.max_buttons]
    home = scene["button_demo"].get("home_goal")
    home_goal = pose_from_list(home) if home else None
    print(f"scene_json: {args.scene_json}")
    for index, button in enumerate(buttons, start=1):
        print(index, button["name"], "approach=", button["approach_goal"], "press=", button["press_pose"])
    if args.print_only:
        return
    rclpy.init()
    sm = ButtonServoStateMachine(args, buttons, home_goal)
    try:
        sm.run()
    except KeyboardInterrupt:
        sm.node.get_logger().info("按钮 Servo 状态机被中断")
    finally:
        sm.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
