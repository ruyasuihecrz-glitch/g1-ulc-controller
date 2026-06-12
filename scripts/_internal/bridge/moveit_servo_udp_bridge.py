#!/usr/bin/env python3
"""Forward MoveIt Servo JointTrajectory commands to Isaac over UDP."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import sys
import time


def _ensure_system_python_for_ros2() -> None:
    system_python = "/usr/bin/python3"
    if os.environ.get("G1_SERVO_UDP_BRIDGE_SYSTEM_PYTHON_REEXEC") == "1":
        return
    if sys.executable != system_python or sys.version_info[:2] != (3, 12):
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
        }
        env["G1_SERVO_UDP_BRIDGE_SYSTEM_PYTHON_REEXEC"] = "1"
        os.execve(system_python, [system_python, *sys.argv], env)


_ensure_system_python_for_ros2()

import importlib.util


def _ensure_ros2_python_environment() -> None:
    if importlib.util.find_spec("rclpy") is not None:
        return
    if os.environ.get("G1_SERVO_UDP_BRIDGE_ROS_REEXEC") == "1":
        return

    ros_setup = os.environ.get("ROS_SETUP", "/opt/ros/jazzy/setup.bash")
    if not os.path.isfile(ros_setup):
        return

    env = os.environ.copy()
    env["G1_SERVO_UDP_BRIDGE_ROS_REEXEC"] = "1"
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
from control_msgs.msg import JointJog
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.executors import ExternalShutdownException
from trajectory_msgs.msg import JointTrajectory


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge MoveIt Servo trajectory output to Isaac UDP arm override.")
    parser.add_argument("--trajectory_topic", default="/g1_right_arm_controller/joint_trajectory")
    parser.add_argument("--isaac_host", default="127.0.0.1")
    parser.add_argument("--arm_udp_port", type=int, default=15003)
    parser.add_argument("--joint_command_topic", default="/g1_servo/delta_joint_cmds")
    parser.add_argument("--twist_command_topic", default="/g1_servo/delta_twist_cmds")
    parser.add_argument("--pose_command_topic", default="/g1_servo/pose_target_cmds")
    parser.add_argument("--forward_timeout", type=float, default=0.35)
    parser.add_argument("--hold_timeout", type=float, default=0.45)
    parser.add_argument(
        "--resend_period",
        type=float,
        default=0.04,
        help="Keep resending the last Servo target so Isaac arm override does not expire.",
    )
    return parser.parse_args()


class ServoUdpBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.node = rclpy.create_node("g1_moveit_servo_udp_bridge")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (args.isaac_host, args.arm_udp_port)
        self.count = 0
        self.last_input_time = float("-inf")
        self.last_send_time = float("-inf")
        self.last_sent_positions: dict[str, float] = {}
        self.node.create_subscription(JointTrajectory, args.trajectory_topic, self._trajectory_cb, 10)
        self.node.create_subscription(JointJog, args.joint_command_topic, self._servo_input_cb, 10)
        self.node.create_subscription(TwistStamped, args.twist_command_topic, self._servo_input_cb, 10)
        self.node.create_subscription(PoseStamped, args.pose_command_topic, self._servo_input_cb, 10)
        self.node.get_logger().info(f"{args.trajectory_topic} -> UDP {self.target}")

    def _servo_input_cb(self, _msg) -> None:
        self.last_input_time = time.monotonic()

    def _trajectory_cb(self, msg: JointTrajectory) -> None:
        if time.monotonic() - self.last_input_time > self.args.forward_timeout:
            return
        if not msg.points:
            return
        point = msg.points[0]
        if not point.positions:
            return
        names = list(msg.joint_names)
        positions = [float(v) for v in point.positions]
        filtered = {
            name: positions[names.index(name)]
            for name in RIGHT_ARM_JOINTS
            if name in names and names.index(name) < len(positions)
        }
        if not filtered:
            return
        now = time.monotonic()
        if self._same_as_last_sent(filtered) and (now - self.last_send_time) < self.args.resend_period:
            return
        payload = {
            "stamp": time.time(),
            "arm_joint_positions": filtered,
            "hold_timeout": self.args.hold_timeout,
            "source": "moveit_servo",
        }
        self.sock.sendto(json.dumps(payload).encode("utf-8"), self.target)
        self.last_sent_positions = dict(filtered)
        self.last_send_time = now
        self.count += 1
        if self.count <= 8 or self.count % 100 == 0:
            compact = {name: round(value, 3) for name, value in filtered.items()}
            self.node.get_logger().info(f"sent arm override count={self.count} joints={compact}")

    def _same_as_last_sent(self, positions: dict[str, float]) -> bool:
        if positions.keys() != self.last_sent_positions.keys():
            return False
        return all(abs(value - self.last_sent_positions[name]) < 1.0e-4 for name, value in positions.items())


def main() -> None:
    args = parse_args()
    rclpy.init()
    bridge = ServoUdpBridge(args)
    try:
        rclpy.spin(bridge.node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        bridge.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
