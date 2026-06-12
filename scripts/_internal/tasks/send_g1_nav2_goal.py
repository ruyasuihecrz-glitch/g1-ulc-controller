#!/usr/bin/env python3
"""Send a NavigateToPose goal to Nav2."""

from __future__ import annotations

import os
import sys


def _ensure_system_python_for_ros2() -> None:
    """ROS 2 Jazzy rclpy is built for system Python 3.12, not Isaac's Python 3.11."""

    system_python = "/usr/bin/python3"
    if os.environ.get("G1_NAV2_GOAL_SYSTEM_PYTHON_REEXEC") == "1":
        return

    has_isaac_python_paths = any("_isaac_sim/kit/python" in path for path in sys.path)
    has_python_env_override = any(
        name in os.environ for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
    )
    if sys.executable != system_python or sys.version_info[:2] != (3, 12) or has_isaac_python_paths or has_python_env_override:
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE")
        }
        env["G1_NAV2_GOAL_SYSTEM_PYTHON_REEXEC"] = "1"
        os.execve(system_python, [system_python, *sys.argv], env)


_ensure_system_python_for_ros2()

import argparse
import importlib.util
import math
import pathlib


def _ensure_ros2_python_environment() -> None:
    """Re-exec through ROS setup when rclpy is not already importable."""

    if importlib.util.find_spec("rclpy") is not None:
        return
    if os.environ.get("G1_NAV2_GOAL_ROS_REEXEC") == "1":
        return

    ros_setup = os.environ.get("ROS_SETUP", "/opt/ros/jazzy/setup.bash")
    if not os.path.isfile(ros_setup):
        return

    env = os.environ.copy()
    env["G1_NAV2_GOAL_ROS_REEXEC"] = "1"
    os.execve(
        "/bin/bash",
        ["bash", "-c", 'source "$1"; shift; exec "$@"', "bash", ros_setup, sys.executable, *sys.argv],
        env,
    )


_ensure_ros2_python_environment()


def _configure_ros_runtime_dirs() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    runtime_dir = pathlib.Path(os.environ.get("RUNTIME_DIR", repo_root / ".runtime"))
    ros_home = pathlib.Path(os.environ.get("ROS_HOME", runtime_dir / "ros"))
    ros_log_dir = pathlib.Path(os.environ.get("ROS_LOG_DIR", repo_root / "logs" / "ros2"))
    ros_home.mkdir(parents=True, exist_ok=True)
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ROS_HOME", str(ros_home))
    os.environ.setdefault("ROS_LOG_DIR", str(ros_log_dir))


_configure_ros_runtime_dirs()

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a Nav2 NavigateToPose goal.")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--frame", type=str, default="map")
    parser.add_argument("--action_name", type=str, default="navigate_to_pose")
    parser.add_argument("--wait_timeout", type=float, default=10.0)
    return parser.parse_args()


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("g1_nav2_goal_sender")
    client = ActionClient(node, NavigateToPose, args.action_name)

    node.get_logger().info(f"waiting for action server: {args.action_name}")
    if not client.wait_for_server(timeout_sec=args.wait_timeout):
        raise SystemExit(f"NavigateToPose action server not available after {args.wait_timeout}s")

    pose = PoseStamped()
    pose.header.frame_id = args.frame
    pose.header.stamp = node.get_clock().now().to_msg()
    pose.pose.position.x = args.x
    pose.pose.position.y = args.y
    pose.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_to_quat(args.yaw)
    pose.pose.orientation.x = qx
    pose.pose.orientation.y = qy
    pose.pose.orientation.z = qz
    pose.pose.orientation.w = qw

    goal = NavigateToPose.Goal()
    goal.pose = pose
    node.get_logger().info(f"sending goal x={args.x:.2f} y={args.y:.2f} yaw={args.yaw:.2f}")
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise SystemExit("goal rejected")

    node.get_logger().info("goal accepted")
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result()
    node.get_logger().info(f"goal result status={result.status}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
