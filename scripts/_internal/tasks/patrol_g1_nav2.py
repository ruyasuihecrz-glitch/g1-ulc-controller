#!/usr/bin/env python3
"""Patrol between Nav2 NavigateToPose goals."""

from __future__ import annotations

import os
import sys


def _ensure_system_python_for_ros2() -> None:
    """ROS 2 Jazzy rclpy is built for system Python 3.12, not Isaac's Python."""

    system_python = "/usr/bin/python3"
    if os.environ.get("G1_NAV2_PATROL_SYSTEM_PYTHON_REEXEC") == "1":
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
        env["G1_NAV2_PATROL_SYSTEM_PYTHON_REEXEC"] = "1"
        os.execve(system_python, [system_python, *sys.argv], env)


_ensure_system_python_for_ros2()

import argparse
import importlib.util
import math
import pathlib
import time
from dataclasses import dataclass


def _ensure_ros2_python_environment() -> None:
    """Re-exec through ROS setup when rclpy is not already importable."""

    if importlib.util.find_spec("rclpy") is not None:
        return
    if os.environ.get("G1_NAV2_PATROL_ROS_REEXEC") == "1":
        return

    ros_setup = os.environ.get("ROS_SETUP", "/opt/ros/jazzy/setup.bash")
    if not os.path.isfile(ros_setup):
        return

    env = os.environ.copy()
    env["G1_NAV2_PATROL_ROS_REEXEC"] = "1"
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
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def parse_waypoints(points: str) -> list[Waypoint]:
    waypoints: list[Waypoint] = []
    for item in points.split(";"):
        item = item.strip()
        if not item:
            continue
        fields = [field.strip() for field in item.split(",")]
        if len(fields) not in (2, 3):
            raise argparse.ArgumentTypeError(
                f"Invalid waypoint '{item}'. Use x,y or x,y,yaw, separated by semicolons."
            )
        try:
            x = float(fields[0])
            y = float(fields[1])
            yaw = float(fields[2]) if len(fields) == 3 else 0.0
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid numeric waypoint '{item}'.") from exc
        waypoints.append(Waypoint(x=x, y=y, yaw=yaw))

    if len(waypoints) < 2:
        raise argparse.ArgumentTypeError("Patrol needs at least two waypoints.")
    return waypoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loop Nav2 goals for a simple G1 patrol.")
    parser.add_argument(
        "--points",
        type=parse_waypoints,
        default=parse_waypoints("2.0,0.0,0.0;4.0,0.8,0.0;6.0,1.8,0.0;8.0,3.0,0.0;6.0,4.0,0.0;4.0,3.2,0.0;2.0,1.6,0.0"),
        help="Semicolon-separated waypoints: 'x,y,yaw;x,y,yaw'. Keep default empty map goals within -9..9.",
    )
    parser.add_argument("--loops", type=int, default=0, help="Number of patrol loops. 0 means forever.")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to pause after each reached waypoint.")
    parser.add_argument("--frame", type=str, default="map")
    parser.add_argument("--action_name", type=str, default="navigate_to_pose")
    parser.add_argument("--wait_timeout", type=float, default=60.0)
    parser.add_argument("--continue_on_failure", action="store_true")
    return parser.parse_args()


class PatrolRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.node = rclpy.create_node("g1_nav2_patrol")
        self.client = ActionClient(self.node, NavigateToPose, args.action_name)

    def close(self) -> None:
        self.node.destroy_node()

    def wait_for_server(self) -> None:
        self.node.get_logger().info(f"waiting for action server: {self.args.action_name}")
        if not self.client.wait_for_server(timeout_sec=self.args.wait_timeout):
            raise SystemExit(
                f"NavigateToPose action server not available after {self.args.wait_timeout}s"
            )

    def send_waypoint(self, waypoint: Waypoint, index: int, total: int) -> int:
        pose = PoseStamped()
        pose.header.frame_id = self.args.frame
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = waypoint.x
        pose.pose.position.y = waypoint.y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(waypoint.yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.node.get_logger().info(
            f"waypoint {index}/{total}: x={waypoint.x:.2f} y={waypoint.y:.2f} yaw={waypoint.yaw:.2f}"
        )

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.node.get_logger().error(f"waypoint {index}/{total} rejected")
            return GoalStatus.STATUS_ABORTED

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result()
        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.node.get_logger().info(f"waypoint {index}/{total} reached")
        else:
            self.node.get_logger().error(f"waypoint {index}/{total} failed with status={status}")
        return status


def main() -> None:
    args = parse_args()
    rclpy.init()
    runner = PatrolRunner(args)
    try:
        runner.wait_for_server()
        loop_index = 0
        while args.loops == 0 or loop_index < args.loops:
            loop_index += 1
            runner.node.get_logger().info(
                f"starting patrol loop {loop_index}{' / ' + str(args.loops) if args.loops else ''}"
            )
            for waypoint_index, waypoint in enumerate(args.points, start=1):
                status = runner.send_waypoint(waypoint, waypoint_index, len(args.points))
                if status != GoalStatus.STATUS_SUCCEEDED and not args.continue_on_failure:
                    raise SystemExit(f"patrol stopped after failed waypoint status={status}")
                if args.pause > 0.0:
                    time.sleep(args.pause)
    except KeyboardInterrupt:
        runner.node.get_logger().info("patrol interrupted")
    finally:
        runner.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
