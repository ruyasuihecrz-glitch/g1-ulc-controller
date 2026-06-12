#!/usr/bin/env python3
"""Nav2 + scripted tabletop manipulation state machine for the G1 demo."""

from __future__ import annotations

import os
import sys


def _ensure_system_python_for_ros2() -> None:
    system_python = "/usr/bin/python3"
    if os.environ.get("G1_PICK_PLACE_SYSTEM_PYTHON_REEXEC") == "1":
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
        env["G1_PICK_PLACE_SYSTEM_PYTHON_REEXEC"] = "1"
        os.execve(system_python, [system_python, *sys.argv], env)


_ensure_system_python_for_ros2()

import argparse
import importlib.util
import json
import math
import pathlib
import socket
import time
import uuid
from dataclasses import dataclass


def _ensure_ros2_python_environment() -> None:
    if importlib.util.find_spec("rclpy") is not None:
        return
    if os.environ.get("G1_PICK_PLACE_ROS_REEXEC") == "1":
        return

    ros_setup = os.environ.get("ROS_SETUP", "/opt/ros/jazzy/setup.bash")
    if not os.path.isfile(ros_setup):
        return

    env = os.environ.copy()
    env["G1_PICK_PLACE_ROS_REEXEC"] = "1"
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
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def pose_from_list(values: list[float]) -> Pose2D:
    if len(values) < 3:
        raise ValueError(f"pose needs [x, y, yaw], got: {values}")
    return Pose2D(float(values[0]), float(values[1]), float(values[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G1 Nav2 table pick-place-return demo state machine.")
    parser.add_argument("--scene_json", type=pathlib.Path, default=pathlib.Path("maps/nav2_substation/substation_room_scene.json"))
    parser.add_argument("--frame", type=str, default="map")
    parser.add_argument("--action_name", type=str, default="navigate_to_pose")
    parser.add_argument("--wait_timeout", type=float, default=90.0)
    parser.add_argument("--nav_retries", type=int, default=1, help="Retries per Nav2 goal after the first attempt.")
    parser.add_argument("--loops", type=int, default=1, help="Task loops. 0 means forever.")
    parser.add_argument("--return_x", type=float, default=0.8)
    parser.add_argument("--return_y", type=float, default=0.0)
    parser.add_argument("--return_yaw", type=float, default=0.0)
    parser.add_argument("--skip_return", action="store_true")
    parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
    parser.add_argument("--manip_udp_host", type=str, default="127.0.0.1")
    parser.add_argument("--manip_udp_port", type=int, default=15002)
    parser.add_argument("--manip_timeout", type=float, default=20.0)
    parser.add_argument("--manip_duration_scale", type=float, default=1.0)
    parser.add_argument("--print_only", action="store_true")
    return parser.parse_args()


def load_demo_poses(scene_json: pathlib.Path) -> tuple[Pose2D, Pose2D]:
    scene = json.loads(scene_json.read_text(encoding="utf-8"))
    table = scene["manipulation"]["table"]
    return pose_from_list(table["approach_goal"]), pose_from_list(table["front_goal"])


class PickPlaceStateMachine:
    def __init__(self, args: argparse.Namespace, approach: Pose2D, front: Pose2D) -> None:
        self.args = args
        self.approach = approach
        self.front = front
        self.home = Pose2D(args.return_x, args.return_y, args.return_yaw)
        self.node = rclpy.create_node("g1_nav_pick_place_state_machine")
        self.client = ActionClient(self.node, NavigateToPose, args.action_name)
        self.cmd_pub = self.node.create_publisher(Twist, args.cmd_vel_topic, 10)

    def close(self) -> None:
        self.node.destroy_node()

    def wait_for_server(self) -> None:
        self.node.get_logger().info(f"等待 Nav2 action server: {self.args.action_name}")
        if not self.client.wait_for_server(timeout_sec=self.args.wait_timeout):
            raise SystemExit(f"NavigateToPose action server not available after {self.args.wait_timeout}s")

    def run(self) -> None:
        self.wait_for_server()
        loop_index = 0
        while self.args.loops == 0 or loop_index < self.args.loops:
            loop_index += 1
            loop_text = f"{loop_index}" if self.args.loops == 0 else f"{loop_index}/{self.args.loops}"
            self.node.get_logger().info(f"========== 状态机任务循环 {loop_text} ==========")
            self._nav_with_retry("APPROACH_TABLE/先到桌子外侧缓冲点", self.approach)
            self._nav_with_retry("ALIGN_TABLE/贴近桌前操作位", self.front)
            self._stop_base("STOP_BASE/停稳底盘", duration_s=1.0)
            self._run_manipulation()
            self._stop_base("STOP_AFTER_MANIP/搬运后再次停稳", duration_s=0.6)
            if not self.args.skip_return:
                self._nav_with_retry("RETURN_HOME/返回起点附近", self.home)
            self.node.get_logger().info("任务循环完成")

    def _nav_with_retry(self, state_name: str, pose: Pose2D) -> None:
        for attempt in range(self.args.nav_retries + 1):
            attempt_text = f"第 {attempt + 1}/{self.args.nav_retries + 1} 次"
            self.node.get_logger().info(
                f"[{state_name}] {attempt_text}: x={pose.x:.2f} y={pose.y:.2f} yaw={pose.yaw:.2f}"
            )
            status = self._send_nav_goal(pose)
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.node.get_logger().info(f"[{state_name}] 到达成功")
                return
            self.node.get_logger().warn(f"[{state_name}] 到达失败 status={status}")
            self._stop_base("RECOVERY_STOP/失败后先清零速度", duration_s=0.8)
        raise SystemExit(f"[{state_name}] 多次尝试后仍失败，停止状态机")

    def _send_nav_goal(self, pose2d: Pose2D) -> int:
        pose = PoseStamped()
        pose.header.frame_id = self.args.frame
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = pose2d.x
        pose.pose.position.y = pose2d.y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(pose2d.yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal = NavigateToPose.Goal()
        goal.pose = pose
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.node, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return GoalStatus.STATUS_ABORTED

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self.node, result_future)
        result = result_future.result()
        return int(result.status)

    def _stop_base(self, state_name: str, duration_s: float, *, log: bool = True) -> None:
        if log:
            self.node.get_logger().info(f"[{state_name}] 发布零速度 {duration_s:.1f}s")
        msg = Twist()
        end_time = time.monotonic() + duration_s
        while time.monotonic() < end_time:
            self.cmd_pub.publish(msg)
            rclpy.spin_once(self.node, timeout_sec=0.02)
            time.sleep(0.05)

    def _run_manipulation(self) -> None:
        request_id = str(uuid.uuid4())
        self.node.get_logger().info("[MANIPULATE/桌面搬运] 触发 Isaac scripted pick_place")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reply_sock:
            reply_sock.bind(("127.0.0.1", 0))
            reply_sock.settimeout(0.5)
            reply_host, reply_port = reply_sock.getsockname()
            payload = {
                "action": "pick_place",
                "request_id": request_id,
                "reply_host": reply_host,
                "reply_port": reply_port,
                "duration_scale": self.args.manip_duration_scale,
            }
            command_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                command_sock.sendto(
                    json.dumps(payload).encode("utf-8"),
                    (self.args.manip_udp_host, self.args.manip_udp_port),
                )
            finally:
                command_sock.close()

            deadline = time.monotonic() + self.args.manip_timeout
            accepted = False
            while time.monotonic() < deadline:
                self._stop_base("MANIPULATE/搬运期间保持底盘停止", duration_s=0.08, log=False)
                try:
                    data, _ = reply_sock.recvfrom(65535)
                except socket.timeout:
                    continue
                status = json.loads(data.decode("utf-8"))
                if status.get("request_id") != request_id:
                    continue
                state = status.get("status", "")
                message = status.get("message", "")
                self.node.get_logger().info(f"[MANIPULATE/桌面搬运] Isaac 回包: {state} {message}")
                if state == "accepted":
                    accepted = True
                elif state == "done":
                    return
                elif state == "error":
                    raise SystemExit(f"Isaac manipulation error: {message}")

            hint = "已发送但没有 done 回包" if accepted else "没有收到 accepted 回包"
            raise SystemExit(f"[MANIPULATE/桌面搬运] 超时：{hint}")


def main() -> None:
    args = parse_args()
    if not args.scene_json.is_file():
        raise SystemExit(f"scene_json not found: {args.scene_json}")
    approach, front = load_demo_poses(args.scene_json)
    print(f"scene_json: {args.scene_json}")
    print(f"approach: x={approach.x:.2f} y={approach.y:.2f} yaw={approach.yaw:.2f}")
    print(f"front: x={front.x:.2f} y={front.y:.2f} yaw={front.yaw:.2f}")
    print(f"return: x={args.return_x:.2f} y={args.return_y:.2f} yaw={args.return_yaw:.2f}")
    if args.print_only:
        return

    rclpy.init()
    sm = PickPlaceStateMachine(args, approach, front)
    try:
        sm.run()
    except KeyboardInterrupt:
        sm.node.get_logger().info("状态机被手动中断")
    finally:
        sm.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
