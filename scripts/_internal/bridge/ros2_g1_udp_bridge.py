#!/usr/bin/env python3
"""ROS 2 <-> UDP bridge for the Isaac Lab G1 policy process.

ROS 2 Jazzy on Ubuntu 24.04 installs ``rclpy`` for system Python 3.12, while
Isaac Sim currently runs its own Python 3.11.  Keeping ROS 2 in this separate
system-Python process avoids importing ``rclpy`` inside Isaac.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import socket
import time


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
from rclpy.executors import ExternalShutdownException
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge ROS 2 cmd_vel and Isaac Lab policy UDP messages.")
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel")
    parser.add_argument("--odom_topic", default="/odom")
    parser.add_argument("--joint_states_topic", default="/joint_states")
    parser.add_argument("--odom_frame", default="odom")
    parser.add_argument("--base_frame", default="base_link")
    parser.add_argument("--pelvis_frame", default="pelvis")
    parser.add_argument("--isaac_host", default="127.0.0.1")
    parser.add_argument("--cmd_udp_port", type=int, default=15000)
    parser.add_argument("--odom_udp_port", type=int, default=15001)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--command_timeout", type=float, default=2.0)
    parser.add_argument("--disable_command_timeout", action="store_true", default=False)
    parser.add_argument(
        "--use_world_odom",
        action="store_true",
        default=False,
        help="Publish Isaac world pose directly instead of zeroing odom at bridge startup.",
    )
    return parser.parse_args()


def make_stamp(sim_time: float) -> Time:
    stamp = Time()
    stamp.sec = int(sim_time)
    stamp.nanosec = int((sim_time - stamp.sec) * 1.0e9)
    return stamp


def yaw_from_quat_wxyz(quat_wxyz: list[float]) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_xyzw_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return (0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class G1UdpRosBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.node = rclpy.create_node("g1_policy_udp_ros_bridge")
        self.cmd_sub = self.node.create_subscription(Twist, args.cmd_vel_topic, self._cmd_callback, 10)
        self.odom_pub = self.node.create_publisher(Odometry, args.odom_topic, 10)
        self.joint_state_pub = self.node.create_publisher(JointState, args.joint_states_topic, 10)
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.tf_broadcaster = TransformBroadcaster(self.node)

        self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_target = (args.isaac_host, args.cmd_udp_port)
        self.odom_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.odom_socket.bind(("0.0.0.0", args.odom_udp_port))
        self.odom_socket.setblocking(False)

        self.latest_cmd = [0.0, 0.0, 0.0]
        self.last_cmd_time = float("-inf")
        self.received_cmd_count = 0
        self.sent_cmd_count = 0
        self.received_odom_count = 0
        self.odom_origin: tuple[float, float, float] | None = None

        period = 1.0 / args.rate
        self.timer = self.node.create_timer(period, self._timer_callback)
        self.node.get_logger().info(
            f"ROS cmd_vel {args.cmd_vel_topic} -> UDP {self.cmd_target}; "
            f"UDP :{args.odom_udp_port} -> ROS {args.odom_topic}, {args.joint_states_topic}, /tf, /clock"
        )

    def _cmd_callback(self, msg: Twist) -> None:
        self.latest_cmd = [float(msg.linear.x), float(msg.linear.y), float(msg.angular.z)]
        self.last_cmd_time = time.monotonic()
        self.received_cmd_count += 1
        if self.received_cmd_count <= 5 or self.received_cmd_count % 50 == 0:
            self.node.get_logger().info(f"received cmd_vel count={self.received_cmd_count} cmd={self.latest_cmd}")

    def _timer_callback(self) -> None:
        self._send_command()
        self._drain_odom()

    def _send_command(self) -> None:
        age = time.monotonic() - self.last_cmd_time if self.received_cmd_count else float("inf")
        timed_out = (
            not self.args.disable_command_timeout
            and (self.received_cmd_count == 0 or age > self.args.command_timeout)
        )
        cmd = [0.0, 0.0, 0.0] if timed_out else self.latest_cmd
        payload = {
            "cmd": cmd,
            "stamp": time.time(),
            "cmd_age": age,
            "timed_out": timed_out,
            "received_cmd_count": self.received_cmd_count,
        }
        self.cmd_socket.sendto(json.dumps(payload).encode("utf-8"), self.cmd_target)
        self.sent_cmd_count += 1

    def _drain_odom(self) -> None:
        while True:
            try:
                data, _ = self.odom_socket.recvfrom(65535)
            except BlockingIOError:
                return
            try:
                payload = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as exc:
                self.node.get_logger().warn(f"bad odom UDP JSON: {exc}")
                continue
            self._publish_odom(payload)

    def _publish_odom(self, payload: dict) -> None:
        sim_time = float(payload.get("sim_time", time.time()))
        pos = payload["pos"]
        quat_wxyz = payload["quat_wxyz"]
        lin_vel_b = payload["lin_vel_b"]
        ang_vel_b = payload["ang_vel_b"]
        stamp = make_stamp(sim_time)

        yaw = yaw_from_quat_wxyz(quat_wxyz)
        if self.args.use_world_odom:
            odom_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
            odom_quat_xyzw = (float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]))
        else:
            if self.odom_origin is None:
                self.odom_origin = (float(pos[0]), float(pos[1]), yaw)
                self.node.get_logger().info(
                    "zeroing odom at Isaac start pose "
                    f"x={self.odom_origin[0]:.3f} y={self.odom_origin[1]:.3f} yaw={self.odom_origin[2]:.3f}"
                )
            origin_x, origin_y, origin_yaw = self.odom_origin
            dx = float(pos[0]) - origin_x
            dy = float(pos[1]) - origin_y
            cos_yaw = math.cos(origin_yaw)
            sin_yaw = math.sin(origin_yaw)
            odom_pos = [
                cos_yaw * dx + sin_yaw * dy,
                -sin_yaw * dx + cos_yaw * dy,
                float(pos[2]),
            ]
            odom_quat_xyzw = quat_xyzw_from_yaw(wrap_to_pi(yaw - origin_yaw))

        clock = Clock()
        clock.clock = stamp
        self.clock_pub.publish(clock)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.args.odom_frame
        odom.child_frame_id = self.args.base_frame
        odom.pose.pose.position.x = odom_pos[0]
        odom.pose.pose.position.y = odom_pos[1]
        odom.pose.pose.position.z = odom_pos[2]
        odom.pose.pose.orientation.x = odom_quat_xyzw[0]
        odom.pose.pose.orientation.y = odom_quat_xyzw[1]
        odom.pose.pose.orientation.z = odom_quat_xyzw[2]
        odom.pose.pose.orientation.w = odom_quat_xyzw[3]
        odom.twist.twist.linear.x = lin_vel_b[0]
        odom.twist.twist.linear.y = lin_vel_b[1]
        odom.twist.twist.linear.z = lin_vel_b[2]
        odom.twist.twist.angular.x = ang_vel_b[0]
        odom.twist.twist.angular.y = ang_vel_b[1]
        odom.twist.twist.angular.z = ang_vel_b[2]
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.args.odom_frame
        tf.child_frame_id = self.args.base_frame
        tf.transform.translation.x = odom_pos[0]
        tf.transform.translation.y = odom_pos[1]
        tf.transform.translation.z = odom_pos[2]
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

        # MoveIt/robot_state_publisher 的 URDF 根是 pelvis，而 Nav2 使用 base_link。
        # 这里把两棵树接起来：map -> odom -> base_link -> pelvis -> torso_link -> right_rubber_hand。
        pelvis_tf = TransformStamped()
        pelvis_tf.header.stamp = stamp
        pelvis_tf.header.frame_id = self.args.base_frame
        pelvis_tf.child_frame_id = self.args.pelvis_frame
        pelvis_tf.transform.translation.x = 0.0
        pelvis_tf.transform.translation.y = 0.0
        pelvis_tf.transform.translation.z = 0.0
        pelvis_tf.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(pelvis_tf)

        joint_names = payload.get("joint_names") or []
        joint_pos = payload.get("joint_pos") or []
        joint_vel = payload.get("joint_vel") or []
        if joint_names and len(joint_pos) == len(joint_names):
            joint_state = JointState()
            joint_state.header.stamp = stamp
            joint_state.name = [str(name) for name in joint_names]
            joint_state.position = [float(value) for value in joint_pos]
            if len(joint_vel) == len(joint_names):
                joint_state.velocity = [float(value) for value in joint_vel]
            self.joint_state_pub.publish(joint_state)

        self.received_odom_count += 1
        if self.received_odom_count <= 5 or self.received_odom_count % 100 == 0:
            self.node.get_logger().info(
                f"published odom count={self.received_odom_count} "
                f"pos={[round(float(v), 3) for v in odom_pos]} "
                f"lin_vel_b={[round(float(v), 3) for v in lin_vel_b]}"
            )


def main() -> None:
    args = parse_args()
    rclpy.init()
    bridge = G1UdpRosBridge(args)
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
