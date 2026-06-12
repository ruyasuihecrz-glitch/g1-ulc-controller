#!/usr/bin/env python3
"""Run arm-only balance experiments for the G1 policy bridge.

The experiment keeps cmd_vel at zero, sends right-arm joint targets through the
same UDP path used by the button demo, and records base drift from /odom.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import socket
import time
from dataclasses import dataclass, field

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

STOW = {
    "right_shoulder_pitch_joint": 0.235,
    "right_shoulder_roll_joint": -0.277,
    "right_shoulder_yaw_joint": -0.004,
    "right_elbow_joint": 0.958,
    "right_wrist_roll_joint": -0.159,
    "right_wrist_pitch_joint": -0.007,
    "right_wrist_yaw_joint": 0.010,
}

# This is close to the successful first-button IK solution. It mainly moves
# shoulder pitch / elbow and avoids large shoulder-yaw or wrist-roll excursions.
BUTTON_LIKE_SAFE = {
    "right_shoulder_pitch_joint": -0.727,
    "right_shoulder_roll_joint": -0.310,
    "right_shoulder_yaw_joint": 0.019,
    "right_elbow_joint": 0.613,
    "right_wrist_roll_joint": -0.076,
    "right_wrist_pitch_joint": 0.281,
    "right_wrist_yaw_joint": -0.086,
}

# This resembles the bad IK branch we saw during debugging: mathematically
# reachable, but dynamically ugly for the locomotion policy.
BAD_BRANCH_AGGRESSIVE = {
    "right_shoulder_pitch_joint": -1.432,
    "right_shoulder_roll_joint": -0.106,
    "right_shoulder_yaw_joint": 2.618,
    "right_elbow_joint": 1.280,
    "right_wrist_roll_joint": 1.458,
    "right_wrist_pitch_joint": 0.155,
    "right_wrist_yaw_joint": -0.147,
}


def yaw_from_quat_xyzw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Sample:
    wall_time: float
    x: float
    y: float
    z: float
    yaw: float
    lin_vx: float
    lin_vy: float
    lin_vz: float
    ang_vx: float
    ang_vy: float
    ang_vz: float
    max_joint_error: float


@dataclass
class CaseResult:
    name: str
    samples: list[Sample] = field(default_factory=list)

    def summarize(self) -> dict[str, float | str | int]:
        if not self.samples:
            return {"case": self.name, "samples": 0}
        first = self.samples[0]
        max_xy = 0.0
        max_yaw = 0.0
        max_lin_xy = 0.0
        max_ang_xy = 0.0
        max_ang_z = 0.0
        min_z = first.z
        max_joint_error = 0.0
        for sample in self.samples:
            max_xy = max(max_xy, math.hypot(sample.x - first.x, sample.y - first.y))
            max_yaw = max(max_yaw, abs(wrap_to_pi(sample.yaw - first.yaw)))
            max_lin_xy = max(max_lin_xy, math.hypot(sample.lin_vx, sample.lin_vy))
            max_ang_xy = max(max_ang_xy, math.hypot(sample.ang_vx, sample.ang_vy))
            max_ang_z = max(max_ang_z, abs(sample.ang_vz))
            min_z = min(min_z, sample.z)
            max_joint_error = max(max_joint_error, sample.max_joint_error)
        last = self.samples[-1]
        return {
            "case": self.name,
            "samples": len(self.samples),
            "duration_s": round(last.wall_time - first.wall_time, 3),
            "final_xy_drift_m": round(math.hypot(last.x - first.x, last.y - first.y), 4),
            "max_xy_drift_m": round(max_xy, 4),
            "max_yaw_drift_rad": round(max_yaw, 4),
            "min_base_z_m": round(min_z, 4),
            "max_lin_xy_mps": round(max_lin_xy, 4),
            "max_ang_xy_radps": round(max_ang_xy, 4),
            "max_ang_z_radps": round(max_ang_z, 4),
            "max_joint_error_rad": round(max_joint_error, 4),
        }


class ArmBalanceExperiment(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("g1_arm_balance_experiment")
        self.args = args
        self.latest_odom: Odometry | None = None
        self.latest_joint_state: JointState | None = None
        self.create_subscription(Odometry, args.odom_topic, self._odom_cb, 20)
        self.create_subscription(JointState, args.joint_states_topic, self._joint_state_cb, 20)
        self.cmd_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.arm_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.arm_target = (args.arm_udp_host, args.arm_udp_port)
        self.active_target: dict[str, float] | None = None

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _joint_state_cb(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def wait_for_inputs(self) -> None:
        deadline = time.monotonic() + self.args.wait_timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_odom is not None and self.latest_joint_state is not None:
                return
        raise SystemExit("Timed out waiting for /odom and /joint_states")

    def publish_zero_cmd(self) -> None:
        self.cmd_pub.publish(Twist())

    def send_arm_target(self, target: dict[str, float] | None) -> None:
        if target is None:
            return
        payload = {
            "stamp": time.time(),
            "arm_joint_positions": target,
            "hold_timeout": 0.5,
            "source": "arm_balance_experiment",
        }
        self.arm_sock.sendto(json.dumps(payload).encode("utf-8"), self.arm_target)

    def joint_error(self, target: dict[str, float] | None) -> float:
        if target is None or self.latest_joint_state is None:
            return 0.0
        current = {
            name: float(pos)
            for name, pos in zip(self.latest_joint_state.name, self.latest_joint_state.position, strict=False)
        }
        errors = [abs(float(value) - current[name]) for name, value in target.items() if name in current]
        if len(errors) != len(target):
            return float("inf")
        return max(errors)

    def current_sample(self, target: dict[str, float] | None) -> Sample:
        assert self.latest_odom is not None
        pose = self.latest_odom.pose.pose
        twist = self.latest_odom.twist.twist
        return Sample(
            wall_time=time.monotonic(),
            x=float(pose.position.x),
            y=float(pose.position.y),
            z=float(pose.position.z),
            yaw=yaw_from_quat_xyzw(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
            lin_vx=float(twist.linear.x),
            lin_vy=float(twist.linear.y),
            lin_vz=float(twist.linear.z),
            ang_vx=float(twist.angular.x),
            ang_vy=float(twist.angular.y),
            ang_vz=float(twist.angular.z),
            max_joint_error=self.joint_error(target),
        )

    def run_case(self, name: str, target: dict[str, float] | None, duration_s: float) -> CaseResult:
        result = CaseResult(name=name)
        self.get_logger().info(f"CASE {name} start duration={duration_s:.1f}s")
        end_time = time.monotonic() + duration_s
        period = 1.0 / self.args.rate
        while time.monotonic() < end_time:
            self.publish_zero_cmd()
            self.send_arm_target(target)
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.latest_odom is not None:
                result.samples.append(self.current_sample(target))
            time.sleep(period)
        summary = result.summarize()
        self.get_logger().info(f"CASE {name} summary {summary}")
        return result


def write_outputs(results: list[CaseResult], output_dir: pathlib.Path, tag: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"arm_balance_summary_{tag}.csv"
    samples_path = output_dir / f"arm_balance_samples_{tag}.csv"

    summaries = [result.summarize() for result in results]
    keys = list(summaries[0].keys()) if summaries else []
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)

    with samples_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["case", *Sample.__dataclass_fields__.keys()]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for sample in result.samples:
                row = {"case": result.name, **sample.__dict__}
                writer.writerow(row)

    print(summary_path)
    print(samples_path)
    for summary in summaries:
        print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G1 right-arm balance perturbation experiments.")
    parser.add_argument("--cmd_vel_topic", default="/cmd_vel")
    parser.add_argument("--odom_topic", default="/odom")
    parser.add_argument("--joint_states_topic", default="/joint_states")
    parser.add_argument("--arm_udp_host", default="127.0.0.1")
    parser.add_argument("--arm_udp_port", type=int, default=15003)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--wait_timeout", type=float, default=30.0)
    parser.add_argument("--baseline_s", type=float, default=5.0)
    parser.add_argument("--case_s", type=float, default=7.0)
    parser.add_argument("--settle_s", type=float, default=4.0)
    parser.add_argument("--tag", default=str(int(time.time())))
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("logs/arm_balance"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = ArmBalanceExperiment(args)
    try:
        node.wait_for_inputs()
        results = [
            node.run_case("baseline_zero_cmd_no_arm_override", None, args.baseline_s),
            node.run_case("settle_stow_before_safe", STOW, args.settle_s),
            node.run_case("button_like_safe_arm_target", BUTTON_LIKE_SAFE, args.case_s),
            node.run_case("settle_stow_before_aggressive", STOW, args.settle_s),
            node.run_case("bad_branch_aggressive_arm_target", BAD_BRANCH_AGGRESSIVE, args.case_s),
            node.run_case("final_stow", STOW, args.settle_s),
        ]
        write_outputs(results, args.output_dir, args.tag)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
