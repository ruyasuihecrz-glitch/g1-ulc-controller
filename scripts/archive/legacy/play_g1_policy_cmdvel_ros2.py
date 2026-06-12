"""Run a trained G1 velocity policy in Isaac Lab and bridge it to ROS 2 cmd_vel.

This script keeps the official Isaac Lab / RSL-RL play pipeline intact, but replaces
the random command generator with commands received from ROS 2:

    /cmd_vel -> command_manager["base_velocity"] -> policy -> 29-DOF action

It also publishes the minimum signals needed by a Nav2 demo: /clock, /odom, and TF
odom->base_link. Mapping and obstacle avoidance should remain in the standard ROS 2
Nav2/slam_toolbox stack.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from importlib.metadata import version

from isaaclab.app import AppLauncher

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # isort: skip
sys.path.pop(0)

parser = argparse.ArgumentParser(description="Play a G1 velocity policy with ROS 2 /cmd_vel input.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-Velocity")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=True)
parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel")
parser.add_argument("--odom_topic", type=str, default="/odom")
parser.add_argument("--odom_frame", type=str, default="odom")
parser.add_argument("--base_frame", type=str, default="base_link")
parser.add_argument("--command_timeout", type=float, default=0.5)
parser.add_argument(
    "--disable_command_timeout",
    action="store_true",
    default=False,
    help="Keep the last received /cmd_vel indefinitely. Useful for demos/manual tests; Nav2 should normally publish continuously.",
)
parser.add_argument("--min_vx", type=float, default=-0.2)
parser.add_argument("--max_vx", type=float, default=0.5)
parser.add_argument("--max_abs_vy", type=float, default=0.10)
parser.add_argument("--max_abs_wz", type=float, default=0.15)
parser.add_argument("--command_print_interval", type=int, default=100)
parser.add_argument("--no_auto_camera", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.rsl_rl_stability import load_runner_checkpoint_compat

try:
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from rosgraph_msgs.msg import Clock
    from tf2_ros import TransformBroadcaster
except ModuleNotFoundError as exc:
    raise SystemExit(
        "ROS 2 Python packages are not available in this Isaac Python environment. "
        "Source your ROS 2 workspace before launching, for example: "
        "source /opt/ros/jazzy/setup.bash"
    ) from exc

try:
    from isaacsim.core.utils.viewports import set_camera_view
except ModuleNotFoundError:
    set_camera_view = None


def get_obs(vec_env):
    obs = vec_env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs
    return obs


def resolve_checkpoint(task_name: str, agent_cfg: RslRlOnPolicyRunnerCfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def clamp_to_policy_range(command: torch.Tensor, env_cfg) -> torch.Tensor:
    ranges = env_cfg.commands.base_velocity.limit_ranges
    command[:, 0] = command[:, 0].clamp(ranges.lin_vel_x[0], ranges.lin_vel_x[1])
    command[:, 1] = command[:, 1].clamp(ranges.lin_vel_y[0], ranges.lin_vel_y[1])
    command[:, 2] = command[:, 2].clamp(ranges.ang_vel_z[0], ranges.ang_vel_z[1])
    return command


def enforce_command(vec_env, command: torch.Tensor) -> torch.Tensor:
    term = vec_env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = command
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = torch.all(torch.isclose(command, torch.zeros_like(command)), dim=1)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False
    if hasattr(term, "heading_target"):
        term.heading_target[:] = 0.0
    if hasattr(term, "time_left"):
        term.time_left[:] = 1.0e9
    return vec_env.unwrapped.command_manager.get_command("base_velocity").clone()


def set_camera_to_robot(vec_env) -> None:
    if args_cli.headless or args_cli.no_auto_camera or set_camera_view is None:
        return
    robot = vec_env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0].detach().cpu()
    target = [float(root_pos[0]), float(root_pos[1]), float(root_pos[2] + 0.7)]
    eye = [target[0] + 3.0, target[1] - 3.0, target[2] + 1.8]
    set_camera_view(eye, target)
    print(f"[ros2-bridge] camera eye={eye} target={target}")


def make_stamp(sim_time: float) -> Time:
    stamp = Time()
    stamp.sec = int(sim_time)
    stamp.nanosec = int((sim_time - stamp.sec) * 1.0e9)
    return stamp


class G1Nav2Bridge:
    def __init__(self):
        rclpy.init(args=None)
        self.node = rclpy.create_node("g1_policy_cmdvel_bridge")
        self.cmd_sub = self.node.create_subscription(Twist, args_cli.cmd_vel_topic, self._cmd_callback, 10)
        self.odom_pub = self.node.create_publisher(Odometry, args_cli.odom_topic, 10)
        self.clock_pub = self.node.create_publisher(Clock, "/clock", 10)
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.latest_cmd = torch.zeros(3, dtype=torch.float32)
        self.last_cmd_wall_time = float("-inf")
        self.received_cmd_count = 0
        self.command_timed_out = True

    def shutdown(self) -> None:
        self.node.destroy_node()
        rclpy.shutdown()

    def spin_once(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def command_tensor(self, device: torch.device, env_cfg) -> torch.Tensor:
        age = self.command_age()
        self.command_timed_out = (
            not args_cli.disable_command_timeout
            and (self.received_cmd_count == 0 or age > args_cli.command_timeout)
        )
        if self.command_timed_out:
            cmd = torch.zeros(3, dtype=torch.float32)
        else:
            cmd = self.latest_cmd.clone()
        cmd[0] = torch.clamp(cmd[0], args_cli.min_vx, args_cli.max_vx)
        cmd[1] = torch.clamp(cmd[1], -args_cli.max_abs_vy, args_cli.max_abs_vy)
        cmd[2] = torch.clamp(cmd[2], -args_cli.max_abs_wz, args_cli.max_abs_wz)
        cmd = cmd.to(device=device).unsqueeze(0)
        return clamp_to_policy_range(cmd, env_cfg)

    def command_age(self) -> float:
        if self.received_cmd_count == 0:
            return float("inf")
        return time.monotonic() - self.last_cmd_wall_time

    def publish_state(self, vec_env, sim_time: float) -> None:
        robot = vec_env.unwrapped.scene["robot"]
        pos = robot.data.root_pos_w[0].detach().cpu().tolist()
        quat_wxyz = robot.data.root_quat_w[0].detach().cpu().tolist()
        lin_vel_b = robot.data.root_lin_vel_b[0].detach().cpu().tolist()
        ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().tolist()
        stamp = make_stamp(sim_time)

        clock = Clock()
        clock.clock = stamp
        self.clock_pub.publish(clock)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = args_cli.odom_frame
        odom.child_frame_id = args_cli.base_frame
        odom.pose.pose.position.x = pos[0]
        odom.pose.pose.position.y = pos[1]
        odom.pose.pose.position.z = pos[2]
        odom.pose.pose.orientation.w = quat_wxyz[0]
        odom.pose.pose.orientation.x = quat_wxyz[1]
        odom.pose.pose.orientation.y = quat_wxyz[2]
        odom.pose.pose.orientation.z = quat_wxyz[3]
        odom.twist.twist.linear.x = lin_vel_b[0]
        odom.twist.twist.linear.y = lin_vel_b[1]
        odom.twist.twist.linear.z = lin_vel_b[2]
        odom.twist.twist.angular.x = ang_vel_b[0]
        odom.twist.twist.angular.y = ang_vel_b[1]
        odom.twist.twist.angular.z = ang_vel_b[2]
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = args_cli.odom_frame
        tf.child_frame_id = args_cli.base_frame
        tf.transform.translation.x = pos[0]
        tf.transform.translation.y = pos[1]
        tf.transform.translation.z = pos[2]
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf)

    def _cmd_callback(self, msg: Twist) -> None:
        self.latest_cmd = torch.tensor(
            [msg.linear.x, msg.linear.y, msg.angular.z],
            dtype=torch.float32,
        )
        self.last_cmd_wall_time = time.monotonic()
        self.received_cmd_count += 1
        if self.received_cmd_count <= 5 or self.received_cmd_count % 50 == 0:
            print(
                "[ros2-bridge] received cmd_vel "
                f"count={self.received_cmd_count} "
                f"raw={[float(self.latest_cmd[0]), float(self.latest_cmd[1]), float(self.latest_cmd[2])]}"
            )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    checkpoint_path = resolve_checkpoint(args_cli.task, agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print(f"[ros2-bridge] loading checkpoint: {checkpoint_path}")
    load_runner_checkpoint_compat(runner, checkpoint_path)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    obs = get_obs(vec_env)
    set_camera_to_robot(vec_env)
    bridge = G1Nav2Bridge()
    dt = vec_env.unwrapped.step_dt
    timestep = 0

    print("[ros2-bridge] ready")
    print(f"[ros2-bridge] subscribe: {args_cli.cmd_vel_topic}")
    print(f"[ros2-bridge] publish: {args_cli.odom_topic}, /tf, /clock")
    print(
        "[ros2-bridge] command clamps: "
        f"vx=[{args_cli.min_vx}, {args_cli.max_vx}], "
        f"|vy|<={args_cli.max_abs_vy}, |wz|<={args_cli.max_abs_wz}"
    )
    print(
        "[ros2-bridge] command timeout: "
        + ("disabled" if args_cli.disable_command_timeout else f"{args_cli.command_timeout}s")
    )

    try:
        while simulation_app.is_running() and rclpy.ok():
            start_time = time.time()
            bridge.spin_once()
            command = bridge.command_tensor(vec_env.unwrapped.device, env_cfg)
            live_command = enforce_command(vec_env, command)
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = vec_env.step(actions)

            sim_time = float(vec_env.unwrapped.common_step_counter * vec_env.unwrapped.step_dt)
            bridge.publish_state(vec_env, sim_time)

            if args_cli.command_print_interval > 0 and timestep % args_cli.command_print_interval == 0:
                root_lin_vel_b = vec_env.unwrapped.scene["robot"].data.root_lin_vel_b[0].tolist()
                root_ang_vel_b = vec_env.unwrapped.scene["robot"].data.root_ang_vel_b[0].tolist()
                age = bridge.command_age()
                age_text = "inf" if age == float("inf") else f"{age:.3f}s"
                print(
                    f"[ros2-bridge] step={timestep} cmd={live_command[0].tolist()} "
                    f"raw_cmd={bridge.latest_cmd.tolist()} "
                    f"cmd_age={age_text} timed_out={bridge.command_timed_out} "
                    f"received={bridge.received_cmd_count} "
                    f"root_lin_vel_b={root_lin_vel_b} root_ang_vel_b={root_ang_vel_b}"
                )
            timestep += 1

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        bridge.shutdown()
        vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
