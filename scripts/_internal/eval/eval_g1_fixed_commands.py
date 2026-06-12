import argparse
import csv
import json
import math
import os
import pathlib
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # isort: skip
sys.path.pop(0)

parser = argparse.ArgumentParser(description="Evaluate StableVelocity checkpoints on fixed Nav2 cmd_vel commands.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-StableVelocity")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from importlib.metadata import version
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "rsl_rl"))
import cli_args as rsl_cli_args  # noqa: E402
sys.path.pop(0)

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.rsl_rl_stability import load_runner_checkpoint_compat


COMMAND_SUITE = [
    ("vx020", (0.20, 0.0, 0.00)),
    ("vx035", (0.35, 0.0, 0.00)),
    ("vx050", (0.50, 0.0, 0.00)),
    ("wz030", (0.00, 0.0, 0.30)),
    ("wzneg030", (0.00, 0.0, -0.30)),
    ("arc_left", (0.25, 0.0, 0.30)),
    ("arc_right", (0.25, 0.0, -0.30)),
    ("stop", (0.00, 0.0, 0.00)),
]


def get_obs(vec_env):
    obs = vec_env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs
    return obs


def resolve_checkpoint(task_name: str, checkpoint: str | None):
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if checkpoint:
        return retrieve_file_path(checkpoint), agent_cfg
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint), agent_cfg


def enforce_fixed_command(vec_env, command: torch.Tensor) -> torch.Tensor:
    term = vec_env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = command
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = torch.all(command == 0.0, dim=1)
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False
    if hasattr(term, "heading_target"):
        term.heading_target[:] = 0.0
    term.time_left[:] = 1.0e9
    return vec_env.unwrapped.command_manager.get_command("base_velocity").clone()


def seed_history(vec_env, command: torch.Tensor, history_length: int = 5):
    vec_env.unwrapped.reset()
    vec_env.unwrapped.observation_manager.reset()
    for _ in range(history_length):
        enforce_fixed_command(vec_env, command)
        vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute(update_history=True)
    return get_obs(vec_env)


def quat_to_euler_xyz(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    w, x, y, z = quat_wxyz.tolist()
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def reward_term_value(env, term_name: str) -> float | None:
    try:
        reward_cfg = env.reward_manager.get_term_cfg(term_name)
    except Exception:
        return None
    return float(reward_cfg.func(env, **reward_cfg.params)[0].item())


def run_case(vec_env, policy, name: str, command_tuple: tuple[float, float, float], steps: int) -> dict:
    device = vec_env.unwrapped.device
    command = torch.tensor([command_tuple], dtype=torch.float32, device=device).repeat(vec_env.unwrapped.num_envs, 1)
    obs = seed_history(vec_env, command)
    robot = vec_env.unwrapped.scene["robot"]
    unwrapped = vec_env.unwrapped

    start_pos = robot.data.root_pos_w[0].clone()
    start_roll, start_pitch, start_yaw = quat_to_euler_xyz(robot.data.root_quat_w[0])
    del start_roll, start_pitch

    actual_vx, actual_wz, heights, rolls, pitches, rewards = [], [], [], [], [], []
    feet_slide_vals, action_rate_vals = [], []
    termination_count = 0

    for _ in range(steps):
        enforce_fixed_command(vec_env, command)
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, dones, _ = vec_env.step(actions)

        roll, pitch, _ = quat_to_euler_xyz(robot.data.root_quat_w[0])
        actual_vx.append(float(robot.data.root_lin_vel_b[0, 0].item()))
        actual_wz.append(float(robot.data.root_ang_vel_b[0, 2].item()))
        heights.append(float(robot.data.root_pos_w[0, 2].item()))
        rolls.append(abs(roll))
        pitches.append(abs(pitch))
        rewards.append(float(reward[0].item()))
        feet_slide = reward_term_value(unwrapped, "feet_slide")
        action_rate = reward_term_value(unwrapped, "action_rate")
        if feet_slide is not None:
            feet_slide_vals.append(feet_slide)
        if action_rate is not None:
            action_rate_vals.append(action_rate)
        if bool(dones[0].item()):
            termination_count += 1
            obs = seed_history(vec_env, command)

    end_pos = robot.data.root_pos_w[0].clone()
    _, _, end_yaw = quat_to_euler_xyz(robot.data.root_quat_w[0])
    return {
        "case": name,
        "cmd_vx": command_tuple[0],
        "cmd_vy": command_tuple[1],
        "cmd_wz": command_tuple[2],
        "actual_vx_mean": float(np.mean(actual_vx)),
        "actual_vx_std": float(np.std(actual_vx)),
        "actual_wz_mean": float(np.mean(actual_wz)),
        "actual_wz_std": float(np.std(actual_wz)),
        "xy_displacement": float(torch.norm(end_pos[:2] - start_pos[:2]).item()),
        "yaw_displacement": float(wrap_to_pi(end_yaw - start_yaw)),
        "root_height_mean": float(np.mean(heights)),
        "root_height_std": float(np.std(heights)),
        "roll_max": float(np.max(rolls)),
        "pitch_max": float(np.max(pitches)),
        "feet_slide_mean": float(np.mean(feet_slide_vals)) if feet_slide_vals else None,
        "termination_ratio": float(termination_count / steps),
        "action_rate_mean": float(np.mean(action_rate_vals)) if action_rate_vals else None,
        "episode_reward_mean": float(np.mean(rewards)),
    }


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False

    checkpoint_path, agent_cfg = resolve_checkpoint(args_cli.task, args_cli.checkpoint)
    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_runner_checkpoint_compat(runner, checkpoint_path)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    checkpoint = pathlib.Path(checkpoint_path)
    output_dir = checkpoint.parent / "fixed_command_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{checkpoint.stem}_eval.json"
    csv_path = output_dir / f"{checkpoint.stem}_eval.csv"

    results = []
    for name, command in COMMAND_SUITE:
        metrics = run_case(vec_env, policy, name, command, args_cli.steps)
        results.append(metrics)
        print(f"[fixed-eval] {name}: {json.dumps(metrics, ensure_ascii=False)}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("[fixed-eval] checkpoint:", checkpoint_path)
    print("[fixed-eval] json:", json_path)
    print("[fixed-eval] csv:", csv_path)
    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
