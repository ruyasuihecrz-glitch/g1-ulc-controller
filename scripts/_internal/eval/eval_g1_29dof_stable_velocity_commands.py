import argparse
import json
import os
import pathlib
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # isort: skip
sys.path.pop(0)


parser = argparse.ArgumentParser(description="Run the formal StableVelocity command evaluation suite.")
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


def clamp_command(command: torch.Tensor, env_cfg) -> torch.Tensor:
    ranges = env_cfg.commands.base_velocity.limit_ranges
    command[:, 0] = command[:, 0].clamp(*ranges.lin_vel_x)
    command[:, 1] = command[:, 1].clamp(*ranges.lin_vel_y)
    command[:, 2] = command[:, 2].clamp(*ranges.ang_vel_z)
    return command


def enforce_fixed_command(env, command: torch.Tensor) -> torch.Tensor:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = command
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[:] = False
    if hasattr(term, "heading_target"):
        term.heading_target[:] = 0.0
    term.time_left[:] = 1.0e9
    return env.unwrapped.command_manager.get_command("base_velocity").clone()


def quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def resolve_checkpoint(task_name: str, checkpoint: str | None):
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if checkpoint:
        return retrieve_file_path(checkpoint), agent_cfg
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint), agent_cfg


def get_obs(vec_env):
    obs = vec_env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = obs
    return obs


def refresh_obs_with_fixed_command(vec_env, env_cfg, command: torch.Tensor):
    fixed_command = clamp_command(command.clone(), env_cfg)
    applied_command = enforce_fixed_command(vec_env, fixed_command)
    obs = get_obs(vec_env)
    return obs, applied_command


def seed_obs_history_with_fixed_command(vec_env, env_cfg, command: torch.Tensor, history_length: int = 5):
    fixed_command = clamp_command(command.clone(), env_cfg)
    vec_env.unwrapped.observation_manager.reset()
    for _ in range(history_length):
        fixed_command = enforce_fixed_command(vec_env, fixed_command)
        vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute(update_history=True)
    return get_obs(vec_env), fixed_command


def make_suite(device: torch.device):
    return [
        ("stand", torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)),
        ("slow_walk", torch.tensor([[0.10, 0.0, 0.0]], dtype=torch.float32, device=device)),
        ("medium_walk", torch.tensor([[0.20, 0.0, 0.0]], dtype=torch.float32, device=device)),
        ("fast_patrol", torch.tensor([[0.35, 0.0, 0.0]], dtype=torch.float32, device=device)),
        ("turn_left", torch.tensor([[0.0, 0.0, 0.15]], dtype=torch.float32, device=device)),
        ("turn_right", torch.tensor([[0.0, 0.0, -0.15]], dtype=torch.float32, device=device)),
        ("side_step", torch.tensor([[0.0, 0.10, 0.0]], dtype=torch.float32, device=device)),
    ]


def pass_fail(metrics: dict) -> bool:
    return (
        metrics["termination_count"] == 0
        and metrics["root_height_mean"] > 0.65
        and abs(metrics["yaw_drift"]) < 0.5
        and abs(metrics["lateral_drift"]) < 0.5
    )


def reward_term_value(env, term_name: str) -> float:
    reward_cfg = env.reward_manager.get_term_cfg(term_name)
    return reward_cfg.func(env, **reward_cfg.params)[0].item()


def run_case(vec_env, policy, env_cfg, name: str, command: torch.Tensor, steps: int) -> dict:
    obs, fixed_command = seed_obs_history_with_fixed_command(vec_env, env_cfg, command)
    robot = vec_env.unwrapped.scene["robot"]
    unwrapped = vec_env.unwrapped

    start_pos = robot.data.root_pos_w[0].clone()
    start_yaw = quat_to_yaw(robot.data.root_quat_w[0:1])[0].item()
    heights, yaws, base_ang_vels, rewards = [], [], [], []
    feet_slide_vals, arm_dev_vals, action_rate_vals = [], [], []
    termination_count = 0

    for _ in range(steps):
        fixed_command = enforce_fixed_command(vec_env, fixed_command)
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, dones, _ = vec_env.step(actions)

        heights.append(robot.data.root_pos_w[0, 2].item())
        yaws.append(quat_to_yaw(robot.data.root_quat_w[0:1])[0].item())
        base_ang_vels.append(robot.data.root_ang_vel_b[0].norm().item())
        rewards.append(reward[0].item())
        feet_slide_vals.append(reward_term_value(unwrapped, "feet_slide"))
        arm_dev_vals.append(reward_term_value(unwrapped, "joint_deviation_arms"))
        action_rate_vals.append(reward_term_value(unwrapped, "action_rate"))
        if dones[0].item():
            termination_count += 1
            obs, fixed_command = seed_obs_history_with_fixed_command(vec_env, env_cfg, fixed_command)

    end_pos = robot.data.root_pos_w[0].clone()
    metrics = {
        "command": fixed_command[0].detach().cpu().numpy().tolist(),
        "root_height_mean": float(np.mean(heights)),
        "root_height_std": float(np.std(heights)),
        "xy_displacement": float(torch.norm(end_pos[:2] - start_pos[:2]).item()),
        "yaw_drift": float(yaws[-1] - start_yaw),
        "lateral_drift": float((end_pos[1] - start_pos[1]).item()),
        "termination_count": int(termination_count),
        "mean_reward": float(np.mean(rewards)),
        "action_rate_mean": float(np.mean(action_rate_vals)),
        "base_ang_vel_mean": float(np.mean(base_ang_vels)),
        "feet_slide_reward_mean": float(np.mean(feet_slide_vals)),
        "arm_deviation_reward_mean": float(np.mean(arm_dev_vals)),
    }
    metrics["pass"] = pass_fail(metrics)
    return metrics


def run_stop_transition(vec_env, policy, env_cfg, steps: int) -> dict:
    robot = vec_env.unwrapped.scene["robot"]
    unwrapped = vec_env.unwrapped
    obs = get_obs(vec_env)
    start_pos = robot.data.root_pos_w[0].clone()
    start_yaw = quat_to_yaw(robot.data.root_quat_w[0:1])[0].item()
    heights, rewards = [], []
    yaws = []
    base_ang_vels, feet_slide_vals, arm_dev_vals, action_rate_vals = [], [], [], []
    phase_steps = steps // 2
    termination_count = 0

    for raw_cmd in (
        torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float32, device=vec_env.unwrapped.device),
        torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=vec_env.unwrapped.device),
    ):
        obs, fixed_command = seed_obs_history_with_fixed_command(vec_env, env_cfg, raw_cmd)
        for _ in range(phase_steps):
            fixed_command = enforce_fixed_command(vec_env, fixed_command)
            with torch.inference_mode():
                actions = policy(obs)
                obs, reward, dones, _ = vec_env.step(actions)
            heights.append(robot.data.root_pos_w[0, 2].item())
            rewards.append(reward[0].item())
            yaws.append(quat_to_yaw(robot.data.root_quat_w[0:1])[0].item())
            base_ang_vels.append(robot.data.root_ang_vel_b[0].norm().item())
            feet_slide_vals.append(reward_term_value(unwrapped, "feet_slide"))
            arm_dev_vals.append(reward_term_value(unwrapped, "joint_deviation_arms"))
            action_rate_vals.append(reward_term_value(unwrapped, "action_rate"))
            if dones[0].item():
                termination_count += 1
                obs, fixed_command = seed_obs_history_with_fixed_command(vec_env, env_cfg, fixed_command)

    end_pos = robot.data.root_pos_w[0].clone()
    metrics = {
        "command": [[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "root_height_mean": float(np.mean(heights)),
        "root_height_std": float(np.std(heights)),
        "xy_displacement": float(torch.norm(end_pos[:2] - start_pos[:2]).item()),
        "yaw_drift": float(yaws[-1] - start_yaw),
        "lateral_drift": float((end_pos[1] - start_pos[1]).item()),
        "termination_count": int(termination_count),
        "mean_reward": float(np.mean(rewards)),
        "action_rate_mean": float(np.mean(action_rate_vals)),
        "base_ang_vel_mean": float(np.mean(base_ang_vels)),
        "feet_slide_reward_mean": float(np.mean(feet_slide_vals)),
        "arm_deviation_reward_mean": float(np.mean(arm_dev_vals)),
    }
    metrics["pass"] = pass_fail(metrics)
    return metrics


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
    env = gym.make(args_cli.task, cfg=env_cfg)

    checkpoint_path, agent_cfg = resolve_checkpoint(args_cli.task, args_cli.checkpoint)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_runner_checkpoint_compat(runner, checkpoint_path)
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    results_dir = pathlib.Path("results/g1_29dof_stable_velocity_eval") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    suite_results = {}

    for name, command in make_suite(vec_env.unwrapped.device):
        start_time = time.time()
        metrics = run_case(vec_env, policy, env_cfg, name, command, args_cli.steps)
        suite_results[name] = metrics
        print(f"[eval] {name}: {json.dumps(metrics, ensure_ascii=False)} elapsed={time.time() - start_time:.2f}s")

    suite_results["stop_transition"] = run_stop_transition(vec_env, policy, env_cfg, args_cli.steps)
    print(f"[eval] stop_transition: {json.dumps(suite_results['stop_transition'], ensure_ascii=False)}")

    metrics_path = results_dir / "metrics.json"
    summary_path = results_dir / "summary.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2, ensure_ascii=False)
    with summary_path.open("w", encoding="utf-8") as f:
        for name, metrics in suite_results.items():
            f.write(f"{name}: pass={metrics['pass']} mean_reward={metrics['mean_reward']:.5f} terminations={metrics['termination_count']}\n")

    print("[eval] results_dir:", results_dir)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
