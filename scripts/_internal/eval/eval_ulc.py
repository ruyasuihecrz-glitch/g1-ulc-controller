#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate Unitree G1 ULC policy.")
parser.add_argument("--task", default="Unitree-G1-29dof-ULC")
parser.add_argument("--scenario", choices=["whole", "edge", "wrist_loaded_2kg", "command_mutation"], default="whole")
parser.add_argument("--run_name", default="debug")
parser.add_argument("--num_steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _mean_std(x: torch.Tensor) -> dict[str, float]:
    return {"mean": float(torch.mean(x).item()), "std": float(torch.std(x).item())}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=None)
    if args_cli.scenario == "edge":
        env_cfg.commands.ulc_command.mode = "edge"
    elif args_cli.scenario == "wrist_loaded_2kg":
        env_cfg.events.add_wrist_mass.params["mass_distribution_params"] = (2.0, 2.0)
    elif args_cli.scenario == "command_mutation":
        env_cfg.commands.ulc_command.resampling_time_range = (0.25, 0.5)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    metric_buf = {name: [] for name in ["E_v", "E_w", "E_h", "E_y", "E_p", "E_r", "E_a"]}

    for _ in range(args_cli.num_steps):
        action = torch.zeros(env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device).repeat(
            env.unwrapped.num_envs, 1
        )
        obs, rew, terminated, truncated, info = env.step(action)
        robot = env.unwrapped.scene["robot"]
        cmd = env.unwrapped.command_manager.get_command("ulc_command")
        metric_buf["E_v"].append(torch.linalg.norm(robot.data.root_lin_vel_b[:, :2] - cmd[:, :2], dim=1))
        metric_buf["E_w"].append(torch.abs(robot.data.root_ang_vel_b[:, 2] - cmd[:, 2]))
        metric_buf["E_h"].append(torch.abs(robot.data.root_pos_w[:, 2] - cmd[:, 3]))
        metric_buf["E_y"].append(1.0 - env.unwrapped.reward_manager._step_reward[:, env.unwrapped.reward_manager._term_names.index("torso_yaw")])
        metric_buf["E_r"].append(1.0 - env.unwrapped.reward_manager._step_reward[:, env.unwrapped.reward_manager._term_names.index("torso_roll")])
        metric_buf["E_p"].append(1.0 - env.unwrapped.reward_manager._step_reward[:, env.unwrapped.reward_manager._term_names.index("torso_pitch")])
        metric_buf["E_a"].append(1.0 - env.unwrapped.reward_manager._step_reward[:, env.unwrapped.reward_manager._term_names.index("arm_tracking")])

    out_dir = Path("logs") / "ulc_eval" / args_cli.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {name: _mean_std(torch.cat(values)) for name, values in metric_buf.items()}
    json_path = out_dir / f"{args_cli.scenario}.json"
    csv_path = out_dir / f"{args_cli.scenario}.csv"
    json_path.write_text(json.dumps(summary, indent=2))
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for name, stats in summary.items():
            writer.writerow([name, stats["mean"], stats["std"]])
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
