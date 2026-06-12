import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        "Unitree-G1-29dof-Velocity",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )

    env = gym.make("Unitree-G1-29dof-Velocity", cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]

    print("\n=== Robot info ===")
    print("num_joints:", robot.num_joints)
    print("num_bodies:", robot.num_bodies)

    print("\n=== joint_names ===")
    for i, name in enumerate(robot.joint_names):
        print(f"{i:02d}: {name}")

    print("\n=== default joint_pos env0 ===")
    jp = robot.data.default_joint_pos[0].detach().cpu().numpy()
    for i, (name, q) in enumerate(zip(robot.joint_names, jp)):
        print(f"{i:02d}: {name:35s} {q:+.6f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
