from isaaclab.app import AppLauncher


parser_description = "Inspect the Unitree G1 29DOF StableVelocity environment."

import argparse

parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-StableVelocity")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.rsl_rl_stability import finite_guard_enabled, std_positive_guard_enabled


def main():
    task_spec = gym.spec(args_cli.task)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    train_env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    train_num_envs = train_env_cfg.scene.num_envs
    samples_per_iteration = train_num_envs * agent_cfg.num_steps_per_env
    minibatch_size = samples_per_iteration // agent_cfg.algorithm.num_mini_batches
    print("task_registered:", args_cli.task in gym.registry)
    print("task_name:", args_cli.task)
    print("rsl_rl_cfg_entry_point:", task_spec.kwargs.get("rsl_rl_cfg_entry_point"))
    print("training_default_num_envs:", train_num_envs)
    print("inspect_num_envs:", env_cfg.scene.num_envs)
    print("total_samples_per_iteration:", samples_per_iteration)
    print("minibatch_size:", minibatch_size)
    print("save_interval:", getattr(agent_cfg, "save_interval", None))
    print("learning_rate:", agent_cfg.algorithm.learning_rate)
    print("value_loss_coef:", agent_cfg.algorithm.value_loss_coef)
    print("use_clipped_value_loss:", agent_cfg.algorithm.use_clipped_value_loss)
    print("clip_param:", agent_cfg.algorithm.clip_param)
    print("max_grad_norm:", agent_cfg.algorithm.max_grad_norm)
    print("init_noise_std:", agent_cfg.policy.init_noise_std)
    print("noise_std_type:", getattr(agent_cfg.policy, "noise_std_type", None))
    print("desired_kl:", getattr(agent_cfg.algorithm, "desired_kl", None))
    print("schedule:", getattr(agent_cfg.algorithm, "schedule", None))
    print("empirical_normalization:", getattr(agent_cfg, "empirical_normalization", None))
    print("num_steps_per_env:", getattr(agent_cfg, "num_steps_per_env", None))
    print("num_learning_epochs:", getattr(agent_cfg.algorithm, "num_learning_epochs", None))
    print("num_mini_batches:", getattr(agent_cfg.algorithm, "num_mini_batches", None))
    print("gamma:", getattr(agent_cfg.algorithm, "gamma", None))
    print("lam:", getattr(agent_cfg.algorithm, "lam", None))
    print("std_positive_guard:", std_positive_guard_enabled(agent_cfg.policy))
    print("finite_guard:", finite_guard_enabled(args_cli.task))
    print("policy_enable_corruption:", env_cfg.observations.policy.enable_corruption)
    print("critic_enable_corruption:", getattr(env_cfg.observations.critic, "enable_corruption", None))
    print("terrain_flat_only:", list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys()))
    print("terrain_curriculum_enabled:", getattr(env_cfg.scene.terrain.terrain_generator, "curriculum", False))
    print("terrain_num_rows:", env_cfg.scene.terrain.terrain_generator.num_rows)
    print("terrain_num_cols:", env_cfg.scene.terrain.terrain_generator.num_cols)
    print("terrain_size:", env_cfg.scene.terrain.terrain_generator.size)
    print("terrain_difficulty_range:", env_cfg.scene.terrain.terrain_generator.difficulty_range)
    print("terrain_max_init_level:", env_cfg.scene.terrain.max_init_terrain_level)
    print("command_ranges:", env_cfg.commands.base_velocity.ranges)
    print("command_limit_ranges:", env_cfg.commands.base_velocity.limit_ranges)
    print("command_class:", getattr(env_cfg.commands.base_velocity, "class_type", None))
    print("rel_standing_envs:", env_cfg.commands.base_velocity.rel_standing_envs)
    print("rel_straight_envs:", getattr(env_cfg.commands.base_velocity, "rel_straight_envs", None))
    print("rel_turn_envs:", getattr(env_cfg.commands.base_velocity, "rel_turn_envs", None))
    print("physics_material_randomization:", env_cfg.events.physics_material.params)
    print("base_mass_randomization:", env_cfg.events.add_base_mass.params)
    print("push_robot_enabled:", env_cfg.events.push_robot is not None)
    if env_cfg.events.push_robot is not None:
        print("push_robot_interval_range_s:", env_cfg.events.push_robot.interval_range_s)
        print("push_robot_velocity_range:", env_cfg.events.push_robot.params["velocity_range"])
    print("reset_base_pose_range:", env_cfg.events.reset_base.params["pose_range"])
    print("reset_yaw_range:", env_cfg.events.reset_base.params["pose_range"]["yaw"])
    print("reset_joint_velocity_range:", env_cfg.events.reset_robot_joints.params["velocity_range"])
    print("bad_orientation_limit_angle:", env_cfg.terminations.bad_orientation.params["limit_angle"])
    print("curriculum_terms:", [name for name, value in env_cfg.curriculum.__dict__.items() if value is not None])
    print("gait_phase_policy_enabled:", hasattr(env_cfg.observations.policy, "gait_phase"))
    print("gait_phase_critic_enabled:", hasattr(env_cfg.observations.critic, "gait_phase"))
    print("arm_deviation_weight:", env_cfg.rewards.joint_deviation_arms.weight)
    print("waist_deviation_weight:", env_cfg.rewards.joint_deviation_waists.weight)
    print("lateral_drift_weight:", env_cfg.rewards.lateral_drift_when_no_vy.weight)
    print("yaw_drift_weight:", env_cfg.rewards.yaw_drift_when_no_wz.weight)
    print("forward_motion_when_commanded_weight:", getattr(env_cfg.rewards, "forward_motion_when_commanded", None).weight)
    print("turn_when_commanded_weight:", getattr(env_cfg.rewards, "turn_when_commanded", None).weight)
    print("feet_clearance_func:", env_cfg.rewards.feet_clearance.func.__name__)
    print("arm_joint_vel_reward_exists:", hasattr(env_cfg.rewards, "arm_joint_vel"))
    print("waist_joint_vel_reward_exists:", hasattr(env_cfg.rewards, "waist_joint_vel"))
    print("arm_action_rate_reward_exists:", hasattr(env_cfg.rewards, "arm_action_rate"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    policy_obs = obs["policy"]
    print("policy_obs_shape:", list(policy_obs.shape))
    print("critic_obs_shape:", list(obs["critic"].shape))
    print("action_shape:", list(env.unwrapped.action_manager.action.shape))
    print("reward_terms:")
    reward_cfg_items = env_cfg.rewards.__dict__.items()
    reward_count = 0
    for name, cfg in reward_cfg_items:
        if hasattr(cfg, "weight"):
            print(f"  {name}: {cfg.weight}")
            reward_count += 1
    print("reward_terms_count:", reward_count)

    missing_formal_items = []
    if not hasattr(env_cfg.rewards, "arm_joint_vel"):
        missing_formal_items.append("arm_joint_vel")
    if not hasattr(env_cfg.rewards, "waist_joint_vel"):
        missing_formal_items.append("waist_joint_vel")
    if not hasattr(env_cfg.rewards, "arm_action_rate"):
        missing_formal_items.append("arm_action_rate")
    if env_cfg.events.push_robot is None:
        missing_formal_items.append("push_robot")
    if not hasattr(env_cfg.rewards, "forward_motion_when_commanded"):
        missing_formal_items.append("forward_motion_when_commanded")
    if not hasattr(env_cfg.rewards, "turn_when_commanded"):
        missing_formal_items.append("turn_when_commanded")
    if env_cfg.rewards.feet_clearance.func.__name__ != "gated_foot_clearance_reward":
        missing_formal_items.append("gated_foot_clearance_reward")
    if missing_formal_items:
        for item in missing_formal_items:
            print(f"[WARN] missing formal item: {item}")
    else:
        print("[OK] all formal items present")

    if not agent_cfg.algorithm.use_clipped_value_loss:
        print("[WARN] high-risk PPO config: use_clipped_value_loss=False")
    if agent_cfg.algorithm.learning_rate >= 1.0e-3:
        print("[WARN] high-risk PPO config: learning_rate>=1e-3")
    if agent_cfg.algorithm.value_loss_coef > 0.5:
        print("[WARN] high-risk PPO config: value_loss_coef>0.5")
    if getattr(agent_cfg.algorithm, "max_grad_norm", None) is None:
        print("[WARN] high-risk PPO config: max_grad_norm missing")
    if not std_positive_guard_enabled(agent_cfg.policy):
        print("[WARN] high-risk PPO config: std is not log-parameterized and has no positive guard")

    print("obs_has_nan_after_reset:", bool(torch.isnan(policy_obs).any().item()))
    zero_action = torch.zeros_like(env.unwrapped.action_manager.action)
    for step in range(10):
        obs, rewards, terminated, truncated, _ = env.step(zero_action)
        print(
            f"zero_step={step} "
            f"reward_mean={rewards.mean().item():.6f} "
            f"terminated={bool(terminated.any().item())} "
            f"truncated={bool(truncated.any().item())} "
            f"obs_nan={bool(torch.isnan(obs['policy']).any().item())}"
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
