from __future__ import annotations

import torch
from collections.abc import Sequence


def _avg_unweighted_reward(env, term_name: str, env_ids: Sequence[int]) -> torch.Tensor:
    if term_name not in env.reward_manager._episode_sums:
        return torch.tensor(0.0, device=env.device)
    cfg = env.reward_manager.get_term_cfg(term_name)
    weight = float(cfg.weight)
    if abs(weight) < 1.0e-8:
        return torch.tensor(0.0, device=env.device)
    value = torch.mean(env.reward_manager._episode_sums[term_name][env_ids]) / env.max_episode_length_s
    return value / weight


def ulc_sequential_curriculum(
    env,
    env_ids: Sequence[int],
    command_name: str = "ulc_command",
    check_interval_steps: int = 1000,
    delta_alpha: float = 0.05,
) -> torch.Tensor:
    """ULC sequential skill acquisition curriculum.

    The global alpha values live on the command term and directly affect command
    sampling ranges. Per-env resets do not reset these alphas. The interval only
    controls how often the paper criteria are checked; alphas advance only when
    the corresponding conditions are satisfied.
    """

    command_term = env.command_manager.get_term(command_name)
    if not getattr(command_term.cfg, "enable_curriculum", True):
        command_term.alpha_height = 1.0
        command_term.alpha_upper = 1.0
        return torch.tensor(command_term.alpha_upper, device=env.device)

    last_step = getattr(command_term, "_last_curriculum_step", -check_interval_steps)
    if env.common_step_counter - last_step < check_interval_steps:
        return torch.tensor(command_term.alpha_upper, device=env.device)

    command_term._last_curriculum_step = env.common_step_counter

    r_height = _avg_unweighted_reward(env, "height", env_ids)
    r_vel = 0.5 * (_avg_unweighted_reward(env, "lin_vel", env_ids) + _avg_unweighted_reward(env, "ang_vel", env_ids))
    hip_dev = _avg_unweighted_reward(env, "joint_deviation", env_ids)
    r_hip = torch.clamp(1.0 - hip_dev, min=0.0, max=1.0)

    c_height = bool(r_height >= 0.83)
    c_velocity = bool(r_vel >= 0.8)
    c_hip = bool(r_hip >= 0.2)
    c2 = c_height and c_velocity and c_hip

    if c2:
        command_term.alpha_height = min(1.0, float(command_term.alpha_height) + delta_alpha)

    r_upper = _avg_unweighted_reward(env, "arm_tracking", env_ids)
    r_torso = (
        _avg_unweighted_reward(env, "torso_yaw", env_ids)
        + _avg_unweighted_reward(env, "torso_roll", env_ids)
        + _avg_unweighted_reward(env, "torso_pitch", env_ids)
    ) / 3.0
    c_upper = bool(r_upper >= 0.8)
    c_torso = bool(r_torso >= 0.8)
    c3 = c_upper and c_torso and c2 and command_term.alpha_height >= 0.98
    if c3:
        command_term.alpha_upper = min(1.0, float(command_term.alpha_upper) + delta_alpha)

    command_term.metrics["alpha_height"][:] = command_term.alpha_height
    command_term.metrics["alpha_upper"][:] = command_term.alpha_upper
    command_term.metrics["curriculum_r_height"] = torch.full((env.num_envs,), float(r_height), device=env.device)
    command_term.metrics["curriculum_r_velocity"] = torch.full((env.num_envs,), float(r_vel), device=env.device)
    command_term.metrics["curriculum_r_hip"] = torch.full((env.num_envs,), float(r_hip), device=env.device)
    command_term.metrics["curriculum_r_upper"] = torch.full((env.num_envs,), float(r_upper), device=env.device)
    command_term.metrics["curriculum_r_torso"] = torch.full((env.num_envs,), float(r_torso), device=env.device)
    command_term.metrics["curriculum_c_height"] = torch.full((env.num_envs,), float(c_height), device=env.device)
    command_term.metrics["curriculum_c_velocity"] = torch.full((env.num_envs,), float(c_velocity), device=env.device)
    command_term.metrics["curriculum_c_hip"] = torch.full((env.num_envs,), float(c_hip), device=env.device)
    command_term.metrics["curriculum_c2"] = torch.full((env.num_envs,), float(c2), device=env.device)
    command_term.metrics["curriculum_c_upper"] = torch.full((env.num_envs,), float(c_upper), device=env.device)
    command_term.metrics["curriculum_c_torso"] = torch.full((env.num_envs,), float(c_torso), device=env.device)
    command_term.metrics["curriculum_c3"] = torch.full((env.num_envs,), float(c3), device=env.device)
    return torch.tensor(command_term.alpha_upper, device=env.device)
