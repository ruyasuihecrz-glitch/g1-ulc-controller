from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def gated_foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    motion_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward foot clearance only when commanded and actual base motion are non-trivial.

    This keeps the official foot-clearance shaping, but prevents a standing policy from
    collecting swing-foot reward by lifting or jittering its feet without translating or turning.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.sqrt(torch.square(command[:, 0]) + torch.square(command[:, 2]))
    actual_motion = torch.sqrt(
        torch.square(asset.data.root_lin_vel_b[:, 0]) + torch.square(asset.data.root_ang_vel_b[:, 2])
    )
    motion_gate = torch.logical_and(command_norm > command_threshold, actual_motion > motion_threshold)
    return foot_clearance_reward(env, asset_cfg, target_height, std, tanh_mult) * motion_gate.float()


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


def lateral_vel_when_no_vy_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral drift when the commanded lateral velocity is close to zero.

    The penalty is the squared body-frame base lateral velocity and is only active when
    ``abs(command[:, 1]) < threshold``.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask = torch.abs(command[:, 1]) < threshold
    penalty = torch.square(asset.data.root_lin_vel_b[:, 1])
    return penalty * mask.float()


def yaw_rate_when_no_wz_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw drift when the commanded yaw rate is close to zero.

    The penalty is the squared body-frame yaw angular velocity and is only active when
    ``abs(command[:, 2]) < threshold``.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    mask = torch.abs(command[:, 2]) < threshold
    penalty = torch.square(asset.data.root_ang_vel_b[:, 2])
    return penalty * mask.float()


def forward_motion_when_commanded_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize missing forward motion when a positive forward command is active.

    This term is intentionally one-sided for the first Nav2 patrol policy: it does not
    penalize stop commands and it does not assume backward walking support.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    missing_speed = torch.clamp(command[:, 0] - asset.data.root_lin_vel_b[:, 0], min=0.0)
    return torch.square(missing_speed) * (command[:, 0] > threshold).float()


def turn_when_commanded_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize missing yaw-rate response when a turn command is active."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_wz = command[:, 2]
    actual_along_cmd = asset.data.root_ang_vel_b[:, 2] * torch.sign(cmd_wz)
    missing_turn = torch.clamp(torch.abs(cmd_wz) - actual_along_cmd, min=0.0)
    return torch.square(missing_turn) * (torch.abs(cmd_wz) > threshold).float()


def selected_action_rate_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize the rate of change of selected joint action dimensions using an L2 squared kernel."""

    return torch.sum(
        torch.square(
            env.action_manager.action[:, asset_cfg.joint_ids] - env.action_manager.prev_action[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def low_command_base_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal base drift when the requested base motion is near zero."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.sqrt(torch.square(command[:, 0]) + torch.square(command[:, 1]) + torch.square(command[:, 2]))
    penalty = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    return penalty * (command_norm < threshold).float()


def low_command_base_ang_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body angular motion when standing for upper-body manipulation."""

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_norm = torch.sqrt(torch.square(command[:, 0]) + torch.square(command[:, 1]) + torch.square(command[:, 2]))
    penalty = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return penalty * (command_norm < threshold).float()


def upper_body_com_xy_deadband_l2(
    env: ManagerBasedRLEnv,
    deadband: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Softly penalize large upper-body XY center shifts relative to the base.

    This is an approximate body-average term rather than an exact mass-weighted COM.
    It is meant as a conservative guardrail for bad arm/waist branches while still
    allowing moderate reaching motions.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    upper_body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :].mean(dim=1)
    rel_w = upper_body_pos_w - asset.data.root_pos_w
    rel_b = quat_apply_inverse(asset.data.root_quat_w, rel_w)
    offset_xy = torch.linalg.norm(rel_b[:, :2], dim=1)
    return torch.square(torch.clamp(offset_xy - deadband, min=0.0))


def bad_upper_body_branch_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize dynamically ugly upper-body branches seen to destabilize the base."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    deviation = joint_pos - default_pos
    return torch.sum(torch.square(deviation), dim=1)
