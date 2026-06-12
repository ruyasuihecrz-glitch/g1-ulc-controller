from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, wrap_to_pi

from .commands.ulc_command import G1_29DOF_ARM_JOINT_NAMES


def _robot(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> Articulation:
    return env.scene[asset_cfg.name]


def _command(env, command_name: str = "ulc_command") -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def _body_ids(asset: Articulation, patterns: list[str]) -> list[int]:
    ids = []
    for pattern in patterns:
        try:
            matched_ids, _ = asset.find_bodies(pattern, preserve_order=True)
        except ValueError:
            continue
        for body_id in matched_ids:
            if body_id not in ids:
                ids.append(body_id)
    if not ids:
        raise RuntimeError(f"Could not resolve body patterns {patterns}. Available bodies: {asset.body_names}")
    return ids


def _torso_body_id(asset: Articulation) -> int:
    return _body_ids(asset, ["torso_link", ".*torso.*", ".*trunk.*", "pelvis"])[0]


def _zxy_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return yaw(Z), roll(X), pitch(Y) for R = Rz(yaw) Rx(roll) Ry(pitch)."""
    rot = matrix_from_quat(quat)
    roll = torch.asin(torch.clamp(rot[:, 2, 1], -1.0, 1.0))
    yaw = torch.atan2(-rot[:, 0, 1], rot[:, 1, 1])
    pitch = torch.atan2(-rot[:, 2, 0], rot[:, 2, 2])
    return torch.stack((yaw, roll, pitch), dim=-1)


def _torso_zxy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    body_id = _torso_body_id(asset)
    return _zxy_from_quat(asset.data.body_quat_w[:, body_id])


def _executed_arm_target(env, command_name: str = "ulc_command", action_name: str = "JointPositionAction") -> torch.Tensor:
    try:
        action_term = env.action_manager.get_term(action_name)
        if hasattr(action_term, "desired_arm_pos"):
            return action_term.desired_arm_pos
    except Exception:
        pass
    return _command(env, command_name)[:, 7:21]


def ulc_track_lin_vel_xy_exp(
    env, command_name: str = "ulc_command", sigma: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    err = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2] - _command(env, command_name)[:, :2]), dim=1)
    return torch.exp(-err / sigma**2)


def ulc_track_ang_vel_z_exp(
    env, command_name: str = "ulc_command", sigma: float = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    err = torch.square(asset.data.root_ang_vel_b[:, 2] - _command(env, command_name)[:, 2])
    return torch.exp(-err / sigma**2)


def ulc_track_root_height_exp(
    env, command_name: str = "ulc_command", sigma: float = 0.4, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    err = torch.square(asset.data.root_pos_w[:, 2] - _command(env, command_name)[:, 3])
    return torch.exp(-err / sigma**2)


def ulc_track_arm_joint_pos_exp(
    env,
    command_name: str = "ulc_command",
    action_name: str = "JointPositionAction",
    sigma: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    arm_ids = asset.find_joints(G1_29DOF_ARM_JOINT_NAMES, preserve_order=True)[0]
    err = torch.sum(torch.square(asset.data.joint_pos[:, arm_ids] - _executed_arm_target(env, command_name, action_name)), dim=1)
    return torch.exp(-err / sigma**2)


def ulc_track_torso_yaw_exp(
    env, command_name: str = "ulc_command", sigma: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    err = wrap_to_pi(_torso_zxy(env, asset_cfg)[:, 0] - _command(env, command_name)[:, 4])
    return torch.exp(-torch.square(err) / sigma**2)


def ulc_track_torso_roll_exp(
    env, command_name: str = "ulc_command", sigma: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    err = wrap_to_pi(_torso_zxy(env, asset_cfg)[:, 1] - _command(env, command_name)[:, 5])
    return torch.exp(-torch.square(err) / sigma**2)


def ulc_track_torso_pitch_exp(
    env, command_name: str = "ulc_command", sigma: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    err = wrap_to_pi(_torso_zxy(env, asset_cfg)[:, 2] - _command(env, command_name)[:, 6])
    return torch.exp(-torch.square(err) / sigma**2)


def ulc_track_com_exp(
    env, sigma: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    masses = asset.data.default_mass.to(asset.data.body_pos_w.device)
    if masses.dim() == 2:
        masses = masses.unsqueeze(-1)
    body_com = asset.data.body_com_pos_w if hasattr(asset.data, "body_com_pos_w") else asset.data.body_pos_w
    com_xy = torch.sum(body_com[:, :, :2] * masses, dim=1) / torch.sum(masses, dim=1).clamp(min=1.0e-6)
    foot_ids = _body_ids(asset, ["left_ankle_roll_link", "right_ankle_roll_link", ".*left.*ankle.*", ".*right.*ankle.*"])[:2]
    feet_mid = torch.mean(asset.data.body_pos_w[:, foot_ids, :2], dim=1)
    err = torch.sum(torch.square(com_xy - feet_mid), dim=1)
    return torch.exp(-err / sigma**2)


def ulc_termination(env) -> torch.Tensor:
    return env.termination_manager.terminated.float()


def ulc_z_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.square(_robot(env, asset_cfg).data.root_lin_vel_b[:, 2])


def ulc_energy_abs(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def ulc_joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def ulc_action_rate_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def ulc_base_orientation_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    zxy = _zxy_from_quat(_robot(env, asset_cfg).data.root_quat_w)
    return torch.square(zxy[:, 1]) + torch.square(zxy[:, 2])


def ulc_joint_pos_limit(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    violation = torch.clamp(limits[..., 0] - q, min=0.0) + torch.clamp(q - limits[..., 1], min=0.0)
    return torch.sum(violation, dim=1)


def ulc_joint_effort_limit(env, threshold: float = 0.999, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    tau = torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids])
    limit = threshold * asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum(torch.clamp(tau - limit, min=0.0), dim=1)


def ulc_hip_ankle_deviation(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    ids, names = asset.find_joints([".*_hip_yaw_joint", ".*_ankle_roll_joint", ".*_hip_roll_joint"])
    err = torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids])
    weights = torch.tensor([0.3 if "hip_roll" in name else 0.15 for name in names], device=env.device)
    return torch.sum(err * weights, dim=1)


def ulc_feet_air_time(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.sum(torch.clip(sensor.data.last_air_time[:, sensor_cfg.body_ids], max=0.5), dim=1)


def ulc_feet_force(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.norm(sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1).max(dim=1)[0]
    return torch.sum(torch.clamp(forces - threshold, min=0.0), dim=1)


def ulc_flying(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return (~torch.any(in_contact, dim=1)).float()


def ulc_ankle_orientation(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids].reshape(-1, 4)
    zxy = _zxy_from_quat(quat).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
    return torch.sum(torch.square(zxy[..., 1]) + torch.square(zxy[..., 2]), dim=1)
