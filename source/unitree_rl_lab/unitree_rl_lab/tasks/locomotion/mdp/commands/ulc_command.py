from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers.command_manager import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_29DOF_ARM_JOINT_NAMES = G1_29DOF_JOINT_NAMES[15:29]


class UniformULCCommand(CommandTerm):
    """Procedurally sampled 21D ULC command for G1 low-level loco-manipulation."""

    cfg: "UniformULCCommandCfg"
    robot: Articulation

    def __init__(self, cfg: "UniformULCCommandCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]

        self.command_tensor = torch.zeros(self.num_envs, 21, device=self.device)
        self._arm_joint_ids, self._arm_joint_names = self.robot.find_joints(
            self.cfg.arm_joint_names, preserve_order=True
        )
        if len(self._arm_joint_ids) != 14:
            raise ValueError(
                f"UniformULCCommand requires 14 arm joints, got {len(self._arm_joint_ids)}: {self._arm_joint_names}"
            )

        self._arm_default = self.robot.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._default_root_height = float(self.cfg.default_root_height)
        if self.cfg.default_root_height < 0.0:
            self._default_root_height = float(self.robot.data.default_root_state[0, 2].item())

        self.alpha_height = float(self.cfg.initial_alpha_height)
        self.alpha_upper = float(self.cfg.initial_alpha_upper)
        self.metrics["command_dim"] = torch.full((self.num_envs,), 21.0, device=self.device)
        self.metrics["alpha_height"] = torch.full((self.num_envs,), self.alpha_height, device=self.device)
        self.metrics["alpha_upper"] = torch.full((self.num_envs,), self.alpha_upper, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_tensor

    def _update_metrics(self):
        self.metrics["arm_command_abs_mean"] = torch.mean(torch.abs(self.command_tensor[:, 7:21]), dim=1)
        self.metrics["alpha_height"][:] = self.alpha_height
        self.metrics["alpha_upper"][:] = self.alpha_upper

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        if self.cfg.mode == "fixed_debug":
            self.command_tensor[env_ids] = torch.tensor(self.cfg.fixed_command, dtype=torch.float32, device=self.device)
            self.command_tensor[env_ids, 7:21] = self._clamp_arm_targets(self.command_tensor[env_ids, 7:21], env_ids)
            return

        ranges = self.cfg.ranges
        self.command_tensor[env_ids, 0] = self._sample_uniform(env_ids.numel(), ranges.lin_vel_x)
        self.command_tensor[env_ids, 1] = self._sample_uniform(env_ids.numel(), ranges.lin_vel_y)
        self.command_tensor[env_ids, 2] = self._sample_uniform(env_ids.numel(), ranges.ang_vel_z)

        if self.cfg.mode == "edge":
            self.command_tensor[env_ids, 3] = self._sample_edges(env_ids.numel(), ranges.root_height)
            self.command_tensor[env_ids, 4] = self._sample_edges(env_ids.numel(), ranges.torso_yaw)
            self.command_tensor[env_ids, 5] = self._sample_edges(env_ids.numel(), ranges.torso_roll)
            self.command_tensor[env_ids, 6] = self._sample_edges(env_ids.numel(), ranges.torso_pitch)
            self.command_tensor[env_ids, 7:21] = self._sample_arm_limit_edges(env_ids)
        else:
            height_range, torso_ranges = self._curriculum_ranges()
            self.command_tensor[env_ids, 3] = self._sample_uniform(env_ids.numel(), height_range)
            self.command_tensor[env_ids, 4] = self._sample_uniform(env_ids.numel(), torso_ranges[0])
            self.command_tensor[env_ids, 5] = self._sample_uniform(env_ids.numel(), torso_ranges[1])
            self.command_tensor[env_ids, 6] = self._sample_uniform(env_ids.numel(), torso_ranges[2])
            self.command_tensor[env_ids, 7:21] = self._sample_arm_targets(env_ids)

    def _update_command(self):
        pass

    def _sample_uniform(self, num_samples: int, value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        return torch.empty(num_samples, device=self.device).uniform_(float(low), float(high))

    def _lerp(self, start: float | torch.Tensor, end: float | torch.Tensor, alpha: float) -> float | torch.Tensor:
        return start + alpha * (end - start)

    def _curriculum_ranges(self) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
        if not self.cfg.enable_curriculum:
            alpha_height = 1.0
            alpha_upper = 1.0
        else:
            alpha_height = float(self.alpha_height)
            alpha_upper = float(self.alpha_upper)

        ranges = self.cfg.ranges
        height_min = self._lerp(self._default_root_height - 0.03, ranges.root_height[0], alpha_height)
        height_max = self._lerp(self._default_root_height + 0.03, ranges.root_height[1], alpha_height)
        torso_yaw = (-alpha_upper * abs(ranges.torso_yaw[1]), alpha_upper * abs(ranges.torso_yaw[1]))
        torso_roll = (-alpha_upper * abs(ranges.torso_roll[1]), alpha_upper * abs(ranges.torso_roll[1]))
        torso_pitch = (
            self._lerp(0.0, ranges.torso_pitch[0], alpha_upper),
            self._lerp(0.0, ranges.torso_pitch[1], alpha_upper),
        )
        return (float(height_min), float(height_max)), (torso_yaw, torso_roll, torso_pitch)

    def _sample_edges(self, num_samples: int, value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        choose_high = torch.rand(num_samples, device=self.device) > 0.5
        return torch.where(
            choose_high,
            torch.full((num_samples,), float(high), device=self.device),
            torch.full((num_samples,), float(low), device=self.device),
        )

    def _arm_limits(self, env_ids: torch.Tensor) -> torch.Tensor:
        if hasattr(self.robot.data, "soft_joint_pos_limits"):
            return self.robot.data.soft_joint_pos_limits[env_ids][:, self._arm_joint_ids]
        return self.robot.data.joint_pos_limits[env_ids][:, self._arm_joint_ids]

    def _sample_arm_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
        limits = self._arm_limits(env_ids)
        lower = limits[..., 0]
        upper = limits[..., 1]
        if self.cfg.enable_curriculum:
            alpha_upper = float(self.alpha_upper)
            near_low = self._arm_default[env_ids] - self.cfg.arm_default_delta
            near_high = self._arm_default[env_ids] + self.cfg.arm_default_delta
            lower = self._lerp(near_low, lower, alpha_upper)
            upper = self._lerp(near_high, upper, alpha_upper)
            lower = torch.clamp(lower, min=limits[..., 0], max=limits[..., 1])
            upper = torch.clamp(upper, min=limits[..., 0], max=limits[..., 1])
        sample = torch.rand(env_ids.numel(), len(self._arm_joint_ids), device=self.device)
        return lower + sample * (upper - lower)

    def _sample_arm_limit_edges(self, env_ids: torch.Tensor) -> torch.Tensor:
        limits = self._arm_limits(env_ids)
        choose_upper = torch.rand(env_ids.numel(), len(self._arm_joint_ids), device=self.device) > 0.5
        return torch.where(choose_upper, limits[..., 1], limits[..., 0])

    def _clamp_arm_targets(self, arm_targets: torch.Tensor, env_ids: torch.Tensor) -> torch.Tensor:
        limits = self._arm_limits(env_ids)
        return torch.clamp(arm_targets, min=limits[..., 0], max=limits[..., 1])

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if self.cfg.mode != "fixed_debug" and self.cfg.start_from_default_pose:
            self.command_tensor[env_ids, 3] = self._default_root_height
            self.command_tensor[env_ids, 4:7] = 0.0
            self.command_tensor[env_ids, 7:21] = self._arm_default[env_ids]
        return extras


@configclass
class UniformULCCommandCfg(CommandTermCfg):
    """Configuration for the 21D ULC command generator."""

    class_type: type = UniformULCCommand

    asset_name: str = MISSING
    mode: str = "normal"
    default_root_height: float = 0.76
    start_from_default_pose: bool = False
    enable_curriculum: bool = True
    initial_alpha_height: float = 0.0
    initial_alpha_upper: float = 0.0
    arm_default_delta: float = 0.05
    arm_joint_names: list[str] = MISSING
    fixed_command: tuple[float, ...] = (0.0,) * 21

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-0.45, 0.55)
        lin_vel_y: tuple[float, float] = (-0.45, 0.45)
        ang_vel_z: tuple[float, float] = (-1.2, 1.2)
        root_height: tuple[float, float] = (0.30, 0.75)
        torso_yaw: tuple[float, float] = (-2.62, 2.62)
        torso_roll: tuple[float, float] = (-0.52, 0.52)
        torso_pitch: tuple[float, float] = (-0.52, 1.57)

    ranges: Ranges = Ranges()
