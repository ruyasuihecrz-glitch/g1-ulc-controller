from __future__ import annotations

import torch


def compose_ulc_joint_position_targets(
    default_joint_pos: torch.Tensor,
    scaled_actions: torch.Tensor,
    desired_arm_pos: torch.Tensor,
    leg_ids: torch.Tensor,
    waist_ids: torch.Tensor,
    arm_ids: torch.Tensor,
    enable_residual_action: bool = True,
) -> torch.Tensor:
    """Compose ULC PD position targets without double-counting arm defaults."""

    joint_targets = default_joint_pos.clone()
    joint_targets[:, leg_ids] = default_joint_pos[:, leg_ids] + scaled_actions[:, leg_ids]
    joint_targets[:, waist_ids] = default_joint_pos[:, waist_ids] + scaled_actions[:, waist_ids]
    if enable_residual_action:
        joint_targets[:, arm_ids] = desired_arm_pos + scaled_actions[:, arm_ids]
    else:
        joint_targets[:, arm_ids] = desired_arm_pos
    return joint_targets


def apply_incremental_delay_step(
    prev_theoretical: torch.Tensor,
    delay_buffer: torch.Tensor,
    executed: torch.Tensor,
    theoretical: torch.Tensor,
    delay_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one ULC stochastic delay step to command increments."""

    delta_q = theoretical - prev_theoretical
    release_mask = ~delay_mask
    effective_delta = (delta_q + delay_buffer) * release_mask
    next_buffer = (delay_buffer + delta_q) * delay_mask
    next_executed = executed + effective_delta
    return theoretical, next_buffer, next_executed


def ulc_curriculum_ranges(
    default_height: float,
    alpha_height: float,
    alpha_upper: float,
    root_height_range: tuple[float, float] = (0.30, 0.75),
    torso_yaw_range: tuple[float, float] = (-2.62, 2.62),
    torso_roll_range: tuple[float, float] = (-0.52, 0.52),
    torso_pitch_range: tuple[float, float] = (-0.52, 1.57),
) -> dict[str, tuple[float, float]]:
    """Pure helper for ULC command curriculum ranges."""

    ah = max(0.0, min(1.0, float(alpha_height)))
    au = max(0.0, min(1.0, float(alpha_upper)))
    default_height = min(max(float(default_height), root_height_range[0]), root_height_range[1])
    height_min = (default_height - 0.03) + ah * (root_height_range[0] - (default_height - 0.03))
    height_max = (default_height + 0.03) + ah * (root_height_range[1] - (default_height + 0.03))
    height_min = min(max(height_min, root_height_range[0]), root_height_range[1])
    height_max = min(max(height_max, root_height_range[0]), root_height_range[1])
    return {
        "height": (height_min, height_max),
        "torso_yaw": (au * torso_yaw_range[0], au * torso_yaw_range[1]),
        "torso_roll": (au * torso_roll_range[0], au * torso_roll_range[1]),
        "torso_pitch": (au * torso_pitch_range[0], au * torso_pitch_range[1]),
    }
