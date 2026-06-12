from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .ulc_action_math import apply_incremental_delay_step, compose_ulc_joint_position_targets


class ULCJointPositionAction(JointAction):
    """ULC joint-position target action with absolute arm commands plus residuals.

    Legs and waist are residual targets around the robot default pose. Arms are
    absolute desired joint positions from a command term, with the policy action
    applied only as a residual around that desired arm pose.
    """

    cfg: "ULCJointPositionActionCfg"

    def __init__(self, cfg: "ULCJointPositionActionCfg", env):
        super().__init__(cfg, env)

        self._step_dt = float(getattr(env, "step_dt", getattr(env, "physics_dt", 0.02)))
        self._arm_command_start, self._arm_command_stop = self.cfg.arm_command_slice
        if self._arm_command_stop <= self._arm_command_start:
            raise ValueError(
                f"Invalid arm_command_slice={self.cfg.arm_command_slice}; stop must be larger than start."
            )
        if not 0.0 <= self.cfg.arm_delay_probability <= 1.0:
            raise ValueError(f"arm_delay_probability must be in [0, 1], got {self.cfg.arm_delay_probability}.")

        self._leg_local_ids, self._waist_local_ids, self._arm_local_ids = self._resolve_joint_groups()
        self._arm_dim = len(self._arm_local_ids)
        self._arm_command_dim = self._arm_command_stop - self._arm_command_start
        if self._arm_dim != self._arm_command_dim:
            raise ValueError(
                f"Arm command dimension ({self._arm_command_dim}) must match resolved arm joints ({self._arm_dim}). "
                f"Arm joints: {self._selected_names(self._arm_local_ids)}"
            )

        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._arm_default = self._default_joint_pos[:, self._arm_local_ids].clone()

        self._arm_desired = self._arm_default.clone()
        self._arm_interp_start = self._arm_default.clone()
        self._arm_interp_goal = self._arm_default.clone()
        self._arm_interp_time = torch.zeros(self.num_envs, device=self.device)
        self._history_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._arm_prev_theoretical = self._arm_default.clone()
        self._arm_delay_buffer = torch.zeros_like(self._arm_default)
        self._arm_executed = self._arm_default.clone()
        self.delay_ratio = torch.zeros(self.num_envs, device=self.device)

    def _resolve_joint_groups(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        leg_ids: list[int] = []
        waist_ids: list[int] = []
        arm_ids: list[int] = []
        unknown_names: list[str] = []

        for local_id, joint_name in enumerate(self._joint_names):
            if any(token in joint_name for token in self.cfg.leg_joint_name_tokens):
                leg_ids.append(local_id)
            elif any(token in joint_name for token in self.cfg.waist_joint_name_tokens):
                waist_ids.append(local_id)
            elif any(token in joint_name for token in self.cfg.arm_joint_name_tokens):
                arm_ids.append(local_id)
            else:
                unknown_names.append(joint_name)

        if unknown_names:
            raise ValueError(
                "ULCJointPositionAction could not classify these joints: "
                f"{unknown_names}. Add matching name tokens in the action config."
            )
        if not leg_ids or not waist_ids or not arm_ids:
            raise ValueError(
                "ULCJointPositionAction requires non-empty leg, waist, and arm joint groups. "
                f"Resolved leg={self._selected_names(leg_ids)}, waist={self._selected_names(waist_ids)}, "
                f"arm={self._selected_names(arm_ids)}"
            )

        return (
            torch.tensor(leg_ids, dtype=torch.long, device=self.device),
            torch.tensor(waist_ids, dtype=torch.long, device=self.device),
            torch.tensor(arm_ids, dtype=torch.long, device=self.device),
        )

    def _selected_names(self, local_ids: Sequence[int] | torch.Tensor) -> list[str]:
        if isinstance(local_ids, torch.Tensor):
            local_ids = local_ids.detach().cpu().tolist()
        return [self._joint_names[i] for i in local_ids]

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _read_arm_command(self) -> torch.Tensor:
        command = self._env.command_manager.get_command(self.cfg.command_name)
        if command.shape[1] < self._arm_command_stop:
            raise ValueError(
                f"Command '{self.cfg.command_name}' has dim {command.shape[1]}, but arm_command_slice "
                f"{self.cfg.arm_command_slice} requires dim >= {self._arm_command_stop}."
            )
        return command[:, self._arm_command_start : self._arm_command_stop]

    def _initialize_arm_command(self, raw_arm_command: torch.Tensor) -> None:
        first_update_envs = (~self._history_initialized).nonzero(as_tuple=False).flatten()
        if first_update_envs.numel() > 0:
            self._arm_interp_start[first_update_envs] = self._arm_desired[first_update_envs]
            self._arm_interp_goal[first_update_envs] = raw_arm_command[first_update_envs]
            self._arm_prev_theoretical[first_update_envs] = self._arm_desired[first_update_envs]
            self._arm_executed[first_update_envs] = self._arm_desired[first_update_envs]
            self._arm_delay_buffer[first_update_envs] = 0.0
            self._arm_interp_time[first_update_envs] = 0.0
            self._history_initialized[first_update_envs] = True

    def _interpolate_arm_command(self, arm_command: torch.Tensor) -> torch.Tensor:
        changed = torch.linalg.norm(arm_command - self._arm_interp_goal, dim=1) > self.cfg.arm_target_epsilon
        if changed.any():
            self._arm_interp_start[changed] = self._arm_desired[changed]
            self._arm_interp_goal[changed] = arm_command[changed]
            self._arm_interp_time[changed] = 0.0

        if self.cfg.arm_interp_duration_s <= 0.0:
            self._arm_desired[:] = self._arm_interp_goal
            return self._arm_desired

        self._arm_interp_time = torch.clamp(
            self._arm_interp_time + self._step_dt,
            min=0.0,
            max=self.cfg.arm_interp_duration_s,
        )
        phase = self._arm_interp_time / self.cfg.arm_interp_duration_s
        blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
        self._arm_desired[:] = self._arm_interp_start + blend.unsqueeze(1) * (
            self._arm_interp_goal - self._arm_interp_start
        )
        return self._arm_desired

    def _apply_stochastic_delay(self, theoretical_arm_command: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable_stochastic_delay or self.cfg.arm_delay_probability <= 0.0:
            self._arm_prev_theoretical[:] = theoretical_arm_command
            self._arm_delay_buffer[:] = 0.0
            self._arm_executed[:] = theoretical_arm_command
            self.delay_ratio[:] = 0.0
            return self._arm_executed

        delay_mask = torch.rand_like(theoretical_arm_command) < self.cfg.arm_delay_probability
        self._arm_prev_theoretical[:], self._arm_delay_buffer[:], self._arm_executed[:] = apply_incremental_delay_step(
            self._arm_prev_theoretical,
            self._arm_delay_buffer,
            self._arm_executed,
            theoretical_arm_command,
            delay_mask,
        )
        self.delay_ratio[:] = delay_mask.float().mean(dim=1)
        return self._arm_executed

    def _clamp_to_joint_limits(self, joint_targets: torch.Tensor) -> torch.Tensor:
        if hasattr(self._asset.data, "soft_joint_pos_limits"):
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        else:
            limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        return torch.clamp(joint_targets, min=limits[..., 0], max=limits[..., 1])

    @property
    def desired_arm_pos(self) -> torch.Tensor:
        """Delayed/interpolated absolute arm target currently used by the ULC action."""
        return self._arm_executed

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        scaled_actions = self._raw_actions * self._scale

        raw_arm_command = self._read_arm_command()
        self._initialize_arm_command(raw_arm_command)
        theoretical_arm_pos = self._interpolate_arm_command(raw_arm_command)
        desired_arm_pos = self._apply_stochastic_delay(theoretical_arm_pos)

        joint_targets = compose_ulc_joint_position_targets(
            self._default_joint_pos,
            scaled_actions,
            desired_arm_pos,
            self._leg_local_ids,
            self._waist_local_ids,
            self._arm_local_ids,
            enable_residual_action=self.cfg.enable_residual_action,
        )

        self._processed_actions[:] = self._clamp_to_joint_limits(joint_targets)
        if self.cfg.clip is not None:
            self._processed_actions[:] = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        resolved_env_ids = self._env_ids_tensor(env_ids)
        self._arm_desired[resolved_env_ids] = self._arm_default[resolved_env_ids]
        self._arm_interp_start[resolved_env_ids] = self._arm_default[resolved_env_ids]
        self._arm_interp_goal[resolved_env_ids] = self._arm_default[resolved_env_ids]
        self._arm_prev_theoretical[resolved_env_ids] = self._arm_default[resolved_env_ids]
        self._arm_delay_buffer[resolved_env_ids] = 0.0
        self._arm_executed[resolved_env_ids] = self._arm_default[resolved_env_ids]
        self._arm_interp_time[resolved_env_ids] = 0.0
        self.delay_ratio[resolved_env_ids] = 0.0
        self._history_initialized[resolved_env_ids] = False


@configclass
class ULCJointPositionActionCfg(JointActionCfg):
    """Configuration for the ULC 29-DoF joint position action."""

    class_type: type[ActionTerm] = ULCJointPositionAction

    command_name: str = "ulc_command"
    """Name of the command term that provides the 21D ULC command."""

    arm_command_slice: tuple[int, int] = (7, 21)
    """Slice of the ULC command containing the absolute 14D arm joint target."""

    arm_interp_duration_s: float = 0.12
    """Quintic interpolation duration for delayed arm target changes."""

    enable_stochastic_delay: bool = True
    """Whether to apply ULC-style stochastic delay to arm command increments."""

    arm_delay_probability: float = 0.5
    """Per-dimension Bernoulli probability for delaying an arm command increment."""

    arm_target_epsilon: float = 1.0e-4
    """Minimum target change that restarts arm interpolation."""

    enable_residual_action: bool = True
    """Whether policy arm outputs are residuals around the absolute desired arm command."""

    leg_joint_name_tokens: tuple[str, ...] = ("_hip_", "_knee_", "_ankle_")
    waist_joint_name_tokens: tuple[str, ...] = ("waist_",)
    arm_joint_name_tokens: tuple[str, ...] = ("_shoulder_", "_elbow_", "_wrist_")
