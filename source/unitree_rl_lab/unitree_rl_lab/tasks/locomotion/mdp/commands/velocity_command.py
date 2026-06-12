from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class Nav2VelocityCommand(UniformVelocityCommand):
    """Mixture sampler for Nav2-style ``cmd_vel`` training.

    The official uniform velocity sampler is still the base implementation, but
    this sampler explicitly allocates reset samples to stop, straight walking,
    turn-in-place, and arc walking. This prevents the command distribution from
    being dominated by tiny near-zero commands that a standing policy can satisfy.
    """

    cfg: "Nav2VelocityCommandCfg"

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_envs = len(env_ids)
        if num_envs == 0:
            return

        self.vel_command_b[env_ids, :] = 0.0
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False

        sample = torch.rand(num_envs, device=self.device)
        stop_cut = self.cfg.rel_standing_envs
        straight_cut = stop_cut + self.cfg.rel_straight_envs
        turn_cut = straight_cut + self.cfg.rel_turn_envs

        stop_mask = sample < stop_cut
        straight_mask = torch.logical_and(sample >= stop_cut, sample < straight_cut)
        turn_mask = torch.logical_and(sample >= straight_cut, sample < turn_cut)
        arc_mask = sample >= turn_cut

        self.is_standing_env[env_ids[stop_mask]] = True

        if straight_mask.any():
            ids = env_ids[straight_mask]
            self.vel_command_b[ids, 0] = self._sample_forward_speed(len(ids))

        if turn_mask.any():
            ids = env_ids[turn_mask]
            self.vel_command_b[ids, 2] = self._sample_signed_yaw_rate(len(ids))

        if arc_mask.any():
            ids = env_ids[arc_mask]
            self.vel_command_b[ids, 0] = self._sample_forward_speed(len(ids))
            self.vel_command_b[ids, 2] = self._sample_signed_yaw_rate(len(ids))

    def _sample_forward_speed(self, num_samples: int) -> torch.Tensor:
        lower = max(float(self.cfg.ranges.lin_vel_x[0]), self.cfg.min_forward_speed)
        upper = float(self.cfg.ranges.lin_vel_x[1])
        if upper <= lower:
            return torch.full((num_samples,), upper, device=self.device)
        return torch.empty(num_samples, device=self.device).uniform_(lower, upper)

    def _sample_signed_yaw_rate(self, num_samples: int) -> torch.Tensor:
        low, high = self.cfg.ranges.ang_vel_z
        min_abs = min(self.cfg.min_turn_rate, max(abs(float(low)), abs(float(high))))
        max_abs = max(abs(float(low)), abs(float(high)))
        if max_abs <= min_abs:
            magnitude = torch.full((num_samples,), max_abs, device=self.device)
        else:
            magnitude = torch.empty(num_samples, device=self.device).uniform_(min_abs, max_abs)
        sign = torch.where(torch.rand(num_samples, device=self.device) < 0.5, -1.0, 1.0)
        return magnitude * sign


@configclass
class Nav2VelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Configuration for the Nav2-compatible velocity mixture sampler."""

    class_type: type = Nav2VelocityCommand
    rel_straight_envs: float = 0.35
    rel_turn_envs: float = 0.25
    min_forward_speed: float = 0.10
    min_turn_rate: float = 0.15
