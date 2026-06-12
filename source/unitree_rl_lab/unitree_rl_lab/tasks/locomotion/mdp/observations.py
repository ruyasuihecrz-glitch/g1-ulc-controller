from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def ulc_executed_command(
    env: ManagerBasedRLEnv, command_name: str = "ulc_command", action_name: str = "JointPositionAction"
) -> torch.Tensor:
    """Return the 21D ULC command with arms replaced by the actually executed delayed arm target."""

    command = env.command_manager.get_command(command_name).clone()
    try:
        action_term = env.action_manager.get_term(action_name)
        if hasattr(action_term, "desired_arm_pos"):
            command[:, 7:21] = action_term.desired_arm_pos
    except Exception:
        pass
    return command
