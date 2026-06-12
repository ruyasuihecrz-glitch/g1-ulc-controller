"""Turn fine-tuning task derived from the official G1 29DOF velocity recipe."""

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from . import velocity_env_cfg as official_velocity


@configclass
class TurnFineTuneCommandsCfg(official_velocity.CommandsCfg):
    """Keep the official velocity command interface, but sample more useful turns."""

    base_velocity = mdp.Nav2VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=0.02,
        rel_straight_envs=0.18,
        rel_turn_envs=0.45,
        min_forward_speed=0.04,
        min_turn_rate=0.18,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.Nav2VelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.30),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.35, 0.35),
        ),
        limit_ranges=mdp.Nav2VelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.60),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.60, 0.60),
        ),
    )


@configclass
class TurnFineTuneRewardsCfg(official_velocity.RewardsCfg):
    """Official reward recipe with stronger yaw-rate learning pressure."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.8,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    turn_when_commanded = RewTerm(
        func=mdp.turn_when_commanded_l2,
        weight=-1.5,
        params={"command_name": "base_velocity", "threshold": 0.12},
    )
    yaw_drift_when_no_wz = RewTerm(
        func=mdp.yaw_rate_when_no_wz_l2,
        weight=-0.2,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )

    # Slightly relax smoothness/energy penalties so the policy can discover a
    # useful stepping pattern for low-speed yaw turns during fine-tuning.
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.03)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.035)
    energy = RewTerm(func=mdp.energy, weight=-1.5e-5)


@configclass
class TurnFineTuneCurriculumCfg(official_velocity.CurriculumCfg):
    """Add angular command curriculum while preserving the official linear one."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


@configclass
class TurnFineTuneRobotEnvCfg(official_velocity.RobotEnvCfg):
    """Official G1 velocity environment with only turn-learning changes."""

    commands: TurnFineTuneCommandsCfg = TurnFineTuneCommandsCfg()
    rewards: TurnFineTuneRewardsCfg = TurnFineTuneRewardsCfg()
    curriculum: TurnFineTuneCurriculumCfg = TurnFineTuneCurriculumCfg()


@configclass
class TurnFineTuneRobotPlayEnvCfg(TurnFineTuneRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
