"""Arm-disturbance stability fine-tuning task for G1 29DoF.

The manipulation stack owns the arm motion during button pressing.  This task
therefore trains the locomotion policy to keep tracking low-speed commands and
remain stable while the arms are moved into varied poses.  Arm pose changes are
treated as disturbances; waist and leg joints are not randomized by this task.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from . import turn_finetune_env_cfg as turn_finetune


ARM_JOINTS = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]


@configclass
class ArmDisturbanceCommandsCfg(turn_finetune.TurnFineTuneCommandsCfg):
    """Bias samples toward stand/low-speed operation instead of turn-only tuning."""

    base_velocity = mdp.Nav2VelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 9.0),
        rel_standing_envs=0.35,
        rel_straight_envs=0.35,
        rel_turn_envs=0.15,
        min_forward_speed=0.02,
        min_turn_rate=0.04,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.Nav2VelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.25),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.18, 0.18),
        ),
        limit_ranges=mdp.Nav2VelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.60),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.60, 0.60),
        ),
    )


@configclass
class ArmDisturbanceEventsCfg(turn_finetune.official_velocity.EventCfg):
    """Keep default events and move only arm joints as external disturbances."""

    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "position_range": (-0.35, 0.35),
            "velocity_range": (-0.25, 0.25),
        },
    )

    interval_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "position_range": (-0.45, 0.45),
            "velocity_range": (-0.35, 0.35),
        },
    )


@configclass
class ArmDisturbanceRewardsCfg(turn_finetune.TurnFineTuneRewardsCfg):
    """Low-speed tracking and base stability under arm-only disturbances."""

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.6,
        params={"command_name": "base_velocity", "std": 0.18},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.4,
        params={"command_name": "base_velocity", "std": 0.16},
    )
    forward_motion_when_commanded = RewTerm(
        func=mdp.forward_motion_when_commanded_l2,
        weight=-1.2,
        params={"command_name": "base_velocity", "threshold": 0.02},
    )
    turn_when_commanded = RewTerm(
        func=mdp.turn_when_commanded_l2,
        weight=-0.8,
        params={"command_name": "base_velocity", "threshold": 0.04},
    )

    low_command_base_lin_vel_xy = RewTerm(
        func=mdp.low_command_base_lin_vel_xy_l2,
        weight=-7.0,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )
    low_command_base_ang_vel = RewTerm(
        func=mdp.low_command_base_ang_vel_l2,
        weight=-3.0,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )
    lateral_drift_when_no_vy = RewTerm(
        func=mdp.lateral_vel_when_no_vy_l2,
        weight=-0.6,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )
    yaw_drift_when_no_wz = RewTerm(
        func=mdp.yaw_rate_when_no_wz_l2,
        weight=-0.35,
        params={"command_name": "base_velocity", "threshold": 0.05},
    )

    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.04)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.40,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.gated_foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "motion_threshold": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # The arms are external manipulation disturbances.  Do not reward the policy
    # for pulling them back to default, but still keep waist/leg posture terms
    # inherited from the locomotion recipe so non-arm joints stay disciplined.
    joint_deviation_arms = None


@configclass
class UpperBodyStabilityRobotEnvCfg(turn_finetune.TurnFineTuneRobotEnvCfg):
    """Fine-tuning environment for base robustness under arm-only motion."""

    commands: ArmDisturbanceCommandsCfg = ArmDisturbanceCommandsCfg()
    events: ArmDisturbanceEventsCfg = ArmDisturbanceEventsCfg()
    rewards: ArmDisturbanceRewardsCfg = ArmDisturbanceRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 16.0


@configclass
class UpperBodyStabilityRobotPlayEnvCfg(UpperBodyStabilityRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
