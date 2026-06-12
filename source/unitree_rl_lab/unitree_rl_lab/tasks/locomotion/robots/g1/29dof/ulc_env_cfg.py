"""ULC low-level controller task scaffold for Unitree G1 29DoF."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG
from . import velocity_env_cfg as official_velocity


@configclass
class ULCCommandsCfg:
    """21D ULC command specification."""

    ulc_command = mdp.UniformULCCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0, 1.0),
        mode="normal",
        start_from_default_pose=True,
        enable_curriculum=True,
        initial_alpha_height=0.0,
        initial_alpha_upper=0.0,
        arm_default_delta=0.05,
        arm_joint_names=mdp.G1_29DOF_ARM_JOINT_NAMES,
        ranges=mdp.UniformULCCommandCfg.Ranges(
            lin_vel_x=(-0.45, 0.55),
            lin_vel_y=(-0.45, 0.45),
            ang_vel_z=(-1.2, 1.2),
            root_height=(0.30, 0.75),
            torso_yaw=(-2.62, 2.62),
            torso_roll=(-0.52, 0.52),
            torso_pitch=(-0.52, 1.57),
        ),
    )


@configclass
class ULCActionsCfg:
    """29D policy action mapped to PD joint-position targets."""

    JointPositionAction = mdp.ULCJointPositionActionCfg(
        asset_name="robot",
        joint_names=mdp.G1_29DOF_JOINT_NAMES,
        preserve_order=True,
        scale=0.25,
        command_name="ulc_command",
        arm_command_slice=(7, 21),
        arm_interp_duration_s=1.0,
        enable_stochastic_delay=True,
        arm_delay_probability=0.5,
        enable_residual_action=True,
    )


@configclass
class ULCObservationsCfg:
    """Actor-critic observations for the unified low-level policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        ulc_commands = ObsTerm(func=mdp.ulc_executed_command, params={"command_name": "ulc_command"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 6
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        ulc_commands = ObsTerm(func=mdp.ulc_executed_command, params={"command_name": "ulc_command"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 6

    critic: CriticCfg = CriticCfg()


@configclass
class ULCRewardsCfg:
    """ULC reward terms and weights."""

    lin_vel = RewTerm(func=mdp.ulc_track_lin_vel_xy_exp, weight=1.0, params={"command_name": "ulc_command", "sigma": 0.5})
    ang_vel = RewTerm(func=mdp.ulc_track_ang_vel_z_exp, weight=1.25, params={"command_name": "ulc_command", "sigma": 0.5})
    height = RewTerm(func=mdp.ulc_track_root_height_exp, weight=1.0, params={"command_name": "ulc_command", "sigma": 0.4})
    arm_tracking = RewTerm(func=mdp.ulc_track_arm_joint_pos_exp, weight=1.0, params={"command_name": "ulc_command", "sigma": 0.35})
    torso_yaw = RewTerm(func=mdp.ulc_track_torso_yaw_exp, weight=0.25, params={"command_name": "ulc_command", "sigma": 0.2})
    torso_roll = RewTerm(func=mdp.ulc_track_torso_roll_exp, weight=0.25, params={"command_name": "ulc_command", "sigma": 0.2})
    torso_pitch = RewTerm(func=mdp.ulc_track_torso_pitch_exp, weight=0.5, params={"command_name": "ulc_command", "sigma": 0.2})
    com_tracking = RewTerm(func=mdp.ulc_track_com_exp, weight=0.5, params={"sigma": 0.2})

    termination = RewTerm(func=mdp.ulc_termination, weight=-200.0)
    z_vel = RewTerm(func=mdp.ulc_z_vel_l2, weight=-1.0)
    energy = RewTerm(func=mdp.ulc_energy_abs, weight=-0.001)
    joint_acc = RewTerm(func=mdp.ulc_joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.ulc_action_rate_l2, weight=-0.1)
    base_orientation = RewTerm(func=mdp.ulc_base_orientation_penalty, weight=-5.0)
    joint_pos_limit = RewTerm(func=mdp.ulc_joint_pos_limit, weight=-2.0)
    joint_effort_limit = RewTerm(
        func=mdp.ulc_joint_effort_limit,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    )
    joint_deviation = RewTerm(func=mdp.ulc_hip_ankle_deviation, weight=-1.0)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.ulc_feet_air_time,
        weight=0.3,
        params={
            "command_name": "ulc_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.ulc_feet_force,
        weight=-3.0e-3,
        params={"threshold": 500.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    feet_stumble = RewTerm(
        func=mdp.ulc_feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    flying = RewTerm(
        func=mdp.ulc_flying,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )
    ankle_orientation = RewTerm(
        func=mdp.ulc_ankle_orientation,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")},
    )


@configclass
class ULCEventsCfg(official_velocity.EventCfg):
    """ULC domain randomization."""

    # ULC paper does not specify locomotion-style random base velocity pushes.
    # Disable the inherited velocity_env_cfg push_robot during staged curriculum training.
    push_robot = None

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.4, 0.7),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    add_wrist_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link", "right_wrist_yaw_link"]),
            "mass_distribution_params": (0.0, 2.0),
            "operation": "add",
        },
    )


@configclass
class ULCCurriculumCfg:
    """ULC staged curriculum."""

    ulc_sequential = CurrTerm(func=mdp.ulc_sequential_curriculum)


@configclass
class ULCRobotEnvCfg(official_velocity.RobotEnvCfg):
    """Trainable ULC low-level controller environment."""

    observations: ULCObservationsCfg = ULCObservationsCfg()
    actions: ULCActionsCfg = ULCActionsCfg()
    commands: ULCCommandsCfg = ULCCommandsCfg()
    rewards: ULCRewardsCfg = ULCRewardsCfg()
    events: ULCEventsCfg = ULCEventsCfg()
    curriculum: ULCCurriculumCfg = ULCCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_29DOF_MIMIC_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.num_envs = 8192
        self.episode_length_s = 20.0


@configclass
class ULCRobotPlayEnvCfg(ULCRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 8
        self.commands.ulc_command.mode = "fixed_debug"
        self.commands.ulc_command.fixed_command = (0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0) + (0.0,) * 14
