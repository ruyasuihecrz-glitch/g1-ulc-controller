import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-StableVelocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stable_velocity_env_cfg:StableRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stable_velocity_env_cfg:StableRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G1StableVelocityPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-TurnFineTune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.turn_finetune_env_cfg:TurnFineTuneRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.turn_finetune_env_cfg:TurnFineTuneRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G1TurnFineTunePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-UpperBodyStabilityFineTune",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.upper_body_stability_finetune_env_cfg:UpperBodyStabilityRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.upper_body_stability_finetune_env_cfg:UpperBodyStabilityRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:"
            "G1UpperBodyStabilityFineTunePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-G1-29dof-ULC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ulc_env_cfg:ULCRobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.ulc_env_cfg:ULCRobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:G1ULCPPORunnerCfg",
    },
)
