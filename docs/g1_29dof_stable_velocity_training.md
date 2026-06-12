# G1 29DOF Stable Velocity Locomotion Training

## Goal

`Unitree-G1-29dof-StableVelocity` is the formal locomotion base policy training task for G1 29DOF patrol and navigation. It is intended to produce a policy that can later back a skill layer with:

1. `walk`
2. `stand`
3. `turn`
4. `sidestep`
5. `stop_transition`

This environment is not a standalone rewrite. It is a formal incremental modification of the official:

`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`

## Why Not Reuse `velocity/v0`

The official `velocity/v0` policy can walk in Isaac Lab deploy, but it shows:

1. noticeable high-frequency jitter
2. yaw drift when `wz=0`
3. lateral drift when `vy=0`
4. weak low-speed stability

Those behaviors are not good enough for the patrol demo quality bar.

## Differences From Official `velocity_env_cfg.py`

All changes are explainable deltas on top of the official G1 29DOF velocity task structure:

1. `RobotSceneCfg`: still uses a valid terrain generator, but now mixes mostly flat terrain with mild roughness.
2. `EventCfg`: keeps the official event structure, but broadens formal friction/mass randomization and uses a stable warm-start disturbance schedule for `push_robot`.
3. `CommandsCfg`: still uses `UniformLevelVelocityCommandCfg`, but shifts the curriculum toward stable patrol-speed commands.
4. `ActionsCfg`: unchanged from the official task, still `JointPositionAction(scale=0.25, use_default_offset=True)`.
5. `ObservationsCfg`: keeps the official observation order and only adds `gait_phase`.
6. `RewardsCfg`: keeps the official reward set, changes weights for stability, and adds anti-drift plus upper-body smoothness terms.
7. `TerminationsCfg`: uses a stable warm-start `bad_orientation=1.0` so the random initial policy is not immediately truncated by posture failure.
8. `CurriculumCfg`: keeps `lin_vel_cmd_levels`; `terrain_levels` remains disabled because the formal target is patrol-grade locomotion rather than rough-terrain mastery.
9. `Export`: uses a separate `stable_velocity/v0` deploy path and does not overwrite `velocity/v0`.

## Reward Table

### Task

| Reward | Weight |
| --- | ---: |
| `track_lin_vel_xy` | `+1.0` |
| `track_ang_vel_z` | `+0.5` |
| `alive` | `+0.15` |

### Base Stability

| Reward | Weight | Notes |
| --- | ---: | --- |
| `base_linear_velocity` | `-2.0` | `lin_vel_z_l2` |
| `base_angular_velocity` | `-0.10` | `ang_vel_xy_l2` |
| `flat_orientation_l2` | `-5.0` | official term retained |
| `base_height` | `-10.0` | target height `0.78` |

### Smoothness

| Reward | Weight |
| --- | ---: |
| `joint_vel` | `-0.002` |
| `joint_acc` | `-5e-7` |
| `action_rate` | `-0.08` |
| `energy` | `-2e-5` |
| `arm_joint_vel` | `-0.002` |
| `waist_joint_vel` | `-0.003` |
| `arm_action_rate` | `-0.02` |

These terms remain part of the formal reward recipe. The warm-start change for current training is not a reward simplification, only an easier startup schedule.

### Joint Safety And Posture

| Reward | Weight |
| --- | ---: |
| `dof_pos_limits` | `-5.0` |
| `joint_deviation_arms` | `-0.3` |
| `joint_deviation_waists` | `-1.5` |
| `joint_deviation_legs` | `-1.0` |

### Feet / Contact

| Reward | Weight |
| --- | ---: |
| `gait` | `+0.5` |
| `feet_slide` | `-0.4` |
| `feet_clearance` | `+1.0` |
| `undesired_contacts` | `-1.0` |

`feet_gait + feet_slide + feet_clearance` are currently the main balance and contact-timing constraints. A dedicated feet contact balance term is not added yet.

### Anti-Drift

| Reward | Weight |
| --- | ---: |
| `lateral_drift_when_no_vy` | `-0.5` |
| `yaw_drift_when_no_wz` | `-0.2` |

## Observation Spec

The official observation order is preserved:

1. `base_ang_vel`
2. `projected_gravity`
3. `velocity_commands`
4. `joint_pos_rel`
5. `joint_vel_rel`
6. `last_action`

New term appended after the official set:

7. `gait_phase`

Configuration:

1. `history_length = 5`
2. `concatenate_terms = True`
3. `enable_corruption = True`
4. `gait_phase period = 0.8`

Shapes:

1. policy obs: `490`
2. critic obs: `505`

Because `gait_phase` is enabled, the new policy is not compatible with the old `velocity/v0` 480-dim deploy configuration. A separate `stable_velocity/v0` deploy config is required.

## Command Curriculum

Formal command settings:

Initial ranges:

1. `lin_vel_x = (-0.05, 0.25)`
2. `lin_vel_y = (-0.05, 0.05)`
3. `ang_vel_z = (-0.08, 0.08)`

Limit ranges:

1. `lin_vel_x = (-0.15, 0.55)`
2. `lin_vel_y = (-0.20, 0.20)`
3. `ang_vel_z = (-0.30, 0.30)`

Curriculum behavior:

1. `rel_standing_envs = 0.15`
2. `lin_vel_cmd_levels` is retained from the official task
3. curriculum grows command difficulty toward patrol-speed motion instead of high-speed running

## Stable Warm-Start Setting

The current formal training launch uses a stable warm-start setting. This is not a fast test and not a temporary debug branch. It is part of the formal training process for a 29DOF humanoid policy that would otherwise terminate almost entirely from `bad_orientation` during the first phase of learning.

Current warm-start choices:

1. dedicated PPO config with `init_noise_std = 0.4`
2. smaller reset yaw range `(-0.1, 0.1)`
3. smaller reset x/y range `(-0.1, 0.1)`
4. weaker push schedule: interval `15.0s` to `20.0s`, planar push `(-0.1, 0.1)`
5. relaxed `bad_orientation` limit angle `1.0`

After the policy becomes stable, stronger push and wider reset yaw randomization can be re-enabled in the same formal environment.

## Randomization And Robustness

Formal event strategy:

1. friction randomization: `static_friction_range=(0.6, 1.3)`, `dynamic_friction_range=(0.6, 1.3)`
2. base mass randomization: `(-0.5, 1.5)`
3. reset pose randomization: `x/y in (-0.1, 0.1)`, `yaw in (-0.1, 0.1)`
4. reset joint position scale: `(0.95, 1.05)`
5. reset joint velocity: `(-0.1, 0.1)`
6. push disturbance enabled through official `push_robot` event:
   interval `15.0s` to `20.0s`
   planar velocity push `(-0.1, 0.1)`

`base_external_force_torque` remains at zero because the formal disturbance injection is handled through `push_robot`.

## Terrain

The task still uses a legal terrain generator and does not set `terrain_generator=None`.

Current formal terrain mix:

1. flat terrain proportion `0.8`
2. mild random uniform roughness proportion `0.2`
3. terrain size `(8.0, 8.0)`
4. `num_rows = 5`
5. `num_cols = 5`

`terrain_levels` is currently disabled because the primary target is stable patrol locomotion, not rough-terrain progression. This avoids mismatches between the training objective and the deployment demo objective.

## Training

Inspect first:

```bash
/workspace/isaaclab/isaaclab.sh -p scripts/_internal/debug/inspect_g1_29dof_stable_env.py --headless
```

Launch formal training:

```bash
NUM_ENVS=4096 MAX_ITERATIONS=50000 SEED=42 bash scripts/demo.sh train-stable
```

This uses tmux session:

`g1_stable_velocity_train`

The default training run name is:

`g1_29dof_stable_velocity_formal`

## PPO Numerical Stability And Resume

During formal training, the locomotion metrics stayed healthy around iteration `8553`:

1. `time_out ≈ 0.98`
2. `bad_orientation ≈ 0.017`
3. `mean reward ≈ 43`
4. `mean episode length ≈ 980`

However, the critic then failed numerically:

1. iteration `8553`: `Mean value_function loss = 0.4955`
2. iteration `8554`: `Mean value_function loss = 1.55862215688192e17`
3. iteration `8555`: `Mean value_function loss = inf`

This indicates a PPO / critic stability failure, not a reward, observation, action, terrain, or event design failure.

The formal StableVelocity PPO config now applies:

1. `use_clipped_value_loss = True`
2. `learning_rate = 3e-4`
3. `value_loss_coef = 0.5`
4. `max_grad_norm = 1.0`
5. `desired_kl = 0.01`
6. `schedule = adaptive`
7. `init_noise_std = 0.4`
8. `noise_std_type = log`

The training entrypoint also adds runtime guards for the StableVelocity task:

1. non-finite obs / rewards / actions stop training immediately
2. non-finite PPO losses stop training immediately
3. actor-critic parameter NaN / Inf stop training immediately
4. policy std is checked to remain positive before sampling

Checkpoint recovery workflow:

1. list candidate checkpoints:

```bash
bash scripts/_internal/training_tools/find_stable_velocity_checkpoints.sh
```

2. resume from a checkpoint before the explosion:

```bash
RESUME=true \
LOAD_RUN=<stable_velocity_run_dir> \
CHECKPOINT=model_8500.pt \
NUM_ENVS=4096 \
MAX_ITERATIONS=50000 \
SEED=42 \
bash scripts/demo.sh train-stable
```

Do not resume from checkpoints written after the value loss became `1e17`, `inf`, or `nan`.

## Evaluation

Interactive rollout through the official play entrypoint:

```bash
bash scripts/play_g1_29dof_stable_velocity.sh play
```

Formal command-suite evaluation:

```bash
bash scripts/play_g1_29dof_stable_velocity.sh eval
```

The evaluation suite tests:

1. `stand`
2. `slow_walk`
3. `medium_walk`
4. `fast_patrol`
5. `turn_left`
6. `turn_right`
7. `side_step`
8. `stop_transition`

It saves results under:

`results/g1_29dof_stable_velocity_eval/<timestamp>/`

## Export

```bash
bash scripts/_internal/export/export_g1_29dof_stable_velocity_policy.sh
```

Output paths:

1. `deploy/robots/g1_29dof/config/policy/stable_velocity/v0/exported/policy.onnx`
2. `deploy/robots/g1_29dof/config/policy/stable_velocity/v0/params/deploy.yaml`

This does not overwrite `deploy/robots/g1_29dof/config/policy/velocity/v0`.

## Success Criteria

The formal policy is considered good when it satisfies most of the following:

1. stand for `20s` without falling
2. walk at `vx=0.10`, `0.20`, and `0.35` without falling
3. keep root height within a tight patrol-stable band
4. reduce yaw drift when `wz=0`
5. reduce lateral drift when `vy=0`
6. keep action-rate jitter visibly lower than the old policy
7. keep arm deviation and waist oscillation small
8. improve `feet_slide` behavior under evaluation

## Skill Layer Interface

This locomotion base is intended to be the callable substrate for future skill-layer commands:

1. `walk`
2. `stand`
3. `turn`
4. `sidestep`
5. `stop_transition`

## Official Basis

This environment is not a fresh design. It is a formal G1 29DOF locomotion environment built as an incremental modification of:

`source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`
