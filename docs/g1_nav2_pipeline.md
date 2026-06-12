# G1 Nav2 Locomotion Pipeline

This pipeline treats the G1 policy as a Nav2-compatible `cmd_vel` tracking base controller.

```text
Isaac sensors or real sensors
  -> SLAM / localization
  -> Nav2 global planner
  -> Nav2 local planner and obstacle avoidance
  -> /cmd_vel
  -> G1 policy bridge
  -> Unitree G1 29DOF action
```

## Locomotion Bridge

Run the Isaac Lab policy bridge. By default it loads the current demo candidate `model_6000.pt`.
The launcher starts two processes:

- a system-Python ROS 2 node that handles `/cmd_vel`, `/odom`, `/tf`, and `/clock`
- an Isaac-Python policy player that runs the G1 simulation and policy inference

They communicate over localhost UDP. This split is intentional: ROS 2 Jazzy installs
`rclpy` for system Python 3.12, while Isaac Sim uses its own Python 3.11, so importing
`rclpy` directly inside Isaac can fail.

```bash
cd /workspace/data/repos/unitree_rl_lab
bash scripts/demo.sh policy-empty
```

Headless:

```bash
HEADLESS=true bash scripts/demo.sh policy-empty
```

Use a different checkpoint:

```bash
CHECKPOINT=logs/rsl_rl/unitree_g1_29dof_velocity/<RUN_DIR>/model_10000.pt \
bash scripts/demo.sh policy-empty
```

The bridge:

- subscribes to `/cmd_vel`
- publishes `/odom`
- publishes TF `odom -> base_link`
- publishes `/clock`
- clamps commands to conservative humanoid-safe limits
- zeroes odom at the first Isaac pose, so Nav2 starts near `(0, 0)` even if Isaac spawns the robot far from world origin

Default command limits:

```text
vx: -0.10 to 0.45 m/s
vy: -0.08 to 0.08 m/s
wz: -0.15 to 0.15 rad/s
```

The bridge expects `/cmd_vel` to be a heartbeat, just like a real navigation controller.
For manual tests, publish with `-r 10` or higher. If you publish only once, the bridge will
stop the robot after `COMMAND_TIMEOUT` seconds. For a demo-only hold-last-command test, run:

```bash
DISABLE_COMMAND_TIMEOUT=true bash scripts/demo.sh policy-empty
```

## Quick Command Test

In another terminal:

```bash
set +u
source /opt/ros/jazzy/setup.bash
set -u
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.35, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10
```

Turn:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.25, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.10}}" -r 10
```

Stop:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
"{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10
```

## Nav2 Bringup

### Empty-Map Navigation Smoke Run

Before adding LiDAR/SLAM, validate that Nav2 can drive the G1 policy through `/cmd_vel`.
This uses an empty map and an identity `map -> odom` transform, so it is for path-following
integration only, not obstacle avoidance.

Terminal 1, G1 policy bridge:

```bash
cd /workspace/data/repos/unitree_rl_lab
bash scripts/demo.sh policy-empty
```

Terminal 2, Nav2 empty-map stack:

```bash
cd /workspace/data/repos/unitree_rl_lab
bash scripts/demo.sh nav2-empty
```

This uses `launch/g1_nav2_empty_launch.py`, a minimal Nav2 bringup for the first
integration step. It starts `map_server`, planner, controller, BT navigator, waypoint follower,
and velocity smoother. It intentionally does not start `collision_monitor`, docking, AMCL, or SLAM
because the empty-map smoke test has no `/scan` yet. The velocity smoother remaps
`cmd_vel_smoothed` back to `/cmd_vel`, which feeds the G1 bridge.

Terminal 3, send a short goal:

```bash
cd /workspace/data/repos/unitree_rl_lab
set +u
source /opt/ros/jazzy/setup.bash
set -u
python3 scripts/_internal/tasks/send_g1_nav2_goal.py --x 2.0 --y 0.0 --yaw 0.0
```

Expected signs of life:

- `ros2_g1_udp_bridge.py` logs received `/cmd_vel` messages.
- `play_g1_policy_cmdvel_udp.py` logs non-zero `cmd`.
- The robot walks roughly toward positive `x`.
- `/odom`, `/tf`, and `/clock` are visible with `ros2 topic list`.

This mode has no `/scan`, so it cannot avoid obstacles yet.

Use the parameter template:

```bash
configs/nav2/g1_nav2_params.yaml
```

Known-map localization flow:

```bash
set +u
source /opt/ros/jazzy/setup.bash
set -u
ros2 launch nav2_bringup localization_launch.py \
  map:=/path/to/map.yaml \
  params_file:=/workspace/data/repos/unitree_rl_lab/configs/nav2/g1_nav2_params.yaml \
  use_sim_time:=true

ros2 launch nav2_bringup navigation_launch.py \
  params_file:=/workspace/data/repos/unitree_rl_lab/configs/nav2/g1_nav2_params.yaml \
  use_sim_time:=true
```

Mapping flow with `slam_toolbox`:

```bash
set +u
source /opt/ros/jazzy/setup.bash
set -u
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=/workspace/data/repos/unitree_rl_lab/configs/nav2/g1_nav2_params.yaml \
  use_sim_time:=true

ros2 launch nav2_bringup navigation_launch.py \
  params_file:=/workspace/data/repos/unitree_rl_lab/configs/nav2/g1_nav2_params.yaml \
  use_sim_time:=true
```

## Sensor Requirement

Nav2 needs obstacle observations such as `/scan` or point clouds. The policy bridge publishes odometry and TF only.
For full obstacle avoidance in Isaac Lab, add an Isaac LiDAR/ROS publisher that publishes:

```text
/scan              sensor_msgs/LaserScan
TF base_link -> lidar frame
```

You can initially test planning with a known static map and no dynamic obstacles, then add `/scan` for local obstacle avoidance.

## Demo Guidance

- Use `model_6000.pt` first because it has more natural arm posture.
- Prefer patrol speeds around `vx=0.30` to `0.45`.
- Keep `wz` under `0.15 rad/s`.
- Avoid aggressive stop/start commands; use Nav2 velocity smoothing.
- If the robot walks but looks awkward at very low speed, increase Nav2 minimum cruising speed rather than forcing `vx=0.1`.
