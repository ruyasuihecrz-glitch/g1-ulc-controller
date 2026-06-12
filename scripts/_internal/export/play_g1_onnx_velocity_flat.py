import argparse
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Play Unitree G1 29DOF ONNX velocity policy with deploy-style observations."
)
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--history_order", type=str, default="oldest_first", choices=["oldest_first", "newest_first"])
parser.add_argument("--action_clip", type=float, default=2.0)
parser.add_argument("--warmup_steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


TASK = "Unitree-G1-29dof-Velocity"
POLICY_PATH = "/workspace/data/repos/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse quaternion q. q assumed [w, x, y, z]."""
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    qvec_inv = torch.stack((-qx, -qy, -qz), dim=-1)
    t = 2.0 * torch.cross(qvec_inv, v, dim=-1)
    return v + qw.unsqueeze(-1) * t + torch.cross(qvec_inv, t, dim=-1)


def build_deploy_terms(robot, cmd, last_action, default_joint_pos, device):
    """
    Official deploy-style terms, in policy/deploy joint order.

    Terms:
      base_ang_vel       3, scale 0.2
      projected_gravity  3, scale 1.0
      velocity_commands  3, scale 1.0
      joint_pos_rel      29, scale 1.0
      joint_vel_rel      29, scale 0.05
      last_action        29, scale 1.0

    Official deploy ObservationManager has use_gym_history=false by default.
    Therefore final obs layout is term-major:
      base_ang_vel history(5), projected_gravity history(5), cmd history(5),
      joint_pos_rel history(5), joint_vel_rel history(5), last_action history(5)
    """

    # In Isaac Lab, robot.data.joint_pos is already in policy / training joint order.
    # joint_ids_map is only needed in real robot deploy to map Unitree motor_state order
    # into this policy order. Do NOT apply joint_ids_map again here.
    joint_pos = robot.data.joint_pos[0:1]
    joint_vel = robot.data.joint_vel[0:1]

    base_ang_vel = robot.data.root_ang_vel_b[0:1] * 0.2
    projected_gravity = robot.data.projected_gravity_b[0:1]

    joint_pos_rel = joint_pos - default_joint_pos.unsqueeze(0)
    joint_vel_rel = joint_vel * 0.05

    return {
        "base_ang_vel": base_ang_vel,
        "projected_gravity": projected_gravity,
        "velocity_commands": cmd,
        "joint_pos_rel": joint_pos_rel,
        "joint_vel_rel": joint_vel_rel,
        "last_action": last_action,
    }


def flatten_deploy_history(term_histories):
    """
    Match official deploy ObservationManager with use_gym_history=false:
    for each term, concatenate its whole history buffer, then move to next term.
    """
    order = [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    ]
    chunks = []
    for name in order:
        chunks.append(torch.cat(list(term_histories[name]), dim=-1))
    obs = torch.cat(chunks, dim=-1)
    if obs.shape[-1] != 480:
        raise RuntimeError(f"deploy obs dim should be 480, got {obs.shape}")
    return obs


def main():
    env_cfg = parse_env_cfg(
        TASK,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.scene.num_envs = args_cli.num_envs

    # 让 env_0 回到原点附近：保留 terrain_generator，但改成 1x1，不设 None
    try:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.num_rows = 1
        tg.num_cols = 1
        env_cfg.scene.terrain.max_init_terrain_level = 0
        print("[info] terrain generator set to 1x1")
    except Exception as e:
        print("[warn] could not set terrain generator to 1x1:", e)

    # 固定命令，不要随机 resample
    try:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)
        env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        env_cfg.commands.base_velocity.limit_ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
        env_cfg.commands.base_velocity.limit_ranges.ang_vel_z = (args_cli.wz, args_cli.wz)
        env_cfg.commands.base_velocity.resampling_time_range = (9999.0, 9999.0)
        print(f"[info] fixed command: vx={args_cli.vx}, vy={args_cli.vy}, wz={args_cli.wz}")
    except Exception as e:
        print("[warn] could not fix command:", e)

    # 关训练随机项
    for name in ["push_robot", "base_external_force_torque", "add_base_mass", "physics_material"]:
        try:
            setattr(env_cfg.events, name, None)
            print(f"[info] disabled event: {name}")
        except Exception:
            pass

    # 关 curriculum
    try:
        env_cfg.curriculum.terrain_levels = None
        print("[info] disabled curriculum: terrain_levels")
    except Exception:
        pass
    try:
        env_cfg.curriculum.lin_vel_cmd_levels = None
        print("[info] disabled curriculum: lin_vel_cmd_levels")
    except Exception:
        pass

    # 关 observation corruption，虽然我们不用 obs["policy"]，但保持环境干净
    try:
        env_cfg.observations.policy.enable_corruption = False
        print("[info] disabled observation corruption")
    except Exception:
        pass

    # 固定 reset
    try:
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        print("[info] fixed reset pose/joints")
    except Exception as e:
        print("[warn] could not fix reset:", e)

    env = gym.make(TASK, cfg=env_cfg)
    obs, info = env.reset()

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    print("\n=== Robot loaded ===")
    print("num_joints:", robot.num_joints)
    print("initial root_pos:", robot.data.root_pos_w[0].detach().cpu().numpy())
    try:
        print("scene env_origins:", env.unwrapped.scene.env_origins)
    except Exception:
        pass

    print("\n=== joint_names ===")
    for i, name in enumerate(robot.joint_names):
        print(f"{i:02d}: {name}")

    sess = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print("\n[info] ONNX input:", input_name, sess.get_inputs()[0].shape)
    print("[info] ONNX output:", output_name, sess.get_outputs()[0].shape)

    if sess.get_inputs()[0].shape[-1] != 480:
        raise RuntimeError("Expected ONNX input dim 480")
    if sess.get_outputs()[0].shape[-1] != 29:
        raise RuntimeError("Expected ONNX output dim 29")

    # Isaac Lab joint order is already policy / training order.
    default_joint_pos = robot.data.default_joint_pos[0].detach().clone().to(device)

    # last_action is in policy order
    last_action = torch.zeros((args_cli.num_envs, robot.num_joints), device=device)
    cmd = torch.tensor([[args_cli.vx, args_cli.vy, args_cli.wz]], dtype=torch.float32, device=device)

    # 先模拟官方 FixStand：action=0 -> JointPositionAction offset，也就是 default pose
    zero_action = torch.zeros((args_cli.num_envs, robot.num_joints), device=device)
    print(f"[info] warmup default pose for {args_cli.warmup_steps} steps")
    for _ in range(args_cli.warmup_steps):
        obs, rew, terminated, truncated, info = env.step(zero_action)

    last_action.zero_()

    # 初始化 deploy-style term-major histories.
    term_histories = {
        "base_ang_vel": deque(maxlen=5),
        "projected_gravity": deque(maxlen=5),
        "velocity_commands": deque(maxlen=5),
        "joint_pos_rel": deque(maxlen=5),
        "joint_vel_rel": deque(maxlen=5),
        "last_action": deque(maxlen=5),
    }

    terms = build_deploy_terms(robot, cmd, last_action, default_joint_pos, device)
    for name, value in terms.items():
        for _ in range(5):
            term_histories[name].append(value.clone())

    count = 0
    # Debug first deploy observation terms
    debug_terms = build_deploy_terms(robot, cmd, last_action, default_joint_pos, device)
    debug_obs = flatten_deploy_history(term_histories)
    print("[debug] base_ang_vel:", debug_terms["base_ang_vel"].detach().cpu().numpy()[0])
    print("[debug] projected_gravity:", debug_terms["projected_gravity"].detach().cpu().numpy()[0])
    print("[debug] cmd:", debug_terms["velocity_commands"].detach().cpu().numpy()[0])
    print("[debug] joint_pos_rel min/max:", debug_terms["joint_pos_rel"].min().item(), debug_terms["joint_pos_rel"].max().item())
    print("[debug] deploy_obs shape:", tuple(debug_obs.shape))
    print("[debug] deploy_obs min/max:", debug_obs.min().item(), debug_obs.max().item())

    print("[info] start ONNX deploy-style control")

    while simulation_app.is_running():
        with torch.inference_mode():
            terms = build_deploy_terms(robot, cmd, last_action, default_joint_pos, device)
            for name, value in terms.items():
                term_histories[name].append(value.clone())

            # Official deploy default: use_gym_history=false -> term-major history.
            obs_480 = flatten_deploy_history(term_histories)

            if torch.isnan(obs_480).any():
                print("[warn] NaN in obs, reset")
                obs, info = env.reset()
                last_action.zero_()
                for q in term_histories.values():
                    q.clear()
                terms = build_deploy_terms(robot, cmd, last_action, default_joint_pos, device)
                for name, value in terms.items():
                    for _ in range(5):
                        term_histories[name].append(value.clone())
                continue

            action_np = sess.run(
                [output_name],
                {input_name: obs_480.detach().cpu().numpy().astype(np.float32)},
            )[0].astype(np.float32)

            if np.isnan(action_np).any():
                print("[warn] NaN in ONNX action, reset")
                obs, info = env.reset()
                last_action.zero_()
                for q in term_histories.values():
                    q.clear()
                terms = build_deploy_terms(robot, cmd, last_action, default_joint_pos, device)
                for name, value in terms.items():
                    for _ in range(5):
                        term_histories[name].append(value.clone())
                continue

            action = torch.from_numpy(action_np).to(device=device, dtype=torch.float32)

            if args_cli.action_clip > 0:
                action = torch.clamp(action, -args_cli.action_clip, args_cli.action_clip)

            # ONNX action is already in Isaac Lab / policy joint order.
            obs, rew, terminated, truncated, info = env.step(action)

            # last_action in deploy observation is raw ONNX action in policy order
            last_action = action.clone()

            count += 1
            if count % 100 == 0:
                root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
                print(
                    f"[step {count}] "
                    f"root_pos={root_pos}, "
                    f"root_quat={root_quat}, "
                    f"action_minmax=({action.min().item():+.3f},{action.max().item():+.3f}), "
                    f"history_order={args_cli.history_order}"
                )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
