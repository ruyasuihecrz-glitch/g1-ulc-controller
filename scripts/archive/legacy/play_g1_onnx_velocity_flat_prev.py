import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Unitree G1 29DOF ONNX velocity policy using Isaac Lab env obs/action managers.")
parser.add_argument("--vx", type=float, default=0.3)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=1)
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


def get_policy_obs(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        # fallback: first tensor value
        for v in obs.values():
            if torch.is_tensor(v):
                return v
    return obs


def safe_set(obj, attr, value):
    try:
        setattr(obj, attr, value)
        return True
    except Exception:
        return False


def main():
    # 优先用 play cfg，失败再退回 env cfg
    try:
        env_cfg = parse_env_cfg(
            TASK,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            entry_point_key="play_env_cfg_entry_point",
        )
    except Exception:
        env_cfg = parse_env_cfg(
            TASK,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            entry_point_key="env_cfg_entry_point",
        )

    env_cfg.scene.num_envs = args_cli.num_envs

    # 部署调试：保留 terrain_generator，但只生成 1x1 地形块，让 env_0 回到世界原点附近。
    # 不要把 terrain_generator 设成 None，否则 curriculum 可能访问 terrain_generator.size 报错。
    try:
        tg = env_cfg.scene.terrain.terrain_generator
        tg.num_rows = 1
        tg.num_cols = 1
        env_cfg.scene.terrain.max_init_terrain_level = 0
        print("[info] set terrain generator to 1x1 single flat tile")
    except Exception as e:
        print("[warn] could not set terrain generator to 1x1:", e)

    # 固定速度命令范围，避免随机命令
    try:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)
        env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        env_cfg.commands.base_velocity.limit_ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
        env_cfg.commands.base_velocity.limit_ranges.ang_vel_z = (args_cli.wz, args_cli.wz)
        print(f"[info] fixed command: vx={args_cli.vx}, vy={args_cli.vy}, wz={args_cli.wz}")
    except Exception as e:
        print("[warn] could not fix command ranges:", e)

    # 关掉观测噪声，先看 policy 能不能稳定走
    try:
        env_cfg.observations.policy.enable_corruption = False
        print("[info] disabled observation corruption")
    except Exception as e:
        print("[warn] could not disable obs corruption:", e)

    # 关掉训练用随机项：部署验证时先求稳定，不要随机推、随机质量、随机摩擦、课程学习
    for name in ["push_robot", "base_external_force_torque", "add_base_mass", "physics_material"]:
        try:
            setattr(env_cfg.events, name, None)
            print(f"[info] disabled event: {name}")
        except Exception:
            pass

    # 关掉 terrain / command curriculum，避免 reset 时地形等级和命令等级变化
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

    # 固定 reset 初始姿态：不要随机 yaw，不要随机位置
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
        print("[info] fixed reset_base pose and velocity")
    except Exception as e:
        print("[warn] could not fix reset_base:", e)

    # 关节 reset 不给随机速度
    try:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        print("[info] fixed reset_robot_joints")
    except Exception as e:
        print("[warn] could not fix reset_robot_joints:", e)

    env = gym.make(TASK, cfg=env_cfg)
    obs, info = env.reset()

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    print("\n=== Robot loaded ===")
    print("num_joints:", robot.num_joints)
    print("prim path: /World/envs/env_0/Robot probably")
    print("initial root_pos:", robot.data.root_pos_w[0].detach().cpu().numpy())
    print("If GUI cannot see it: select Robot in Stage and press F in viewport.\n")

    sess = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    print("[info] ONNX input:", sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
    print("[info] ONNX output:", sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            policy_obs = get_policy_obs(obs)

            if not torch.is_tensor(policy_obs):
                raise RuntimeError(f"policy_obs is not tensor: {type(policy_obs)}")

            if torch.isnan(policy_obs).any():
                print("[warn] NaN in obs, resetting env")
                obs, info = env.reset()
                continue

            obs_np = policy_obs.detach().cpu().numpy().astype(np.float32)
            action_np = sess.run([output_name], {input_name: obs_np})[0].astype(np.float32)

            if np.isnan(action_np).any():
                print("[warn] NaN in action, resetting env")
                obs, info = env.reset()
                continue

            action = torch.from_numpy(action_np).to(device=device)

            # 关键：只用 env.step(action)
            # 不再手动 robot.set_joint_position_target
            obs, rew, terminated, truncated, info = env.step(action)

            count += 1
            if count % 100 == 0:
                root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
                print(f"[step {count}] root_pos={root_pos}, root_quat={root_quat}, action_minmax=({action.min().item():+.3f},{action.max().item():+.3f})")

                if np.isnan(root_pos).any():
                    print("[warn] NaN root_pos, resetting env")
                    obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
