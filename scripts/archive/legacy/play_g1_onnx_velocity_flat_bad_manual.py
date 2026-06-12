import argparse
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Unitree G1 29DOF ONNX velocity policy in Isaac Lab.")
parser.add_argument("--vx", type=float, default=0.3, help="Forward velocity command.")
parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command.")
parser.add_argument("--wz", type=float, default=0.0, help="Yaw velocity command.")
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


POLICY_PATH = "/workspace/data/repos/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse quaternion q. q is [w, x, y, z]."""
    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    # inverse(q) = [w, -x, -y, -z]
    # use quaternion-vector rotation formula
    # t = 2 * cross(q_vec_inv, v)
    qvec = torch.stack((-qx, -qy, -qz), dim=-1)
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + qw.unsqueeze(-1) * t + torch.cross(qvec, t, dim=-1)


def main():
    env_cfg = parse_env_cfg(
        "Unitree-G1-29dof-Velocity",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )

    # 减少随机扰动，先让官方 policy 在干净平地上跑
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make("Unitree-G1-29dof-Velocity", cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    sess = ort.InferenceSession(POLICY_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    num_actions = robot.num_joints
    assert num_actions == 29, f"Expected 29 joints, got {num_actions}"

    default_joint_pos = robot.data.default_joint_pos[0].detach().clone().to(device)
    last_action = torch.zeros((1, num_actions), device=device)

    # deploy.yaml 里所有 policy observation history_length = 5
    obs_history = deque(maxlen=5)

    cmd = torch.tensor([[args_cli.vx, args_cli.vy, args_cli.wz]], dtype=torch.float32, device=device)

    print("=== G1 ONNX Velocity Policy ===")
    print("policy:", POLICY_PATH)
    print("cmd:", cmd.detach().cpu().numpy())
    print("num_joints:", robot.num_joints)
    print("joint_names:")
    for i, name in enumerate(robot.joint_names):
        print(f"{i:02d}: {name}")

    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # 读取机器人状态，env 只有 num_envs=1 时取第 0 个
            joint_pos = robot.data.joint_pos[0:1]
            joint_vel = robot.data.joint_vel[0:1]

            # Isaac Lab root quaternion 通常是 [w, x, y, z]
            root_quat_w = robot.data.root_quat_w[0:1]
            root_ang_vel_w = robot.data.root_ang_vel_w[0:1]

            # 转到 base/body 坐标系
            base_ang_vel = quat_rotate_inverse(root_quat_w, root_ang_vel_w) * 0.2

            gravity_w = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device)
            projected_gravity = quat_rotate_inverse(root_quat_w, gravity_w)

            joint_pos_rel = joint_pos - default_joint_pos.unsqueeze(0)
            joint_vel_rel = joint_vel * 0.05

            # 单帧 96 维：
            # 3 ang vel + 3 gravity + 3 cmd + 29 q + 29 dq + 29 last action = 96
            obs_single = torch.cat(
                [
                    base_ang_vel,
                    projected_gravity,
                    cmd,
                    joint_pos_rel,
                    joint_vel_rel,
                    last_action,
                ],
                dim=-1,
            )

            if len(obs_history) == 0:
                for _ in range(5):
                    obs_history.append(obs_single.clone())
            else:
                obs_history.append(obs_single.clone())

            obs = torch.cat(list(obs_history), dim=-1)
            assert obs.shape[-1] == 480, f"obs shape wrong: {obs.shape}"

            obs_np = obs.detach().cpu().numpy().astype(np.float32)
            action_np = sess.run([output_name], {input_name: obs_np})[0]
            action = torch.from_numpy(action_np).to(device=device, dtype=torch.float32)

            # deploy.yaml: scale = 0.25, offset = default_joint_pos
            target_joint_pos = action * 0.25 + default_joint_pos.unsqueeze(0)

            # 写入 Isaac Lab 机器人关节目标
            robot.set_joint_position_target(target_joint_pos)

            # 推进一步仿真
            env.step(action)

            last_action = action.clone()

            count += 1
            if count % 100 == 0:
                root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                print(f"[step {count}] root_pos={root_pos}, cmd={cmd.detach().cpu().numpy()[0]}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
