import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Compare Isaac Lab policy obs against official deploy-style obs.")
parser.add_argument("--vx", type=float, default=0.0)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--warmup_steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "Unitree-G1-29dof-Velocity"
DEPLOY_YAML_PATH = REPO_ROOT / "deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml"
POLICY_OBS_TERM_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


@dataclass
class JointOrderInfo:
    need_reorder: bool
    sim_to_policy: list[int]
    policy_to_sim: list[int]


@dataclass
class ObservationTermState:
    name: str
    scale: torch.Tensor | None
    clip: tuple[float, float] | None
    history_length: int
    scale_first: bool
    history: deque


def load_deploy_cfg() -> dict:
    with DEPLOY_YAML_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def configure_env(cfg, deploy_cfg: dict) -> None:
    cfg.scene.num_envs = 1
    cfg.observations.policy.enable_corruption = False
    if hasattr(cfg.observations, "critic"):
        cfg.observations.critic.enable_corruption = False

    terrain_generator = cfg.scene.terrain.terrain_generator
    if terrain_generator is not None:
        terrain_generator.num_rows = 1
        terrain_generator.num_cols = 1
        terrain_generator.curriculum = False
    cfg.scene.terrain.max_init_terrain_level = 0

    cfg.events.physics_material = None
    cfg.events.add_base_mass = None
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.events.reset_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    cfg.commands.base_velocity.ranges.lin_vel_x = tuple(deploy_cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"])
    cfg.commands.base_velocity.ranges.lin_vel_y = tuple(deploy_cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"])
    cfg.commands.base_velocity.ranges.ang_vel_z = tuple(deploy_cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"])
    cfg.commands.base_velocity.limit_ranges.lin_vel_x = tuple(deploy_cfg["commands"]["base_velocity"]["ranges"]["lin_vel_x"])
    cfg.commands.base_velocity.limit_ranges.lin_vel_y = tuple(deploy_cfg["commands"]["base_velocity"]["ranges"]["lin_vel_y"])
    cfg.commands.base_velocity.limit_ranges.ang_vel_z = tuple(deploy_cfg["commands"]["base_velocity"]["ranges"]["ang_vel_z"])
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.commands.base_velocity.debug_vis = False


def clamp_velocity_command(raw_command: torch.Tensor, deploy_cfg: dict) -> torch.Tensor:
    ranges = deploy_cfg["commands"]["base_velocity"]["ranges"]
    cmd = raw_command.clone()
    cmd[:, 0] = cmd[:, 0].clamp(float(ranges["lin_vel_x"][0]), float(ranges["lin_vel_x"][1]))
    cmd[:, 1] = cmd[:, 1].clamp(float(ranges["lin_vel_y"][0]), float(ranges["lin_vel_y"][1]))
    cmd[:, 2] = cmd[:, 2].clamp(float(ranges["ang_vel_z"][0]), float(ranges["ang_vel_z"][1]))
    return cmd


def enforce_fixed_base_velocity_command(env, deploy_cfg: dict, target_command: torch.Tensor) -> torch.Tensor:
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    fixed_command = clamp_velocity_command(target_command, deploy_cfg)
    command_term.vel_command_b[:] = fixed_command
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "heading_target"):
        command_term.heading_target[:] = 0.0
    command_term.time_left[:] = 1.0e9
    return env.unwrapped.command_manager.get_command("base_velocity").clone()


def get_live_base_velocity_command(env, deploy_cfg: dict) -> torch.Tensor:
    return clamp_velocity_command(env.unwrapped.command_manager.get_command("base_velocity").clone(), deploy_cfg)


def make_term_state(name: str, cfg: dict, device: torch.device) -> ObservationTermState:
    scale = cfg.get("scale")
    clip = cfg.get("clip")
    return ObservationTermState(
        name=name,
        scale=None if scale is None else torch.tensor(scale, dtype=torch.float32, device=device).view(1, -1),
        clip=None if clip is None else (float(clip[0]), float(clip[1])),
        history_length=int(cfg.get("history_length", 1)),
        scale_first=bool(cfg.get("scale_first", False)),
        history=deque(maxlen=int(cfg.get("history_length", 1))),
    )


def apply_obs_postprocess(raw: torch.Tensor, term_state: ObservationTermState) -> torch.Tensor:
    obs = raw.clone()
    if term_state.scale_first:
        if term_state.scale is not None:
            obs = obs * term_state.scale
        if term_state.clip is not None:
            obs = obs.clamp(min=term_state.clip[0], max=term_state.clip[1])
    else:
        if term_state.clip is not None:
            obs = obs.clamp(min=term_state.clip[0], max=term_state.clip[1])
        if term_state.scale is not None:
            obs = obs * term_state.scale
    return obs


def build_joint_order_info(robot, deploy_cfg: dict, device: torch.device) -> tuple[JointOrderInfo, torch.Tensor]:
    deploy_default = torch.tensor(deploy_cfg["default_joint_pos"], dtype=torch.float32, device=device)
    isaac_default = robot.data.default_joint_pos[0].detach().to(device=device, dtype=torch.float32)
    joint_ids_map = [int(x) for x in deploy_cfg["joint_ids_map"]]
    if torch.allclose(isaac_default, deploy_default, atol=1e-5, rtol=1e-5):
        return JointOrderInfo(False, list(range(robot.num_joints)), list(range(robot.num_joints))), deploy_default
    try_policy_default = isaac_default[torch.tensor(joint_ids_map, dtype=torch.long, device=device)]
    if torch.allclose(try_policy_default, deploy_default, atol=1e-5, rtol=1e-5):
        policy_to_sim = list(joint_ids_map)
        sim_to_policy = [0] * len(policy_to_sim)
        for policy_index, sim_index in enumerate(policy_to_sim):
            sim_to_policy[sim_index] = policy_index
        return JointOrderInfo(True, sim_to_policy, policy_to_sim), deploy_default
    return JointOrderInfo(False, list(range(robot.num_joints)), list(range(robot.num_joints))), deploy_default


def reorder_sim_to_policy(tensor: torch.Tensor, order_info: JointOrderInfo) -> torch.Tensor:
    if not order_info.need_reorder:
        return tensor
    return tensor[:, order_info.policy_to_sim]


def create_obs_states(deploy_cfg: dict, device: torch.device) -> tuple[dict[str, ObservationTermState], bool]:
    obs_cfg = deploy_cfg["observations"]
    scale_first = bool(obs_cfg.get("scale_first", False))
    use_gym_history = bool(obs_cfg.get("use_gym_history", False))
    states: dict[str, ObservationTermState] = {}
    for term_name in POLICY_OBS_TERM_ORDER:
        term_cfg = dict(obs_cfg[term_name])
        term_cfg["scale_first"] = term_cfg.get("scale_first", scale_first)
        states[term_name] = make_term_state(term_name, term_cfg, device)
    return states, use_gym_history


def reset_obs_history(obs_states: dict[str, ObservationTermState], raw_terms: dict[str, torch.Tensor]) -> None:
    for term_name, term_state in obs_states.items():
        processed = apply_obs_postprocess(raw_terms[term_name], term_state)
        term_state.history.clear()
        for _ in range(term_state.history_length):
            term_state.history.append(processed.clone())


def flatten_obs(obs_states: dict[str, ObservationTermState], use_gym_history: bool) -> torch.Tensor:
    if use_gym_history:
        history_length = obs_states[POLICY_OBS_TERM_ORDER[0]].history_length
        frames = []
        for history_index in range(history_length):
            frame = [obs_states[term_name].history[history_index] for term_name in POLICY_OBS_TERM_ORDER]
            frames.append(torch.cat(frame, dim=-1))
        return torch.cat(frames, dim=-1)
    return torch.cat(
        [torch.cat(list(obs_states[term_name].history), dim=-1) for term_name in POLICY_OBS_TERM_ORDER],
        dim=-1,
    )


def extract_raw_terms(
    env,
    robot,
    deploy_cfg: dict,
    order_info: JointOrderInfo,
    deploy_default_joint_pos: torch.Tensor,
    last_action_policy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cmd = get_live_base_velocity_command(env, deploy_cfg)
    return {
        "base_ang_vel": robot.data.root_ang_vel_b.clone(),
        "projected_gravity": robot.data.projected_gravity_b.clone(),
        "velocity_commands": cmd.clone(),
        "joint_pos_rel": reorder_sim_to_policy(robot.data.joint_pos, order_info) - deploy_default_joint_pos.view(1, -1),
        "joint_vel_rel": reorder_sim_to_policy(robot.data.joint_vel, order_info).clone(),
        "last_action": last_action_policy.clone(),
    }


def print_vector_preview(name: str, tensor: torch.Tensor, count: int = 30) -> None:
    flat = tensor[0].detach().cpu().numpy()
    preview = np.array2string(flat[:count], precision=5, separator=", ")
    print(f"{name} first_{count}: {preview}")


def print_term_breakdown(raw_terms: dict[str, torch.Tensor], obs_states: dict[str, ObservationTermState]) -> None:
    for term_name in POLICY_OBS_TERM_ORDER:
        processed = apply_obs_postprocess(raw_terms[term_name], obs_states[term_name])
        print(
            f"{term_name}: shape={list(processed.shape)} "
            f"min={processed.min().item(): .5f} max={processed.max().item(): .5f} "
            f"history_length={obs_states[term_name].history_length}"
        )


def print_segment_comparison(
    policy_obs: torch.Tensor, deploy_obs: torch.Tensor, raw_terms: dict[str, torch.Tensor], obs_states: dict[str, ObservationTermState]
) -> None:
    print("=== deploy-layout segment comparison ===")
    start = 0
    for term_name in POLICY_OBS_TERM_ORDER:
        processed = apply_obs_postprocess(raw_terms[term_name], obs_states[term_name])
        segment_dim = processed.shape[-1] * obs_states[term_name].history_length
        policy_segment = policy_obs[:, start : start + segment_dim]
        deploy_segment = deploy_obs[:, start : start + segment_dim]
        print(
            f"{term_name}: slice=[{start}:{start + segment_dim}] "
            f"policy[min={policy_segment.min().item(): .5f}, max={policy_segment.max().item(): .5f}] "
            f"deploy[min={deploy_segment.min().item(): .5f}, max={deploy_segment.max().item(): .5f}] "
            f"l2={torch.linalg.norm(policy_segment - deploy_segment).item(): .5f}"
        )
        start += segment_dim


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError("--num_envs must be 1 for this comparison script.")

    deploy_cfg = load_deploy_cfg()
    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
        entry_point_key="env_cfg_entry_point",
    )
    configure_env(env_cfg, deploy_cfg)
    env = gym.make(TASK_NAME, cfg=env_cfg)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    order_info, deploy_default_joint_pos = build_joint_order_info(robot, deploy_cfg, device)
    obs_states, use_gym_history = create_obs_states(deploy_cfg, device)
    fixed_command = clamp_velocity_command(
        torch.tensor([[args_cli.vx, args_cli.vy, args_cli.wz]], dtype=torch.float32, device=device), deploy_cfg
    )
    zero_action = torch.zeros((1, robot.num_joints), dtype=torch.float32, device=device)
    live_command = enforce_fixed_base_velocity_command(env, deploy_cfg, fixed_command)
    if not torch.allclose(live_command, fixed_command, atol=1e-5, rtol=1e-5):
        print("[command][warn] requested:", fixed_command.detach().cpu().numpy().tolist())
        print("[command][warn] live:", live_command.detach().cpu().numpy().tolist())

    for _ in range(args_cli.warmup_steps):
        obs, _, _, _, _ = env.step(zero_action)
        live_command = env.unwrapped.command_manager.get_command("base_velocity").clone()
        if not torch.allclose(live_command, fixed_command, atol=1e-5, rtol=1e-5):
            enforce_fixed_base_velocity_command(env, deploy_cfg, fixed_command)

    raw_terms = extract_raw_terms(
        env=env,
        robot=robot,
        deploy_cfg=deploy_cfg,
        order_info=order_info,
        deploy_default_joint_pos=deploy_default_joint_pos,
        last_action_policy=zero_action,
    )
    reset_obs_history(obs_states, raw_terms)
    deploy_obs = flatten_obs(obs_states, use_gym_history)
    policy_obs = obs["policy"]

    print("=== Comparison ===")
    print("isaac obs[policy] shape:", list(policy_obs.shape))
    print("manual deploy obs shape:", list(deploy_obs.shape))
    print("L2 difference:", torch.linalg.norm(policy_obs - deploy_obs).item())
    print("max abs difference:", torch.max(torch.abs(policy_obs - deploy_obs)).item())

    print("=== obs[policy] stats ===")
    print("min:", policy_obs.min().item(), "max:", policy_obs.max().item())
    print_vector_preview("obs[policy]", policy_obs)

    print("=== manual deploy obs stats ===")
    print("min:", deploy_obs.min().item(), "max:", deploy_obs.max().item())
    print_vector_preview("deploy_obs", deploy_obs)

    print("=== manual deploy term breakdown ===")
    print_term_breakdown(raw_terms, obs_states)
    print_segment_comparison(policy_obs, deploy_obs, raw_terms, obs_states)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
