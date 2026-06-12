import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play the official G1 29dof deploy ONNX policy in Isaac Lab.")
parser.add_argument("--vx", type=float, default=0.0, help="Fixed x velocity command in policy command space.")
parser.add_argument("--vy", type=float, default=0.0, help="Fixed y velocity command in policy command space.")
parser.add_argument("--wz", type=float, default=0.0, help="Fixed yaw velocity command in policy command space.")
parser.add_argument("--num_envs", type=int, default=1, help="Only 1 is supported because ONNX input is [1, 480].")
parser.add_argument("--warmup_steps", type=int, default=150, help="Number of zero-action/default-pose warmup steps.")
parser.add_argument("--action_clip", type=float, default=0.0, help="Clip raw ONNX action to [-x, x]. 0 disables.")
parser.add_argument(
    "--direct_joint_target",
    action="store_true",
    help="Bypass Isaac Lab action manager and send processed joint position targets directly.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch
import yaml

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "Unitree-G1-29dof-Velocity"
DEPLOY_YAML_PATH = REPO_ROOT / "deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml"
POLICY_PATH = REPO_ROOT / "deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"
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
    reason: str


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


def make_term_state(name: str, cfg: dict, device: torch.device, num_envs: int) -> ObservationTermState:
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

    print("=== Isaac Lab Joint Names ===")
    for index, name in enumerate(robot.joint_names):
        print(f"{index:02d}: {name}")

    print("=== default_joint_pos comparison ===")
    print("isaac_default_joint_pos:", isaac_default.detach().cpu().numpy().tolist())
    print("deploy_default_joint_pos:", deploy_default.detach().cpu().numpy().tolist())

    if torch.allclose(isaac_default, deploy_default, atol=1e-5, rtol=1e-5):
        info = JointOrderInfo(
            need_reorder=False,
            sim_to_policy=list(range(robot.num_joints)),
            policy_to_sim=list(range(robot.num_joints)),
            reason="Isaac Lab default_joint_pos already matches deploy policy order.",
        )
        print("[joint-order]", info.reason)
        return info, deploy_default

    try_policy_default = isaac_default[torch.tensor(joint_ids_map, dtype=torch.long, device=device)]
    if torch.allclose(try_policy_default, deploy_default, atol=1e-5, rtol=1e-5):
        policy_to_sim = list(joint_ids_map)
        sim_to_policy = [0] * len(policy_to_sim)
        for policy_index, sim_index in enumerate(policy_to_sim):
            sim_to_policy[sim_index] = policy_index
        info = JointOrderInfo(
            need_reorder=True,
            sim_to_policy=sim_to_policy,
            policy_to_sim=policy_to_sim,
            reason="Isaac Lab default_joint_pos matches deploy default only after applying joint_ids_map.",
        )
        print("[joint-order]", info.reason)
        print("[joint-order] policy_to_sim:", policy_to_sim)
        print("[joint-order] sim_to_policy:", sim_to_policy)
        return info, deploy_default

    info = JointOrderInfo(
        need_reorder=False,
        sim_to_policy=list(range(robot.num_joints)),
        policy_to_sim=list(range(robot.num_joints)),
        reason="No verified joint reordering found. Keeping Isaac Lab order to avoid incorrect remapping.",
    )
    print("[joint-order][warn]", info.reason)
    print("[joint-order][warn] isaac[joint_ids_map]:", try_policy_default.detach().cpu().numpy().tolist())
    return info, deploy_default


def reorder_sim_to_policy(tensor: torch.Tensor, order_info: JointOrderInfo) -> torch.Tensor:
    if not order_info.need_reorder:
        return tensor
    return tensor[:, order_info.policy_to_sim]


def reorder_policy_to_sim(tensor: torch.Tensor, order_info: JointOrderInfo) -> torch.Tensor:
    if not order_info.need_reorder:
        return tensor
    return tensor[:, order_info.sim_to_policy]


def extract_raw_terms(
    env,
    robot,
    deploy_cfg: dict,
    order_info: JointOrderInfo,
    deploy_default_joint_pos: torch.Tensor,
    last_action_policy: torch.Tensor,
) -> dict[str, torch.Tensor]:
    cmd = get_live_base_velocity_command(env, deploy_cfg)
    joint_pos_policy = reorder_sim_to_policy(robot.data.joint_pos, order_info)
    joint_vel_policy = reorder_sim_to_policy(robot.data.joint_vel, order_info)
    raw_terms = {
        "base_ang_vel": robot.data.root_ang_vel_b.clone(),
        "projected_gravity": robot.data.projected_gravity_b.clone(),
        "velocity_commands": cmd.clone(),
        "joint_pos_rel": joint_pos_policy - deploy_default_joint_pos.view(1, -1),
        "joint_vel_rel": joint_vel_policy.clone(),
        "last_action": last_action_policy.clone(),
    }
    return raw_terms


def create_obs_states(deploy_cfg: dict, device: torch.device, num_envs: int) -> tuple[dict[str, ObservationTermState], bool]:
    obs_cfg = deploy_cfg["observations"]
    scale_first = bool(obs_cfg.get("scale_first", False))
    use_gym_history = bool(obs_cfg.get("use_gym_history", False))
    states: dict[str, ObservationTermState] = {}
    for term_name in POLICY_OBS_TERM_ORDER:
        term_cfg = dict(obs_cfg[term_name])
        term_cfg["scale_first"] = term_cfg.get("scale_first", scale_first)
        states[term_name] = make_term_state(term_name, term_cfg, device, num_envs)
    return states, use_gym_history


def reset_obs_history(obs_states: dict[str, ObservationTermState], raw_terms: dict[str, torch.Tensor]) -> None:
    for term_name, term_state in obs_states.items():
        processed = apply_obs_postprocess(raw_terms[term_name], term_state)
        term_state.history.clear()
        for _ in range(term_state.history_length):
            term_state.history.append(processed.clone())


def update_obs_history(obs_states: dict[str, ObservationTermState], raw_terms: dict[str, torch.Tensor]) -> None:
    for term_name, term_state in obs_states.items():
        processed = apply_obs_postprocess(raw_terms[term_name], term_state)
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


def print_term_stats(prefix: str, raw_terms: dict[str, torch.Tensor], obs_states: dict[str, ObservationTermState]) -> None:
    print(prefix)
    for term_name in POLICY_OBS_TERM_ORDER:
        raw = raw_terms[term_name]
        processed = apply_obs_postprocess(raw, obs_states[term_name])
        print(
            f"  {term_name}: raw[min={raw.min().item(): .5f}, max={raw.max().item(): .5f}] "
            f"processed[min={processed.min().item(): .5f}, max={processed.max().item(): .5f}]"
        )


def load_onnx_session() -> tuple[ort.InferenceSession, str, str]:
    session = ort.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    print("=== ONNX IO ===")
    print("input:", input_info.name, input_info.shape)
    print("output:", output_info.name, output_info.shape)
    if input_info.shape != [1, 480]:
        raise RuntimeError(f"Expected ONNX input shape [1, 480], got {input_info.shape}")
    if output_info.shape != [1, 29]:
        raise RuntimeError(f"Expected ONNX output shape [1, 29], got {output_info.shape}")
    return session, input_info.name, output_info.name


def manual_direct_step(env, target_joint_pos: torch.Tensor):
    unwrapped = env.unwrapped
    is_rendering = unwrapped.sim.has_gui() or unwrapped.sim.has_rtx_sensors()
    for _ in range(unwrapped.cfg.decimation):
        unwrapped._sim_step_counter += 1
        unwrapped.scene["robot"].set_joint_position_target(target_joint_pos)
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.step(render=False)
        if unwrapped._sim_step_counter % unwrapped.cfg.sim.render_interval == 0 and is_rendering:
            unwrapped.sim.render()
        unwrapped.scene.update(dt=unwrapped.physics_dt)

    unwrapped.episode_length_buf += 1
    unwrapped.common_step_counter += 1
    reset_buf = unwrapped.termination_manager.compute()
    terminated = unwrapped.termination_manager.terminated.clone()
    truncated = unwrapped.termination_manager.time_outs.clone()
    if hasattr(unwrapped, "command_manager"):
        unwrapped.command_manager.compute(dt=unwrapped.step_dt)
    if "interval" in unwrapped.event_manager.available_modes:
        unwrapped.event_manager.apply(mode="interval", dt=unwrapped.step_dt)
    return terminated, truncated, reset_buf


def reset_and_warmup(
    env,
    deploy_cfg: dict,
    obs_states: dict[str, ObservationTermState],
    use_gym_history: bool,
    order_info: JointOrderInfo,
    deploy_default_joint_pos: torch.Tensor,
    fixed_command_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    env.reset()
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    zero_raw_action_policy = torch.zeros((1, robot.num_joints), dtype=torch.float32, device=device)
    zero_raw_action_sim = reorder_policy_to_sim(zero_raw_action_policy, order_info)
    direct_scale = torch.tensor(
        deploy_cfg["actions"]["JointPositionAction"]["scale"], dtype=torch.float32, device=device
    ).view(1, -1)
    direct_offset = torch.tensor(
        deploy_cfg["actions"]["JointPositionAction"]["offset"], dtype=torch.float32, device=device
    ).view(1, -1)
    live_command = enforce_fixed_base_velocity_command(env, deploy_cfg, fixed_command_target)
    if not torch.allclose(live_command, fixed_command_target, atol=1e-5, rtol=1e-5):
        print("[command][warn] command after reset differs from requested target.")
        print("[command][warn] requested:", fixed_command_target.detach().cpu().numpy().tolist())
        print("[command][warn] live:", live_command.detach().cpu().numpy().tolist())

    for _ in range(args_cli.warmup_steps):
        if args_cli.direct_joint_target:
            target_joint_pos_policy = zero_raw_action_policy * direct_scale + direct_offset
            target_joint_pos_sim = reorder_policy_to_sim(target_joint_pos_policy, order_info)
            manual_direct_step(env, target_joint_pos_sim)
        else:
            env.step(zero_raw_action_sim)
        live_command = env.unwrapped.command_manager.get_command("base_velocity").clone()
        if not torch.allclose(live_command, fixed_command_target, atol=1e-5, rtol=1e-5):
            enforce_fixed_base_velocity_command(env, deploy_cfg, fixed_command_target)

    last_action_policy = zero_raw_action_policy.clone()
    raw_terms = extract_raw_terms(
        env=env,
        robot=robot,
        deploy_cfg=deploy_cfg,
        order_info=order_info,
        deploy_default_joint_pos=deploy_default_joint_pos,
        last_action_policy=last_action_policy,
    )
    reset_obs_history(obs_states, raw_terms)
    obs = flatten_obs(obs_states, use_gym_history)
    return last_action_policy, raw_terms, obs


def main() -> None:
    if args_cli.num_envs != 1:
        raise ValueError(f"--num_envs must be 1 because the official ONNX input shape is [1, 480], got {args_cli.num_envs}")

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
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    onnx_session, input_name, output_name = load_onnx_session()
    order_info, deploy_default_joint_pos = build_joint_order_info(robot, deploy_cfg, device)
    obs_states, use_gym_history = create_obs_states(deploy_cfg, device, 1)

    fixed_command = torch.tensor([[args_cli.vx, args_cli.vy, args_cli.wz]], dtype=torch.float32, device=device)
    fixed_command = clamp_velocity_command(fixed_command, deploy_cfg)
    action_scale = torch.tensor(
        deploy_cfg["actions"]["JointPositionAction"]["scale"], dtype=torch.float32, device=device
    ).view(1, -1)
    action_offset = torch.tensor(
        deploy_cfg["actions"]["JointPositionAction"]["offset"], dtype=torch.float32, device=device
    ).view(1, -1)

    print("=== Run Mode ===")
    print("mode:", "direct_joint_target" if args_cli.direct_joint_target else "env.step(raw_action)")
    print("use_gym_history:", use_gym_history)
    print("fixed_command:", fixed_command.detach().cpu().numpy().tolist())
    print("warmup_steps:", args_cli.warmup_steps)
    print("action_clip:", args_cli.action_clip)

    last_action_policy, raw_terms, obs = reset_and_warmup(
        env=env,
        deploy_cfg=deploy_cfg,
        obs_states=obs_states,
        use_gym_history=use_gym_history,
        order_info=order_info,
        deploy_default_joint_pos=deploy_default_joint_pos,
        fixed_command_target=fixed_command,
    )

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs_np = obs.detach().cpu().numpy().astype(np.float32)
            raw_action_policy = torch.from_numpy(
                onnx_session.run([output_name], {input_name: obs_np})[0]
            ).to(device=device, dtype=torch.float32)

            if args_cli.action_clip > 0.0:
                clipped_action_policy = raw_action_policy.clamp(-args_cli.action_clip, args_cli.action_clip)
                action_was_clipped = not torch.allclose(clipped_action_policy, raw_action_policy)
                raw_action_policy = clipped_action_policy
            else:
                action_was_clipped = False

            if torch.isnan(raw_action_policy).any() or torch.isinf(raw_action_policy).any() or raw_action_policy.abs().max().item() > 20.0:
                print(f"[step {step_count}] action explosion detected, resetting.")
                print_term_stats("obs term stats before reset:", raw_terms, obs_states)
                last_action_policy, raw_terms, obs = reset_and_warmup(
                    env=env,
                    deploy_cfg=deploy_cfg,
                    obs_states=obs_states,
                    use_gym_history=use_gym_history,
                    order_info=order_info,
                    deploy_default_joint_pos=deploy_default_joint_pos,
                    fixed_command_target=fixed_command,
                )
                step_count = 0
                continue

            raw_action_sim = reorder_policy_to_sim(raw_action_policy, order_info)
            if args_cli.direct_joint_target:
                target_joint_pos_policy = raw_action_policy * action_scale + action_offset
                target_joint_pos_sim = reorder_policy_to_sim(target_joint_pos_policy, order_info)
                terminated, truncated, _ = manual_direct_step(env, target_joint_pos_sim)
            else:
                _, _, terminated, truncated, _ = env.step(raw_action_sim)

            live_command = env.unwrapped.command_manager.get_command("base_velocity").clone()
            if not torch.allclose(live_command, fixed_command, atol=1e-5, rtol=1e-5):
                print(
                    f"[step {step_count}] [command][warn] requested={fixed_command[0].detach().cpu().numpy().tolist()} "
                    f"live={live_command[0].detach().cpu().numpy().tolist()}"
                )
                enforce_fixed_base_velocity_command(env, deploy_cfg, fixed_command)

            last_action_policy = raw_action_policy.clone()
            raw_terms = extract_raw_terms(
                env=env,
                robot=robot,
                deploy_cfg=deploy_cfg,
                order_info=order_info,
                deploy_default_joint_pos=deploy_default_joint_pos,
                last_action_policy=last_action_policy,
            )
            update_obs_history(obs_states, raw_terms)
            obs = flatten_obs(obs_states, use_gym_history)
            step_count += 1

            terminated_flag = bool(terminated[0].item()) if terminated.numel() > 0 else False
            truncated_flag = bool(truncated[0].item()) if truncated.numel() > 0 else False

            if step_count % 100 == 0:
                root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
                root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
                projected_gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy()
                base_ang_vel = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
                print(
                    f"[step {step_count}] "
                    f"root_pos={root_pos.tolist()} "
                    f"root_quat={root_quat.tolist()} "
                    f"projected_gravity={projected_gravity.tolist()} "
                    f"base_ang_vel={base_ang_vel.tolist()} "
                    f"command={env.unwrapped.command_manager.get_command('base_velocity')[0].detach().cpu().numpy().tolist()} "
                    f"obs_min={obs.min().item(): .5f} obs_max={obs.max().item(): .5f} "
                    f"action_min={raw_action_policy.min().item(): .5f} action_max={raw_action_policy.max().item(): .5f} "
                    f"action_clipped={action_was_clipped} "
                    f"root_height={root_pos[2]: .5f} "
                    f"terminated={terminated_flag} truncated={truncated_flag}"
                )

            if terminated_flag or truncated_flag:
                print(f"[step {step_count}] environment terminated={terminated_flag} truncated={truncated_flag}, resetting.")
                last_action_policy, raw_terms, obs = reset_and_warmup(
                    env=env,
                    deploy_cfg=deploy_cfg,
                    obs_states=obs_states,
                    use_gym_history=use_gym_history,
                    order_info=order_info,
                    deploy_default_joint_pos=deploy_default_joint_pos,
                    fixed_command_target=fixed_command,
                )
                step_count = 0

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
