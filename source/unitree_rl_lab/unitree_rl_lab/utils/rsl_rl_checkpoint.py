from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


CHECKPOINT_INFO_KEY = "unitree_rl_lab_resume_state"
CHECKPOINT_STATE_VERSION = 1


def _base_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _cpu_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    return value


def _to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    return value


def _copy_tensor_attr(obj: Any, attr_name: str, value: torch.Tensor, device: torch.device | str) -> bool:
    if not hasattr(obj, attr_name):
        return False
    current = getattr(obj, attr_name)
    value = value.to(device=device)
    if torch.is_tensor(current) and current.shape == value.shape:
        current.copy_(value)
    else:
        setattr(obj, attr_name, value)
    return True


def _scene_entity(env: Any, name: str) -> Any | None:
    scene = getattr(env, "scene", None)
    if scene is None:
        return None
    try:
        return scene[name]
    except Exception:
        return None


def _command_terms(env: Any) -> Mapping[str, Any]:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None:
        return {}
    terms = getattr(command_manager, "_terms", None)
    if isinstance(terms, Mapping):
        return terms
    active_terms = getattr(command_manager, "active_terms", ())
    result = {}
    for name in active_terms:
        try:
            result[name] = command_manager.get_term(name)
        except Exception:
            continue
    return result


def collect_resume_state(env: Any) -> dict[str, Any]:
    """Collect training state that RSL-RL does not store in its checkpoint.

    RSL-RL checkpoints already contain the policy, optimizer, and iteration. This
    payload stores the Isaac Lab side of training that controls ULC curriculum
    progression and makes a resumed run behave like a continuation of the same
    training run rather than a fresh environment with an old policy.
    """

    env = _base_env(env)
    device = getattr(env, "device", "cpu")
    state: dict[str, Any] = {
        "version": CHECKPOINT_STATE_VERSION,
        "device": str(device),
        "common_step_counter": int(getattr(env, "common_step_counter", 0)),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
        "commands": {},
        "buffers": {},
        "action_manager": {},
        "reward_manager": {},
        "articulations": {},
    }
    if torch.cuda.is_available():
        state["rng"]["torch_cuda_all"] = torch.cuda.get_rng_state_all()

    for attr_name in ("episode_length_buf", "reset_buf", "reset_terminated", "reset_time_out"):
        if hasattr(env, attr_name):
            value = getattr(env, attr_name)
            if torch.is_tensor(value):
                state["buffers"][attr_name] = _cpu_clone(value)

    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        for attr_name in ("action", "prev_action"):
            if hasattr(action_manager, attr_name):
                value = getattr(action_manager, attr_name)
                if torch.is_tensor(value):
                    state["action_manager"][attr_name] = _cpu_clone(value)

    reward_manager = getattr(env, "reward_manager", None)
    episode_sums = getattr(reward_manager, "_episode_sums", None)
    if isinstance(episode_sums, Mapping):
        state["reward_manager"]["episode_sums"] = {
            name: _cpu_clone(value) for name, value in episode_sums.items() if torch.is_tensor(value)
        }

    for name, term in _command_terms(env).items():
        term_state: dict[str, Any] = {}
        for attr_name in ("alpha_height", "alpha_upper", "_last_curriculum_step"):
            if hasattr(term, attr_name):
                value = getattr(term, attr_name)
                term_state[attr_name] = _cpu_clone(value)
        if hasattr(term, "command_tensor") and torch.is_tensor(term.command_tensor):
            term_state["command_tensor"] = _cpu_clone(term.command_tensor)
        if term_state:
            state["commands"][name] = term_state

    robot = _scene_entity(env, "robot")
    if robot is not None and hasattr(robot, "data"):
        robot_state: dict[str, Any] = {}
        data = robot.data
        for attr_name in (
            "root_state_w",
            "joint_pos",
            "joint_vel",
            "joint_pos_target",
            "joint_vel_target",
            "joint_effort_target",
        ):
            if hasattr(data, attr_name):
                value = getattr(data, attr_name)
                if torch.is_tensor(value):
                    robot_state[attr_name] = _cpu_clone(value)
        if robot_state:
            state["articulations"]["robot"] = robot_state

    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    rng_state = state.get("rng", {})
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and "torch_cuda_all" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda_all"])


def restore_resume_state(env: Any, state: Mapping[str, Any] | None) -> bool:
    """Restore the Isaac Lab side of a checkpoint.

    Returns ``True`` when a compatible state payload was found and applied.
    """

    if not isinstance(state, Mapping):
        return False
    if int(state.get("version", -1)) != CHECKPOINT_STATE_VERSION:
        return False

    env = _base_env(env)
    device = getattr(env, "device", "cpu")

    if "common_step_counter" in state:
        env.common_step_counter = int(state["common_step_counter"])

    for attr_name, value in state.get("buffers", {}).items():
        if torch.is_tensor(value):
            _copy_tensor_attr(env, attr_name, value, device)

    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        for attr_name, value in state.get("action_manager", {}).items():
            if torch.is_tensor(value):
                _copy_tensor_attr(action_manager, attr_name, value, device)

    reward_manager = getattr(env, "reward_manager", None)
    episode_sums = getattr(reward_manager, "_episode_sums", None)
    saved_episode_sums = state.get("reward_manager", {}).get("episode_sums", {})
    if isinstance(episode_sums, Mapping) and isinstance(saved_episode_sums, Mapping):
        for name, value in saved_episode_sums.items():
            if name in episode_sums and torch.is_tensor(value) and torch.is_tensor(episode_sums[name]):
                if episode_sums[name].shape == value.shape:
                    episode_sums[name].copy_(value.to(device=device))

    command_terms = _command_terms(env)
    for name, term_state in state.get("commands", {}).items():
        if name not in command_terms or not isinstance(term_state, Mapping):
            continue
        term = command_terms[name]
        for attr_name in ("alpha_height", "alpha_upper", "_last_curriculum_step"):
            if attr_name in term_state:
                setattr(term, attr_name, _to_device(term_state[attr_name], device))
        if "command_tensor" in term_state and hasattr(term, "command_tensor"):
            value = term_state["command_tensor"]
            if torch.is_tensor(value) and torch.is_tensor(term.command_tensor) and term.command_tensor.shape == value.shape:
                term.command_tensor.copy_(value.to(device=device))
        metrics = getattr(term, "metrics", None)
        if isinstance(metrics, Mapping):
            if "alpha_height" in metrics and hasattr(term, "alpha_height"):
                metrics["alpha_height"][:] = float(term.alpha_height)
            if "alpha_upper" in metrics and hasattr(term, "alpha_upper"):
                metrics["alpha_upper"][:] = float(term.alpha_upper)

    robot_state = state.get("articulations", {}).get("robot", {})
    robot = _scene_entity(env, "robot")
    if robot is not None and isinstance(robot_state, Mapping):
        root_state = robot_state.get("root_state_w")
        if torch.is_tensor(root_state) and hasattr(robot, "write_root_state_to_sim"):
            robot.write_root_state_to_sim(root_state.to(device=device))
        joint_pos = robot_state.get("joint_pos")
        joint_vel = robot_state.get("joint_vel")
        if torch.is_tensor(joint_pos) and torch.is_tensor(joint_vel) and hasattr(robot, "write_joint_state_to_sim"):
            robot.write_joint_state_to_sim(joint_pos.to(device=device), joint_vel.to(device=device))
        for attr_name in ("joint_pos_target", "joint_vel_target", "joint_effort_target"):
            value = robot_state.get(attr_name)
            data = getattr(robot, "data", None)
            if torch.is_tensor(value) and data is not None and hasattr(data, attr_name):
                current = getattr(data, attr_name)
                if torch.is_tensor(current) and current.shape == value.shape:
                    current.copy_(value.to(device=device))

    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim, "forward"):
        sim.forward()

    _restore_rng_state(state)
    return True


def _with_resume_state(infos: dict[str, Any] | None, env: Any) -> dict[str, Any]:
    result = {} if infos is None else dict(infos)
    result[CHECKPOINT_INFO_KEY] = collect_resume_state(env)
    return result


def attach_resumable_checkpointing(runner: Any) -> None:
    """Patch an RSL-RL runner so every ``model_*.pt`` stores ULC resume state."""

    if getattr(runner, "_unitree_resumable_checkpointing", False):
        return

    original_save = runner.save
    original_load = runner.load

    def save_with_resume_state(path: str, infos: dict[str, Any] | None = None) -> None:
        original_save(path, infos=_with_resume_state(infos, runner.env))

    def load_with_resume_state(path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = original_load(path, load_optimizer=load_optimizer, map_location=map_location)
        state = infos.get(CHECKPOINT_INFO_KEY) if isinstance(infos, Mapping) else None
        restored = restore_resume_state(runner.env, state)
        if restored:
            print("[INFO]: Restored Unitree RL Lab environment resume state from checkpoint.")
        else:
            print("[WARNING]: Checkpoint has no Unitree RL Lab environment resume state; restored model only.")
        return infos

    runner.save = save_with_resume_state
    runner.load = load_with_resume_state
    runner._unitree_resumable_checkpointing = True
