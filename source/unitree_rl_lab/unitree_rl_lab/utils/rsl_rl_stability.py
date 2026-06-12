import functools
import math
from typing import Any

import torch


STABLE_VELOCITY_TASK = "Unitree-G1-29dof-StableVelocity"
STABLE_VELOCITY_STD_MIN = 1.0e-4


class StableVelocityNumericalStabilityError(RuntimeError):
    """Fatal numerical stability error raised for StableVelocity PPO training."""


def is_stable_velocity_task(task_name: str | None) -> bool:
    return task_name == STABLE_VELOCITY_TASK


def std_positive_guard_enabled(policy_cfg: Any) -> bool:
    return getattr(policy_cfg, "noise_std_type", "scalar") == "log"


def finite_guard_enabled(task_name: str | None) -> bool:
    return is_stable_velocity_task(task_name)


def load_runner_checkpoint_compat(runner: Any, checkpoint_path: str) -> None:
    """Load an RSL-RL runner checkpoint with std/log_std compatibility for policy noise."""
    policy_module = getattr(getattr(runner, "alg", None), "policy", None)
    if policy_module is None:
        policy_module = getattr(getattr(runner, "alg", None), "actor_critic", None)
    if policy_module is None or not hasattr(policy_module, "load_state_dict"):
        runner.load(checkpoint_path)
        return

    original_load_state_dict = policy_module.load_state_dict

    @functools.wraps(original_load_state_dict)
    def compatible_load_state_dict(state_dict, strict: bool = True):
        adapted_state_dict = adapt_policy_noise_state_dict(policy_module, state_dict)
        return original_load_state_dict(adapted_state_dict, strict=strict)

    policy_module.load_state_dict = compatible_load_state_dict
    try:
        runner.load(checkpoint_path)
    finally:
        policy_module.load_state_dict = original_load_state_dict


def adapt_policy_noise_state_dict(policy_module: Any, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Adapt `std` <-> `log_std` keys in a policy state dict to match the current module."""
    if not isinstance(state_dict, dict):
        return state_dict

    has_model_log_std = hasattr(policy_module, "log_std")
    has_model_std = hasattr(policy_module, "std")
    has_ckpt_log_std = "log_std" in state_dict
    has_ckpt_std = "std" in state_dict

    if has_model_log_std and has_ckpt_std and not has_ckpt_log_std:
        adapted_state_dict = dict(state_dict)
        std = adapted_state_dict.pop("std")
        adapted_state_dict["log_std"] = torch.log(torch.clamp(std, min=STABLE_VELOCITY_STD_MIN))
        print("[INFO][StableVelocity] adapting checkpoint noise parameter: std -> log_std")
        return adapted_state_dict

    if has_model_std and has_ckpt_log_std and not has_ckpt_std:
        adapted_state_dict = dict(state_dict)
        log_std = adapted_state_dict.pop("log_std")
        adapted_state_dict["std"] = torch.exp(log_std).clamp(min=STABLE_VELOCITY_STD_MIN)
        print("[INFO][StableVelocity] adapting checkpoint noise parameter: log_std -> std")
        return adapted_state_dict

    return state_dict


def install_stable_velocity_runtime_guards(
    task_name: str | None, env: Any, runner: Any, policy_cfg: Any | None = None
) -> dict[str, Any]:
    """Install runtime finite/std guards for the StableVelocity PPO training path."""
    status = {
        "enabled": False,
        "std_min": STABLE_VELOCITY_STD_MIN,
        "std_positive_guard": False,
        "finite_guard": False,
    }
    if not is_stable_velocity_task(task_name):
        return status

    alg = getattr(runner, "alg", None)
    actor_critic = getattr(alg, "actor_critic", None)
    policy_cfg = policy_cfg if policy_cfg is not None else getattr(runner, "policy_cfg", None)
    noise_std_type = getattr(policy_cfg, "noise_std_type", "scalar")

    status["enabled"] = True
    status["std_positive_guard"] = std_positive_guard_enabled(policy_cfg)
    status["finite_guard"] = True

    if actor_critic is not None:
        _clamp_std_parameters(actor_critic, noise_std_type, STABLE_VELOCITY_STD_MIN)
        _check_module_parameters_finite(actor_critic, "actor_critic")
        _install_actor_critic_guards(actor_critic, noise_std_type, STABLE_VELOCITY_STD_MIN)

    _install_env_guards(env)
    _install_algorithm_guards(alg, actor_critic, noise_std_type, STABLE_VELOCITY_STD_MIN)
    return status


def _fatal(message: str) -> None:
    raise StableVelocityNumericalStabilityError(f"[FATAL][StableVelocity] {message}")


def _sample_bad_values(value: torch.Tensor) -> list[float]:
    bad_values = value[~torch.isfinite(value)].detach().flatten().cpu()
    return bad_values[:8].tolist()


def _check_scalar_finite(name: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (float, int)):
        if not math.isfinite(float(value)):
            _fatal(f"non-finite value detected: {name}={value}")


def _check_tensor_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        _fatal(f"non-finite tensor detected: {name}, sample={_sample_bad_values(value)}")


def _check_nested_finite(name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        _check_tensor_finite(name, value)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_nested_finite(f"{name}.{key}", item)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_nested_finite(f"{name}[{index}]", item)
        return
    _check_scalar_finite(name, value)


def _check_module_parameters_finite(module: Any, module_name: str) -> None:
    if module is None or not hasattr(module, "named_parameters"):
        return
    for name, param in module.named_parameters():
        _check_tensor_finite(f"{module_name}.{name}", param)


def _clamp_std_parameters(actor_critic: Any, noise_std_type: str, std_min: float) -> bool:
    clamped = False
    log_std_min = math.log(std_min)
    for attr_name in ("std", "log_std"):
        value = getattr(actor_critic, attr_name, None)
        if isinstance(value, torch.Tensor):
            with torch.no_grad():
                if attr_name == "log_std" or noise_std_type == "log":
                    value.clamp_(min=log_std_min)
                else:
                    value.clamp_(min=std_min)
            clamped = True
    return clamped


def _check_distribution_std(actor_critic: Any, noise_std_type: str, std_min: float) -> None:
    _clamp_std_parameters(actor_critic, noise_std_type, std_min)
    distribution = getattr(actor_critic, "distribution", None)
    if distribution is None or not hasattr(distribution, "stddev"):
        return
    stddev = distribution.stddev
    _check_tensor_finite("policy.distribution.stddev", stddev)
    stddev_min = float(stddev.detach().min().item())
    if stddev_min < std_min:
        _fatal(f"policy std below minimum: min_std={stddev_min:.6e}, required>={std_min:.6e}")


def _install_actor_critic_guards(actor_critic: Any, noise_std_type: str, std_min: float) -> None:
    if hasattr(actor_critic, "update_distribution"):
        original_update_distribution = actor_critic.update_distribution

        @functools.wraps(original_update_distribution)
        def guarded_update_distribution(*args, **kwargs):
            _clamp_std_parameters(actor_critic, noise_std_type, std_min)
            result = original_update_distribution(*args, **kwargs)
            _check_distribution_std(actor_critic, noise_std_type, std_min)
            return result

        actor_critic.update_distribution = guarded_update_distribution

    if hasattr(actor_critic, "act"):
        original_act = actor_critic.act

        @functools.wraps(original_act)
        def guarded_act(*args, **kwargs):
            _clamp_std_parameters(actor_critic, noise_std_type, std_min)
            result = original_act(*args, **kwargs)
            _check_distribution_std(actor_critic, noise_std_type, std_min)
            _check_nested_finite("actor_critic.act", result)
            return result

        actor_critic.act = guarded_act


def _install_env_guards(env: Any) -> None:
    if hasattr(env, "reset"):
        original_reset = env.reset

        @functools.wraps(original_reset)
        def guarded_reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            _check_nested_finite("env.reset", result)
            return result

        env.reset = guarded_reset

    if hasattr(env, "step"):
        original_step = env.step

        @functools.wraps(original_step)
        def guarded_step(actions, *args, **kwargs):
            _check_nested_finite("env.step.actions", actions)
            result = original_step(actions, *args, **kwargs)
            _check_nested_finite("env.step.result", result)
            return result

        env.step = guarded_step


def _install_algorithm_guards(alg: Any, actor_critic: Any, noise_std_type: str, std_min: float) -> None:
    if alg is None or not hasattr(alg, "update"):
        return

    original_update = alg.update

    @functools.wraps(original_update)
    def guarded_update(*args, **kwargs):
        _check_module_parameters_finite(actor_critic, "actor_critic")
        _clamp_std_parameters(actor_critic, noise_std_type, std_min)
        _check_storage_finite(getattr(alg, "storage", None), "alg.storage")
        result = original_update(*args, **kwargs)
        _check_nested_finite("alg.update", result)
        _check_module_parameters_finite(actor_critic, "actor_critic")
        _check_distribution_std(actor_critic, noise_std_type, std_min)
        return result

    alg.update = guarded_update


def _check_storage_finite(storage: Any, storage_name: str) -> None:
    if storage is None or not hasattr(storage, "__dict__"):
        return
    for key, value in storage.__dict__.items():
        if isinstance(value, torch.Tensor):
            _check_tensor_finite(f"{storage_name}.{key}", value)
