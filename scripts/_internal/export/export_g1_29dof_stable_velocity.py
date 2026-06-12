import argparse
import os
import pathlib
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # isort: skip
sys.path.pop(0)


parser = argparse.ArgumentParser(description="Export the Unitree G1 29DOF StableVelocity policy to ONNX/JIT and deploy paths.")
parser.add_argument("--task", type=str, default="Unitree-G1-29dof-StableVelocity")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "rsl_rl"))
import cli_args as rsl_cli_args  # noqa: E402
sys.path.pop(0)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def resolve_checkpoint(task_name: str, checkpoint: str | None):
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if checkpoint:
        return retrieve_file_path(checkpoint), agent_cfg, log_root_path
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint), agent_cfg, log_root_path


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    checkpoint_path, agent_cfg, _ = resolve_checkpoint(args_cli.task, args_cli.checkpoint)
    run_dir = Path(checkpoint_path).resolve().parent
    export_dir = run_dir / "exported"
    deploy_root = Path("deploy/robots/g1_29dof/config/policy/stable_velocity/v0")
    deploy_export_dir = deploy_root / "exported"
    deploy_params_dir = deploy_root / "params"
    deploy_export_dir.mkdir(parents=True, exist_ok=True)
    deploy_params_dir.mkdir(parents=True, exist_ok=True)

    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_dir.mkdir(parents=True, exist_ok=True)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=str(export_dir), filename="policy.onnx")

    shutil.copy2(export_dir / "policy.pt", deploy_export_dir / "policy.pt")
    shutil.copy2(export_dir / "policy.onnx", deploy_export_dir / "policy.onnx")
    shutil.copy2(run_dir / "params" / "deploy.yaml", deploy_params_dir / "deploy.yaml")
    shutil.copy2(run_dir / "params" / "env.yaml", deploy_params_dir / "env.yaml")
    shutil.copy2(run_dir / "params" / "agent.yaml", deploy_params_dir / "agent.yaml")

    print("[export] checkpoint:", checkpoint_path)
    print("[export] log_export_dir:", export_dir)
    print("[export] deploy_export_dir:", deploy_export_dir)
    print("[export] deploy_params_dir:", deploy_params_dir)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
