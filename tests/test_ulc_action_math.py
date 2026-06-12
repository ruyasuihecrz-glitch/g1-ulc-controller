import importlib.util
from pathlib import Path

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "unitree_rl_lab"
    / "unitree_rl_lab"
    / "tasks"
    / "locomotion"
    / "mdp"
    / "ulc_action_math.py"
)
spec = importlib.util.spec_from_file_location("ulc_action_math", MODULE_PATH)
ulc_action_math = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ulc_action_math)
compose_ulc_joint_position_targets = ulc_action_math.compose_ulc_joint_position_targets
apply_incremental_delay_step = ulc_action_math.apply_incremental_delay_step
ulc_curriculum_ranges = ulc_action_math.ulc_curriculum_ranges


def test_ulc_arm_residual_does_not_double_count_default():
    default = torch.ones(1, 6) * 10.0
    scaled = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    desired_arms = torch.tensor([[100.0, 200.0]])
    leg_ids = torch.tensor([0, 1])
    waist_ids = torch.tensor([2, 3])
    arm_ids = torch.tensor([4, 5])

    targets = compose_ulc_joint_position_targets(default, scaled, desired_arms, leg_ids, waist_ids, arm_ids)

    assert torch.allclose(targets[:, leg_ids], default[:, leg_ids] + scaled[:, leg_ids])
    assert torch.allclose(targets[:, waist_ids], default[:, waist_ids] + scaled[:, waist_ids])
    assert torch.allclose(targets[:, arm_ids], desired_arms + scaled[:, arm_ids])
    assert not torch.allclose(targets[:, arm_ids], desired_arms + default[:, arm_ids] + scaled[:, arm_ids])


def test_ulc_arm_residual_can_be_disabled():
    default = torch.ones(1, 6) * 10.0
    scaled = torch.arange(6, dtype=torch.float32).unsqueeze(0)
    desired_arms = torch.tensor([[100.0, 200.0]])
    leg_ids = torch.tensor([0, 1])
    waist_ids = torch.tensor([2, 3])
    arm_ids = torch.tensor([4, 5])

    targets = compose_ulc_joint_position_targets(
        default,
        scaled,
        desired_arms,
        leg_ids,
        waist_ids,
        arm_ids,
        enable_residual_action=False,
    )

    assert torch.allclose(targets[:, arm_ids], desired_arms)


def test_delay_incremental_release():
    prev = torch.zeros(1, 2)
    buffer = torch.zeros(1, 2)
    executed = torch.zeros(1, 2)
    theoretical = torch.tensor([[1.0, 2.0]])
    delay_mask = torch.tensor([[True, False]])

    prev, buffer, executed = apply_incremental_delay_step(prev, buffer, executed, theoretical, delay_mask)
    assert torch.allclose(buffer, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(executed, torch.tensor([[0.0, 2.0]]))

    theoretical = torch.tensor([[1.5, 3.0]])
    delay_mask = torch.tensor([[False, False]])
    prev, buffer, executed = apply_incremental_delay_step(prev, buffer, executed, theoretical, delay_mask)
    assert torch.allclose(buffer, torch.zeros(1, 2))
    assert torch.allclose(executed, torch.tensor([[1.5, 3.0]]))


def test_curriculum_range():
    closed = ulc_curriculum_ranges(default_height=0.76, alpha_height=0.0, alpha_upper=0.0)
    open_ = ulc_curriculum_ranges(default_height=0.76, alpha_height=1.0, alpha_upper=1.0)

    assert closed["height"] == (0.73, 0.79)
    assert open_["height"] == (0.30, 0.75)
    assert closed["torso_yaw"] == (-0.0, 0.0)
    assert open_["torso_yaw"] == (-2.62, 2.62)
