import torch

from unitree_rl_lab.utils.rsl_rl_checkpoint import (
    CHECKPOINT_INFO_KEY,
    attach_resumable_checkpointing,
    collect_resume_state,
    restore_resume_state,
)


class FakeCommandTerm:
    def __init__(self):
        self.alpha_height = 0.55
        self.alpha_upper = 0.10
        self._last_curriculum_step = 12345
        self.command_tensor = torch.arange(12, dtype=torch.float32).reshape(2, 6)
        self.metrics = {
            "alpha_height": torch.zeros(2),
            "alpha_upper": torch.zeros(2),
        }


class FakeCommandManager:
    def __init__(self):
        self._terms = {"ulc_command": FakeCommandTerm()}


class FakeActionManager:
    def __init__(self):
        self.action = torch.ones(2, 3)
        self.prev_action = torch.full((2, 3), 2.0)


class FakeRewardManager:
    def __init__(self):
        self._episode_sums = {"height": torch.full((2,), 3.0)}


class FakeRobotData:
    def __init__(self):
        self.root_state_w = torch.arange(26, dtype=torch.float32).reshape(2, 13)
        self.joint_pos = torch.ones(2, 4)
        self.joint_vel = torch.full((2, 4), 2.0)
        self.joint_pos_target = torch.full((2, 4), 3.0)
        self.joint_vel_target = torch.full((2, 4), 4.0)
        self.joint_effort_target = torch.full((2, 4), 5.0)


class FakeRobot:
    def __init__(self):
        self.data = FakeRobotData()

    def write_root_state_to_sim(self, root_state):
        self.data.root_state_w.copy_(root_state)

    def write_joint_state_to_sim(self, joint_pos, joint_vel):
        self.data.joint_pos.copy_(joint_pos)
        self.data.joint_vel.copy_(joint_vel)


class FakeScene:
    def __init__(self):
        self.robot = FakeRobot()

    def __getitem__(self, name):
        if name != "robot":
            raise KeyError(name)
        return self.robot


class FakeSim:
    def __init__(self):
        self.forward_called = False

    def forward(self):
        self.forward_called = True


class FakeEnv:
    def __init__(self):
        self.device = "cpu"
        self.common_step_counter = 77
        self.episode_length_buf = torch.tensor([11, 22])
        self.command_manager = FakeCommandManager()
        self.action_manager = FakeActionManager()
        self.reward_manager = FakeRewardManager()
        self.scene = FakeScene()
        self.sim = FakeSim()

    @property
    def unwrapped(self):
        return self


class FakeRunner:
    def __init__(self, env):
        self.env = env
        self.saved_infos = None
        self.loaded_infos = None

    def save(self, path, infos=None):
        self.saved_infos = infos

    def load(self, path, load_optimizer=True, map_location=None):
        return self.loaded_infos


def test_collect_and_restore_resume_state_round_trip():
    env = FakeEnv()
    state = collect_resume_state(env)

    env.common_step_counter = 0
    env.episode_length_buf.zero_()
    env.command_manager._terms["ulc_command"].alpha_height = 0.0
    env.command_manager._terms["ulc_command"].alpha_upper = 0.0
    env.command_manager._terms["ulc_command"]._last_curriculum_step = -1
    env.command_manager._terms["ulc_command"].command_tensor.zero_()
    env.action_manager.action.zero_()
    env.action_manager.prev_action.zero_()
    env.reward_manager._episode_sums["height"].zero_()
    env.scene.robot.data.root_state_w.zero_()
    env.scene.robot.data.joint_pos.zero_()
    env.scene.robot.data.joint_vel.zero_()

    assert restore_resume_state(env, state)

    term = env.command_manager._terms["ulc_command"]
    assert env.common_step_counter == 77
    assert torch.equal(env.episode_length_buf, torch.tensor([11, 22]))
    assert term.alpha_height == 0.55
    assert term.alpha_upper == 0.10
    assert term._last_curriculum_step == 12345
    assert torch.equal(term.command_tensor, torch.arange(12, dtype=torch.float32).reshape(2, 6))
    assert torch.equal(env.action_manager.action, torch.ones(2, 3))
    assert torch.equal(env.action_manager.prev_action, torch.full((2, 3), 2.0))
    assert torch.equal(env.reward_manager._episode_sums["height"], torch.full((2,), 3.0))
    assert torch.equal(env.scene.robot.data.joint_pos, torch.ones(2, 4))
    assert torch.equal(env.scene.robot.data.joint_vel, torch.full((2, 4), 2.0))
    assert env.sim.forward_called


def test_runner_patch_stores_and_restores_resume_state():
    env = FakeEnv()
    runner = FakeRunner(env)
    attach_resumable_checkpointing(runner)

    runner.save("model.pt", infos={"keep": 1})
    assert runner.saved_infos["keep"] == 1
    assert CHECKPOINT_INFO_KEY in runner.saved_infos

    env.command_manager._terms["ulc_command"].alpha_height = 0.0
    runner.loaded_infos = runner.saved_infos
    runner.load("model.pt")

    assert env.command_manager._terms["ulc_command"].alpha_height == 0.55
