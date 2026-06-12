"""Launch MoveIt move_group for direct G1 right-arm IK queries."""

from __future__ import annotations

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


REPO_ROOT = Path("/workspace/data/repos/unitree_rl_lab")
G1_URDF = Path("/workspace/data/repos/unitree_rl_gym/resources/robots/g1_description/g1_29dof_rev_1_0.urdf")
MOVEIT_CFG = REPO_ROOT / "configs" / "moveit"


def generate_launch_description() -> LaunchDescription:
    robot_description = ParameterValue(G1_URDF.read_text(encoding="utf-8"), value_type=str)
    robot_description_semantic = ParameterValue(
        (MOVEIT_CFG / "g1_29dof_right_arm.srdf").read_text(encoding="utf-8"), value_type=str
    )
    kinematics_params = {
        "robot_description_kinematics": yaml.safe_load((MOVEIT_CFG / "g1_29dof_kinematics.yaml").read_text())
    }
    joint_limit_params = {
        "robot_description_planning": yaml.safe_load((MOVEIT_CFG / "g1_29dof_joint_limits.yaml").read_text())
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="g1_robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            kinematics_params,
            joint_limit_params,
            {
                "robot_description": robot_description,
                "robot_description_semantic": robot_description_semantic,
                "use_sim_time": True,
                "publish_robot_description": True,
                "publish_robot_description_semantic": True,
                "allow_trajectory_execution": False,
                "planning_pipelines": ["ompl"],
                "default_planning_pipeline": "ompl",
                "ompl.planning_plugins": ["ompl_interface/OMPLPlanner"],
                "ompl.start_state_max_bounds_error": 0.1,
            },
        ],
    )

    return LaunchDescription([robot_state_publisher, move_group])
