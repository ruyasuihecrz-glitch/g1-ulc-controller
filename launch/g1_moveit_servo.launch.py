"""Launch MoveIt Servo for the G1 right arm.

This launch file intentionally keeps MoveIt focused on the right arm only.
Isaac remains the hardware simulator: it publishes /joint_states through the
UDP bridge, and receives Servo's JointTrajectory through a small UDP bridge.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
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
    servo_params = {"moveit_servo": yaml.safe_load((MOVEIT_CFG / "g1_servo.yaml").read_text())["moveit_servo"]}
    kinematics_params = {
        "robot_description_kinematics": yaml.safe_load((MOVEIT_CFG / "g1_29dof_kinematics.yaml").read_text())
    }
    joint_limit_params = {
        "robot_description_planning": yaml.safe_load((MOVEIT_CFG / "g1_29dof_joint_limits.yaml").read_text())
    }

    servo_udp_host = LaunchConfiguration("servo_udp_host")
    servo_udp_port = LaunchConfiguration("servo_udp_port")

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

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node",
        name="g1_moveit_servo",
        output="screen",
        parameters=[
            servo_params,
            kinematics_params,
            joint_limit_params,
            {"update_period": 0.01},
            {"planning_group_name": "right_arm"},
            {
                "robot_description": robot_description,
                "robot_description_semantic": robot_description_semantic,
                "use_sim_time": True,
            },
        ],
    )

    servo_to_udp = ExecuteProcess(
        cmd=[
            "/usr/bin/python3",
            str(REPO_ROOT / "scripts" / "_internal" / "bridge" / "moveit_servo_udp_bridge.py"),
            "--isaac_host",
            servo_udp_host,
            "--arm_udp_port",
            servo_udp_port,
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("servo_udp_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("servo_udp_port", default_value="15003"),
            robot_state_publisher,
            servo_node,
            servo_to_udp,
        ]
    )
