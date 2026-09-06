"""Bringup for the CBF safety-filter stack (ROS 2 Humble).

Launches:
  - cbf_safety_node (nominal tracker + UKF + CBF-QP filter @100 Hz)
  - RViz2 with the shipped safety visualization config (optional)

Usage:
  ros2 launch av_cbf_safety_filter cbf_bringup.launch.py
  ros2 launch av_cbf_safety_filter cbf_bringup.launch.py use_rviz:=false controller:=mpc use_ukf:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _nodes(context):
    pkg = FindPackageShare("av_cbf_safety_filter")
    controller = LaunchConfiguration("controller").perform(context)
    use_rviz = LaunchConfiguration("use_rviz").perform(context).lower() == "true"
    use_ukf = LaunchConfiguration("use_ukf").perform(context).lower() == "true"

    nodes = [
        Node(
            package="av_cbf_safety_filter",
            executable="cbf_safety_node",
            name="cbf_safety_filter",
            output="screen",
            parameters=[
                {
                    "vehicle_yaml": PathJoinSubstitution([pkg, "config", "vehicle_params.yaml"]),
                    "lqr_yaml": PathJoinSubstitution([pkg, "config", "lqr_tuning.yaml"]),
                    "mpc_yaml": PathJoinSubstitution([pkg, "config", "mpc_tuning.yaml"]),
                    "barrier_yaml": PathJoinSubstitution([pkg, "config", "barrier_params.yaml"]),
                    "ukf_yaml": PathJoinSubstitution([pkg, "config", "friction_ukf.yaml"]),
                    "controller": controller,
                    "use_ukf": use_ukf,
                    "mu": 0.9,
                    "vx_ref": 15.0,
                    "y_ref": 0.0,
                    "rate_hz": 100.0,
                }
            ],
            remappings=[
                ("/odom", "/vehicle/odometry"),
                ("/imu", "/vehicle/imu"),
                ("/obstacles", "/perception/obstacles"),
            ],
        )
    ]
    if use_rviz:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", PathJoinSubstitution([pkg, "rviz", "cbf_config.rviz"])],
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("controller", default_value="lqr",
                                  description="nominal tracker: lqr|mpc"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_ukf", default_value="true",
                                  description="UKF friction adaptation in the loop"),
            OpaqueFunction(function=_nodes),
        ]
    )
