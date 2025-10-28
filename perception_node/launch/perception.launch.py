import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():

    # Get the path to your package's "share" directory
    pkg_share = get_package_share_directory('perception_node')

    # --- 1. Launch Argument for Bag File ---
    # This allows you to specify the bag file path from the command line
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        description='Path to the ROS bag file (depth.db3)'
    )
    
    # --- 2. RVIZ Configuration ---
    # Path to your .rviz configuration file
    rviz_config_file = PathJoinSubstitution([
        pkg_share, 'config', 'perception.rviz'
    ])

    # --- 3. Bag Player Node ---
    bag_play_node = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play',
            LaunchConfiguration('bag_path'), # Get path from launch argument
            '-l'  # Loop the bag
        ],
        name='ros2_bag_play',
        output='screen'
    )

    # --- 4. Your Perception Node ---
    perception_node = Node(
        package='perception_node',
        executable='depth_processor',
        name='depth_processor_node',
        output='screen'
    )

    # --- 5. RVIZ Node ---
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file], # Load your saved config
        output='screen'
    )

    # --- Create and return the Launch Description ---
    return LaunchDescription([
        bag_path_arg,
        bag_play_node,
        perception_node,
        rviz_node
    ])
