#!/usr/bin/env python3
"""
ROS2 Web Dashboard Launch File
Launches all nodes for web-based visualization on Jetson Orin Nano
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Package share directory
    pkg_share = FindPackageShare('ros2_web_dashboard')

    # Launch arguments
    rosbridge_port_arg = DeclareLaunchArgument(
        'rosbridge_port',
        default_value='9090',
        description='WebSocket port for rosbridge'
    )

    web_video_port_arg = DeclareLaunchArgument(
        'web_video_port',
        default_value='8080',
        description='Port for web video server'
    )

    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/scan',
        description='Input LiDAR topic'
    )

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Input camera topic'
    )

    # Rosbridge WebSocket server
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{
            'port': LaunchConfiguration('rosbridge_port'),
            'address': '0.0.0.0',
            'retry_startup_delay': 5.0,
            'fragment_timeout': 600,
            'delay_between_messages': 0,
            'max_message_size': 10000000,
            'unregister_timeout': 10.0,
            'use_compression': False,
        }],
        output='screen'
    )

    # Web video server for camera streaming
    web_video_server = Node(
        package='web_video_server',
        executable='web_video_server',
        name='web_video_server',
        parameters=[{
            'port': LaunchConfiguration('web_video_port'),
            'address': '0.0.0.0',
            'server_threads': 2,
            'ros_threads': 4,
            'default_stream_type': 'mjpeg',
            'quality': 70,
        }],
        output='screen'
    )

    # LiDAR BEV visualization node
    lidar_bev_node = Node(
        package='ros2_web_dashboard',
        executable='lidar_bev_node.py',
        name='lidar_bev_node',
        parameters=[{
            'input_topic': LaunchConfiguration('lidar_topic'),
            'output_topic': '/lidar/bev_image',
            'x_range': [-5.0, 15.0],
            'y_range': [-10.0, 10.0],
            'resolution': 0.1,
            'decay_factor': 0.9,
            'use_pointcloud2': False,
        }],
        output='screen'
    )

    # Camera stream node
    camera_stream_node = Node(
        package='ros2_web_dashboard',
        executable='camera_stream_node.py',
        name='camera_stream_node',
        parameters=[{
            'input_topic': LaunchConfiguration('camera_topic'),
            'output_topic': '/camera/image_web',
            'compressed_topic': '/camera/image_compressed',
            'target_width': 640,
            'target_height': 360,
            'jpeg_quality': 70,
            'target_fps': 15.0,
            'draw_info': True,
        }],
        output='screen'
    )

    # Vehicle state publisher
    vehicle_state_node = Node(
        package='ros2_web_dashboard',
        executable='vehicle_state_node.py',
        name='vehicle_state_node',
        parameters=[{
            'publish_rate': 10.0,
            'state_topic': '/vehicle/state_json',
        }],
        output='screen'
    )

    # Planner visualization node
    planner_viz_node = Node(
        package='ros2_web_dashboard',
        executable='planner_viz_node.py',
        name='planner_viz_node',
        parameters=[{
            'path_topic': '/planner/path',
            'markers_topic': '/planner/markers',
            'detections_topic': '/planner/detections_json',
            'publish_rate': 10.0,
        }],
        output='screen'
    )

    # Static transform publisher for base_link -> laser
    tf_base_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_laser',
        arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'laser']
    )

    # Static transform publisher for base_link -> camera
    tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_camera',
        arguments=['0.1', '0', '0.15', '0', '0', '0', 'base_link', 'camera_link']
    )

    # HTTP server for web dashboard (using Python)
    web_server = ExecuteProcess(
        cmd=[
            'python3', '-m', 'http.server', '8000',
            '--directory', PathJoinSubstitution([pkg_share, 'web'])
        ],
        name='web_server',
        output='screen'
    )

    return LaunchDescription([
        # Arguments
        rosbridge_port_arg,
        web_video_port_arg,
        lidar_topic_arg,
        camera_topic_arg,

        # Core services
        rosbridge_node,
        web_video_server,

        # Visualization nodes
        lidar_bev_node,
        camera_stream_node,
        vehicle_state_node,
        planner_viz_node,

        # TF publishers
        tf_base_to_laser,
        tf_base_to_camera,

        # Web server
        web_server,
    ])
