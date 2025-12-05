#!/usr/bin/env python3
"""
Jetson Orin Nano Optimized Launch File
Includes hardware-specific optimizations for camera and sensors
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # Environment variables for Jetson optimization
    cuda_cache = SetEnvironmentVariable('CUDA_CACHE_DISABLE', '0')
    cuda_cache_path = SetEnvironmentVariable('CUDA_CACHE_PATH', '/tmp/cuda_cache')

    # Jetson-specific camera topic (CSI camera via gstreamer)
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/csi_camera/image_raw',
        description='CSI camera topic'
    )

    # LiDAR topic
    lidar_topic_arg = DeclareLaunchArgument(
        'lidar_topic',
        default_value='/scan',
        description='LiDAR scan topic (RPLIDAR, YDLidar, etc.)'
    )

    # Use PointCloud2 for 3D LiDAR
    use_pointcloud2_arg = DeclareLaunchArgument(
        'use_pointcloud2',
        default_value='false',
        description='True for 3D LiDAR (Velodyne, Livox, etc.)'
    )

    # Rosbridge WebSocket server
    rosbridge_node = Node(
        package='rosbridge_server',
        executable='rosbridge_websocket',
        name='rosbridge_websocket',
        parameters=[{
            'port': 9090,
            'address': '0.0.0.0',
            'retry_startup_delay': 5.0,
            'fragment_timeout': 600,
            'delay_between_messages': 0,
            'max_message_size': 10000000,
        }],
        output='screen'
    )

    # Web video server with Jetson optimization
    web_video_server = Node(
        package='web_video_server',
        executable='web_video_server',
        name='web_video_server',
        parameters=[{
            'port': 8080,
            'address': '0.0.0.0',
            'server_threads': 1,  # Reduce for Jetson
            'ros_threads': 2,
            'default_stream_type': 'mjpeg',
            'quality': 60,  # Lower quality for bandwidth
        }],
        output='screen'
    )

    # LiDAR BEV node with Jetson-optimized settings
    lidar_bev_node = Node(
        package='ros2_web_dashboard',
        executable='lidar_bev_node.py',
        name='lidar_bev_node',
        parameters=[{
            'input_topic': LaunchConfiguration('lidar_topic'),
            'output_topic': '/lidar/bev_image',
            'x_range': [-3.0, 12.0],  # Smaller range for RC car
            'y_range': [-6.0, 6.0],
            'resolution': 0.1,
            'decay_factor': 0.85,
            'use_pointcloud2': LaunchConfiguration('use_pointcloud2'),
        }],
        output='screen'
    )

    # Camera stream node - reduced resolution for Jetson
    camera_stream_node = Node(
        package='ros2_web_dashboard',
        executable='camera_stream_node.py',
        name='camera_stream_node',
        parameters=[{
            'input_topic': LaunchConfiguration('camera_topic'),
            'output_topic': '/camera/image_web',
            'compressed_topic': '/camera/image_compressed',
            'target_width': 480,  # Reduced for Jetson
            'target_height': 270,
            'jpeg_quality': 60,
            'target_fps': 10.0,  # Lower FPS
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

    # Planner visualization
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

    # TF publishers
    tf_base_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_laser',
        arguments=['0.05', '0', '0.08', '0', '0', '0', 'base_link', 'laser']
    )

    tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_camera',
        arguments=['0.08', '0', '0.12', '0', '0.1', '0', 'base_link', 'camera_link']
    )

    # Delayed start for visualization nodes (wait for rosbridge)
    delayed_viz_nodes = TimerAction(
        period=3.0,
        actions=[
            lidar_bev_node,
            camera_stream_node,
            vehicle_state_node,
            planner_viz_node,
        ]
    )

    return LaunchDescription([
        # Environment
        cuda_cache,
        cuda_cache_path,

        # Arguments
        camera_topic_arg,
        lidar_topic_arg,
        use_pointcloud2_arg,

        # Core services (start first)
        rosbridge_node,
        web_video_server,

        # TF publishers
        tf_base_to_laser,
        tf_base_to_camera,

        # Visualization nodes (delayed)
        delayed_viz_nodes,
    ])
