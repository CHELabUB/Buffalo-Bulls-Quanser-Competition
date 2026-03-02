#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, ExecuteProcess

def generate_launch_description():

    # odom_node = Node(
    #     package='qcar2_pose_custom',
    #     executable='odom_publisher',
    #     name='odom_publisher',
    #     output='screen'
    # )

    odom_node = Node(                      # NEW - EKF
        package='qcar2_pose_custom',
        executable='odom_publisher_ekf',
        name='odom_publisher',
        output='screen'
    )

    static_tf_node = Node(
        package='qcar2_pose_custom',
        executable='static_tf_publisher',
        name='static_tf_publisher',
        output='screen'
    )

    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'odom_frame': 'odom',
            'map_frame': 'map',
            'base_frame': 'base_link',
            'scan_topic': '/scan',
            'mode': 'mapping',
            'debug_logging': False,
            'transform_timeout': 1.0,
            'tf_buffer_duration': 30.0,
            'max_laser_range': 10.0,
            'minimum_travel_distance': 0.001,
            'minimum_travel_heading': 0.001,
            'resolution': 0.05,
            'map_update_interval': 2.0,
            'do_loop_closing': True,
        }]
    )

    global_pose_node = Node(
        package='qcar2_pose_custom',
        executable='global_pose_publisher',
        name='global_pose_publisher',
        output='screen'
    )

    # Publish initial pose after slam starts
    # Position: (-1.16, 0.908) = parking spot in world coords
    # Orientation: z=0.8536, w=0.5208 = 117.8 degrees
    set_initial_pose = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--once',
             '/initialpose',
             'geometry_msgs/msg/PoseWithCovarianceStamped',
             '{"header": {"frame_id": "map"}, "pose": {"pose": {"position": {"x": -1.16, "y": 0.908, "z": 0.0}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.8536, "w": 0.5208}}}}'],
        output='screen'
    )

    delayed_slam = TimerAction(
        period=3.0,
        actions=[slam_node]
    )

    delayed_pose = TimerAction(
        period=5.0,
        actions=[global_pose_node]
    )

    delayed_initial_pose = TimerAction(
        period=7.0,  # after slam is fully up
        actions=[set_initial_pose]
    )

    return LaunchDescription([
        odom_node,
        static_tf_node,
        delayed_slam,
        delayed_pose,
        delayed_initial_pose,
    ])