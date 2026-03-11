#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import subprocess
import threading
import time
import os
import sys


def _launch_global_pose(logger):
    time.sleep(5.0)  # give slam_toolbox time to come up
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_pose.py')
    if not os.path.exists(script):
        logger.error(f'global_pose_publisher.py not found at: {script}')
        return
    logger.info(f'Launching global_pose_publisher from: {script}')
    subprocess.Popen([sys.executable, script])


class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')

        # --- Static TF: base_link → base_scan (from node4) ---
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_tf()
        self.create_timer(1.0, self.publish_static_tf)

        # --- Odom from TF (from node8) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(0.04, self.publish_odom)  # 25Hz

        self.get_logger().info('Odom publisher started')

        # Launch global_pose_publisher after slam_toolbox has time to start
        t = threading.Thread(
            target=_launch_global_pose,
            args=(self.get_logger(),),
            daemon=True
        )
        t.start()

    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'base_scan'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.1
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t)

    def publish_odom(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )

            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_link'
            odom.pose.pose.position.x = t.transform.translation.x
            odom.pose.pose.position.y = t.transform.translation.y
            odom.pose.pose.position.z = 0.0
            odom.pose.pose.orientation = t.transform.rotation

            self.odom_pub.publish(odom)

        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}',
                throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass

if __name__ == '__main__':
    main()