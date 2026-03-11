#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np
import math


class GlobalPosePublisher(Node):
    def __init__(self):
        super().__init__('global_pose_publisher')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pose_pub = self.create_publisher(PoseStamped, '/global_pose', 10)
        self.timer = self.create_timer(0.05, self.publish_pose)  # 20Hz

        # World offset - parking spot in simulator world coordinates
        self.world_offset_x = -1.16
        self.world_offset_y = 0.908
        self.world_offset_yaw = 2.0559979  # 117.8 degrees in radians

        self.get_logger().info('Global pose publisher started')

    def publish_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )

            # Raw pose from slam (relative to map origin)
            x = t.transform.translation.x
            y = t.transform.translation.y

            q = [t.transform.rotation.x,
                 t.transform.rotation.y,
                 t.transform.rotation.z,
                 t.transform.rotation.w]
            yaw = R.from_quat(q).as_euler('xyz')[2]

            # Apply world offset to convert to simulator world coordinates
            cos_o = math.cos(-self.world_offset_yaw)
            sin_o = math.sin(-self.world_offset_yaw)
            world_x = self.world_offset_x + cos_o * x - sin_o * y
            world_y = self.world_offset_y + sin_o * x + cos_o * y
            world_yaw = yaw - self.world_offset_yaw

            # Build pose message in world coordinates
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0

            # Convert world_yaw back to quaternion
            world_q = R.from_euler('z', world_yaw).as_quat()
            pose.pose.orientation.x = world_q[0]
            pose.pose.orientation.y = world_q[1]
            pose.pose.orientation.z = world_q[2]
            pose.pose.orientation.w = world_q[3]

            self.get_logger().info(
                f'x:{world_x:.3f} '
                f'y:{world_y:.3f} '
                f'yaw:{np.degrees(world_yaw):.1f}deg',
                throttle_duration_sec=1.0
            )

            self.pose_pub.publish(pose)

        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}',
                throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
