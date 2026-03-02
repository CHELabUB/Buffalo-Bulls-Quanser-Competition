#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster


class FancyLocalization(Node):

    def __init__(self):
        super().__init__('qcar2_localization')

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.speed = 0.0

        self.wheel_radius = 0.05
        self.last_time = self.get_clock().now()

        self.create_subscription(Imu, '/qcar2_imu', self.imu_cb, 10)
        self.create_subscription(JointState, '/qcar2_joint', self.joint_cb, 10)

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.path_pub = self.create_publisher(Path, '/path', 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.path = Path()
        self.path.header.frame_id = "odom"

        self.get_logger().info("Localization Running")

    def joint_cb(self, msg: JointState):
        if len(msg.velocity) > 0:
            wheel_angular_speed = msg.velocity[0]
            self.speed = wheel_angular_speed * self.wheel_radius

    def imu_cb(self, msg: Imu):

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        omega = msg.angular_velocity.z
        self.theta += omega * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        self.x += self.speed * np.cos(self.theta) * dt
        self.y += self.speed * np.sin(self.theta) * dt

        self.publish_all(now)

    def publish_all(self, now):

        # ---- Odometry ----
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)

        qz = np.sin(self.theta / 2.0)
        qw = np.cos(self.theta / 2.0)
        odom.pose.pose.orientation = Quaternion(z=float(qz), w=float(qw))

        self.odom_pub.publish(odom)

        # ---- TF ----
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = float(self.x)
        t.transform.translation.y = float(self.y)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        self.tf_broadcaster.sendTransform(t)

        # ---- Path ----
        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose

        self.path.header.stamp = now.to_msg()
        self.path.poses.append(pose)

        self.path_pub.publish(self.path)


def main(args=None):
    rclpy.init(args=args)
    node = FancyLocalization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()