#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import PoseStamped


class SimpleLocalization(Node):

    def __init__(self):
        super().__init__('qcar2_localization')

        # ---- State ----
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.speed = 0.0

        self.wheel_radius = 0.05  # <-- adjust if needed (meters)

        self.last_time = self.get_clock().now()

        # ---- Subscribers ----
        self.create_subscription(Imu, '/qcar2_imu', self.imu_cb, 10)
        self.create_subscription(JointState, '/qcar2_joint', self.joint_cb, 10)

        # ---- Publisher ----
        self.pose_pub = self.create_publisher(PoseStamped, '/qcar2_pose', 10)

        self.get_logger().info("Localization running using IMU + JointState")

    # --- Get speed from wheel joint velocity ---
    def joint_cb(self, msg: JointState):
        if len(msg.velocity) > 0:
            wheel_angular_speed = msg.velocity[0]  # rad/s
            self.speed = wheel_angular_speed * self.wheel_radius  # m/s

    # --- Update pose using IMU ---
    def imu_cb(self, msg: Imu):

        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0 or dt > 0.5:
            return

        omega = msg.angular_velocity.z

        # Update heading
        self.theta += omega * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        # Update position
        self.x += self.speed * np.cos(self.theta) * dt
        self.y += self.speed * np.sin(self.theta) * dt

        self.publish_pose(now)

    def publish_pose(self, now):
        msg = PoseStamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "map"

        msg.pose.position.x = float(self.x)
        msg.pose.position.y = float(self.y)
        msg.pose.position.z = 0.0

        msg.pose.orientation.z = float(np.sin(self.theta / 2.0))
        msg.pose.orientation.w = float(np.cos(self.theta / 2.0))

        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleLocalization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()