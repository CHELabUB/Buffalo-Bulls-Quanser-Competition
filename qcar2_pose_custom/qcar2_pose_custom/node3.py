#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from math import cos, sin

class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')

        # Robot parameters
        self.wheel_radius = 0.033        # meters
        self.gear_ratio = (13.0 * 19.0) / (70.0 * 37.0)
        self.tach_con = 2880.0           # encoder counts per motor revolution
        self.wheelbase = 0.257           # meters (from original EKF code)

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_tick = None
        self.last_time = None
        self.gyro_z = 0.0

        # Publishers and TF
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.create_subscription(JointState, '/qcar2_joint', self.joint_cb, 10)
        self.create_subscription(Imu, '/qcar2_imu', self.imu_cb, 10)

        self.get_logger().info('Odometry publisher started')

    def imu_cb(self, msg: Imu):
        self.gyro_z = msg.angular_velocity.z  # rad/s yaw rate

    def joint_cb(self, msg: JointState):
        current_tick = msg.position[0]
        # current_time = self.get_clock().now()
        current_time = rclpy.time.Time.from_msg(msg.header.stamp)

        # First message — just store and return
        if self.last_tick is None:
            self.last_tick = current_tick
            self.last_time = current_time
            return

        # Time delta
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        # Encoder delta → distance
        delta_ticks = current_tick - self.last_tick
        delta_motor_rev = delta_ticks / self.tach_con
        delta_wheel_rev = delta_motor_rev * self.gear_ratio
        delta_dist = delta_wheel_rev * 2.0 * np.pi * self.wheel_radius

        # Update heading using gyro (better than encoder-only)
        delta_theta = self.gyro_z * dt
        self.theta += delta_theta
        # Keep theta in -pi to pi
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # Update position
        self.x += delta_dist * cos(self.theta)
        self.y += delta_dist * sin(self.theta)

        # Linear velocity
        linear_vel = delta_dist / dt
        angular_vel = self.gyro_z

        # Save for next iteration
        self.last_tick = current_tick
        self.last_time = current_time

        # Publish
        self.publish_odom(current_time, linear_vel, angular_vel)

    def publish_odom(self, current_time, linear_vel, angular_vel):
        now = current_time.to_msg()

        # --- TF: odom → base_link ---
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0

        # Quaternion from yaw
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = sin(self.theta / 2.0)
        t.transform.rotation.w = cos(self.theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

        # --- Odometry message ---
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.z = sin(self.theta / 2.0)
        odom.pose.pose.orientation.w = cos(self.theta / 2.0)
        odom.twist.twist.linear.x = linear_vel
        odom.twist.twist.angular.z = angular_vel

        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()