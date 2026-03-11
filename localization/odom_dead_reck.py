#!/usr/bin/env python3
# node written to compute odometry for QCar2
# version 1.0
#
# This node estimates robot pose using wheel encoder data and IMU yaw rate.
# Encoder ticks provide linear displacement while IMU provides angular velocity.
# The pose is integrated and published as nav_msgs/Odometry along with a TF transform.

# native import
import numpy as np
from math import cos, sin

# ROS2 imports
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomPublisherNode(Node):

    def __init__(self):
        super().__init__('odom_publisher')
        self.wheel_radius = 0.033
        self.gear_ratio = (13.0 * 19.0) / (70.0 * 37.0)
        self.encoder_counts_per_rev = 2880.0
        self.wheelbase = 0.257
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_tick = None
        self.last_time = None
        self.gyro_z = 0.0
        self.odom_publisher = self.create_publisher(Odometry,'/odom',10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(JointState,'/qcar2_joint',self.joint_callback,10)
        self.create_subscription(Imu,'/qcar2_imu',self.imu_callback,10)
        self.get_logger().info('Odometry publisher started')

    def imu_callback(self, msg: Imu):
        self.gyro_z = msg.angular_velocity.z

    def joint_callback(self, msg: JointState):

        current_tick = msg.position[0]
        current_time = rclpy.time.Time.from_msg(msg.header.stamp)

        if self.last_tick is None:
            self.last_tick = current_tick
            self.last_time = current_time
            return

        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            return

        delta_ticks = current_tick - self.last_tick

        delta_motor_rev = delta_ticks / self.encoder_counts_per_rev
        delta_wheel_rev = delta_motor_rev * self.gear_ratio

        delta_distance = (
            delta_wheel_rev
            * 2.0
            * np.pi
            * self.wheel_radius
        )

        delta_theta = self.gyro_z * dt
        self.theta += delta_theta


        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.x += delta_distance * cos(self.theta)
        self.y += delta_distance * sin(self.theta)


        linear_velocity = delta_distance / dt
        angular_velocity = self.gyro_z

        self.last_tick = current_tick
        self.last_time = current_time

        # Publish odometry
        self.publish_odometry(
            current_time,
            linear_velocity,
            angular_velocity
        )

    def publish_odometry(self, current_time, linear_velocity, angular_velocity):

        now = current_time.to_msg()

        transform = TransformStamped()

        transform.header.stamp = now
        transform.header.frame_id = 'odom'
        transform.child_frame_id = 'base_link'

        transform.transform.translation.x = self.x
        transform.transform.translation.y = self.y
        transform.transform.translation.z = 0.0

        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = sin(self.theta / 2.0)
        transform.transform.rotation.w = cos(self.theta / 2.0)

        self.tf_broadcaster.sendTransform(transform)


        odom = Odometry()

        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.z = sin(self.theta / 2.0)
        odom.pose.pose.orientation.w = cos(self.theta / 2.0)

        odom.twist.twist.linear.x = linear_velocity
        odom.twist.twist.angular.z = angular_velocity

        self.odom_publisher.publish(odom)


def main(args=None):

    rclpy.init(args=args)

    node = OdomPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
