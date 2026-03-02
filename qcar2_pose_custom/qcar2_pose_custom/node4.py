#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time

class StaticTFPublisher(Node):
    def __init__(self):
        super().__init__('static_tf_publisher')
        self.broadcaster = StaticTransformBroadcaster(self)
        # Republish every second to ensure it's received
        self.timer = self.create_timer(1.0, self.publish_tf)
        self.publish_tf()
        self.get_logger().info('Static TF publisher started')

    def publish_tf(self):
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
        self.broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = StaticTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()