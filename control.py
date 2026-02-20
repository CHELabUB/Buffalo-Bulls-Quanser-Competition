#!/usr/bin/env python3
# node written by Haosong to demonstrate qcar2 control
# version 1.1, added non-blocking input for manual control
# 

# native import
import sys
import select

# ROS2 imports
import rclpy
from rclpy.node import Node

# Qcar2 interface
from qcar2_interfaces.msg import MotorCommands


class QCar2ControllerNode(Node):

    def __init__(self):
        super().__init__('qcar2_controller_node')

        self.publisher = self.create_publisher(
            MotorCommands,
            'qcar2_motor_speed_cmd',
            10
        )

        self.mode = "AUTO"
        self.steering = 0.0
        self.throttle = 0.0

        self.timer = self.create_timer(0.05, self.loop)

        self.get_logger().info("Started in AUTO mode")
        self.get_logger().info("Type command + Enter")

    def loop(self):
        # ---- Publish ----
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']

        if self.mode == "AUTO":
            msg.values = [0.0, 0.2]
        else:
            msg.values = [self.steering, self.throttle]

        self.publisher.publish(msg)

        # ---- Non-blocking input ----
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.readline().strip().lower()

            if key == "c":
                self.mode = "MANUAL" if self.mode == "AUTO" else "AUTO"
                print(f"Switched to {self.mode}")

            elif self.mode == "MANUAL":
                if key == "i":
                    self.throttle += 0.5
                elif key == "k":
                    self.throttle -= 0.5
                elif key == "j":
                    self.steering += 0.5
                elif key == "l":
                    self.steering -= 0.5
                elif key == "stop":
                    self.throttle = 0.0
                    self.steering = 0.0

                self.throttle = max(min(self.throttle, 1.0), -1.0)
                self.steering = max(min(self.steering, 1.0), -1.0)


def main(args=None):
    rclpy.init(args=args)
    node = QCar2ControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()