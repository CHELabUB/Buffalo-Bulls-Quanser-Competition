#!/usr/bin/env python3
# node written for manual QCar2 keyboard control
# version 1.0
#
# This node publishes steering and throttle commands to the QCar2 motor command topic.
# Keyboard input is handled in a separate thread so that command publishing can continue at 20 Hz.

# native import
import sys
import tty
import termios
import threading

# ROS2 imports
import rclpy
from rclpy.node import Node

# QCar2 interface
from qcar2_interfaces.msg import MotorCommands


HELP = """
QCar2 Keyboard Control
----------------------
W / S : increase / decrease throttle
A / D : steer left / steer right
X     : reset throttle and steering to zero
Q     : stop vehicle and quit
"""


class KeyboardControlNode(Node):

    def __init__(self):
        super().__init__('keyboard_control')

        self.publisher = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',
            10
        )

        self.throttle = 0.0
        self.steering = 0.0
        self.throttle_step = 0.1
        self.steering_step = 0.1
        self.max_throttle = 1.0
        self.max_steering = 0.6
        self.timer = self.create_timer(0.05, self.publish_cmd)

        print(HELP)
        print(
            f'Current -> throttle: {self.throttle:.1f}  '
            f'steering: {self.steering:.2f}'
        )

        # Run keyboard input in a separate thread
        self.running = True
        self.keyboard_thread = threading.Thread(target=self.keyboard_loop)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        return key

    def keyboard_loop(self):
        while self.running:
            key = self.get_key()

            if key == 'w' or key == 'W':
                self.throttle = min(
                    self.throttle + self.throttle_step,
                    self.max_throttle
                )

            elif key == 's' or key == 'S':
                self.throttle = max(
                    self.throttle - self.throttle_step,
                    -self.max_throttle
                )

            elif key == 'd' or key == 'D':
                self.steering = max(
                    self.steering - self.steering_step,
                    -self.max_steering
                )

            elif key == 'a' or key == 'A':
                self.steering = min(
                    self.steering + self.steering_step,
                    self.max_steering
                )

            elif key == 'x' or key == 'X':
                self.throttle = 0.0
                self.steering = 0.0

            elif key == 'q' or key == 'Q':
                self.throttle = 0.0
                self.steering = 0.0
                self.publish_cmd()
                self.running = False
                rclpy.shutdown()
                break

            print(
                f'throttle: {self.throttle:.1f}  '
                f'steering: {self.steering:.2f}'
            )

    def publish_cmd(self):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(self.steering), float(self.throttle)]
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.throttle = 0.0
        node.steering = 0.0
        node.publish_cmd()
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
