#!/usr/bin/env python3

import sys
import termios
import tty
import select
import time
import rclpy
from rclpy.node import Node
from qcar2_interfaces.msg import MotorCommands


class QCarTeleop(Node):

    def __init__(self):
        super().__init__('qcar2_teleop')

        self.publisher = self.create_publisher(
            MotorCommands,
            '/qcar2_motor_speed_cmd',
            10
        )

        # Vehicle state
        self.speed = 0.0
        self.steer = 0.0

        # Limits
        self.max_speed = 0.4
        self.max_steer = 0.9

        # Control increments
        self.accel_step = 0.05
        self.brake_step = 0.07
        self.steer_step = 0.08

        # Decay rates
        self.speed_decay = 0.995
        self.steer_decay = 0.93

        # Timeout before decay starts (seconds)
        self.decay_timeout = 0.15
        self.last_input_time = time.time()

        # Terminal setup
        self.settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("Teleop Ready (Improved Smooth Mode)")

    # --------------------------------------------------
    def get_key(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None

    # --------------------------------------------------
    def loop(self):

        key = self.get_key()

        if key is not None:
            self.last_input_time = time.time()

            if key == 'w':
                self.speed += self.accel_step

            elif key == 's':
                self.speed -= self.brake_step

            elif key == 'a':
                self.steer += self.steer_step

            elif key == 'd':
                self.steer -= self.steer_step

            elif key == ' ':
                self.speed = 0.0
                self.steer = 0.0

            elif key == 'q':
                self.publish(0.0, 0.0)
                rclpy.shutdown()
                return

        # Only decay if no key input for short time
        if time.time() - self.last_input_time > self.decay_timeout:
            self.speed *= self.speed_decay
            self.steer *= self.steer_decay

        # Clamp
        self.speed = max(min(self.speed, self.max_speed), -self.max_speed)
        self.steer = max(min(self.steer, self.max_steer), -self.max_steer)

        self.publish(self.steer, self.speed)

    # --------------------------------------------------
    def publish(self, steer, speed):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [float(steer), float(speed)]
        self.publisher.publish(msg)

    # --------------------------------------------------
    def destroy(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = QCarTeleop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()