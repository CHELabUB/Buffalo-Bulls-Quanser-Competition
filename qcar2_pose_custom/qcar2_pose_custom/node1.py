#!/usr/bon/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu, JointState
from qcar2_interfaces.msg import MotorCommands
import matplotlib
matplotlib.use('Agg')   # Headless backend
import matplotlib.pyplot as plt
import os
import time
import csv

class car_pose(Node):
    def __init__(self):
        super().__init__("node_1")
        self.imu_topic = "/qcar2_imu"
        self.joint_topic = "/qcar2_joint"
        self.cmd_topic = "/qcar2_motor_speed_cmd"
        self.tach_con = 720
        self.imu_sub = self.create_subscription(Imu, self.imu_topic, self.imu_cb, 10)
        self.joint_sub = self.create_subscription(JointState, self.joint_topic, self.joint_cb, 10)
        self.cmd_pub = self.create_publisher(MotorCommands, self.cmd_topic, 10)
        self.get_logger().info("QCar2Controller running.")
        self.get_logger().info(f"SUB: {self.imu_topic}, {self.joint_topic}")
        self.get_logger().info(f"PUB: {self.cmd_topic}")
        self.imu_data = None
        self.joint_data = None
        self.tach_con = 2880.0
        # logging
        self.log_enabled = True
        self.log_limit = 1000
        self.log_count = 0

        self.log_motor_rpm = []
        self.log_motor_rev = []

        self.log_avx = []
        self.log_avy = []
        self.log_avz = []

        self.log_lax = []
        self.log_lay = []
        self.log_laz = []

        self.log_wheel_rpm = []
        self.log_linear_velocity = []
        self.log_distance = []

    def imu_cb(self, msg: Imu):
        self.imu_data = msg
        self.avx = self.imu_data.angular_velocity.x # rad/s
        self.avy = self.imu_data.angular_velocity.y # rad/s
        self.avz = self.imu_data.angular_velocity.z # rad/s
        self.lax = self.imu_data.linear_acceleration.x # m/s^2
        self.lay = self.imu_data.linear_acceleration.y # m/s^2
        self.laz = self.imu_data.linear_acceleration.z # m/s^2
      
    def joint_cb(self, msg: JointState):
        self.joint_data = msg
        self.torque = msg.effort[0] # Nm
        # Raw encoder data (pre-gear motor shaft)
        tick = msg.position[0]        # counts
        tick_p_s = msg.velocity[0]    # counts/s
        # Motor shaft quantities
        self.motor_rev = tick / self.tach_con                  # motor revolutions
        self.motor_rpm = (tick_p_s / self.tach_con) * 60       # motor RPM
        # Gear ratio from motor to wheel
        gear_ratio = (13.0 * 19.0) / (70.0 * 37.0)
        # Wheel speed computation
        self.wheel_rev = self.motor_rev * gear_ratio
        self.wheel_rps = (tick_p_s / self.tach_con) * gear_ratio    # wheel rps
        self.wheel_rpm = self.wheel_rps * 60                        # wheel rpm
        # Linear motion
        wheel_radius = 0.033  # meters
        self.linear_velocity = self.wheel_rps * 2.0 * np.pi * wheel_radius   # m/s
        self.distance = self.wheel_rev * 2.0 * np.pi * wheel_radius          # meters
        self.collect_log()

    def send_cmd(self, steer_rad: float, speed_mps: float):
        cmd = MotorCommands()
        cmd.motor_names = ["steering_angle", "motor_throttle"]
        cmd.values = [float(steer_rad), float(speed_mps)]
        self.cmd_pub.publish(cmd)

    def control_loop(self):
        if self.last_imu is None or self.last_joint is None:
            return
        steer_cmd = 0.0
        throttle_cmd = 0.0
        self.send_cmd(steer_cmd, throttle_cmd)

    def print_params(self):
        if self.joint_data is None or self.imu_data is None:
            self.get_logger().info("No joint or IMU data yet.")
            return

        self.get_logger().info("==== Encoder ====")
        self.get_logger().info(f"Motor revolutions: {self.motor_rev:.3f} rev")
        self.get_logger().info(f"Motor RPM:         {self.motor_rpm:.3f} rpm")
        self.get_logger().info(f"Wheel RPS:         {self.wheel_rps:.3f} rps")
        self.get_logger().info(f"Wheel RPM:         {self.wheel_rpm:.3f} rpm")
        self.get_logger().info(f"Linear velocity:   {self.linear_velocity:.3f} m/s")
        self.get_logger().info(f"Distance traveled: {self.distance:.3f} m")

        self.get_logger().info("---- IMU Data ----")
        self.get_logger().info(f"Angular vel X:     {self.avx:.6f} rad/s")
        self.get_logger().info(f"Angular vel Y:     {self.avy:.6f} rad/s")
        self.get_logger().info(f"Angular vel Z:     {self.avz:.6f} rad/s")
        self.get_logger().info(f"Linear acc X:      {self.lax:.6f} m/s^2")
        self.get_logger().info(f"Linear acc Y:      {self.lay:.6f} m/s^2")
        self.get_logger().info(f"Linear acc Z:      {self.laz:.6f} m/s^2")

        self.get_logger().info("==============================")

    def collect_log(self):
        if not self.log_enabled:
            return

        if self.joint_data is None or self.imu_data is None:
            return

        # Append values
        self.log_motor_rev.append(self.motor_rev)
        self.log_motor_rpm.append(self.motor_rpm)

        self.log_avx.append(self.avx)
        self.log_avy.append(self.avy)
        self.log_avz.append(self.avz)

        self.log_lax.append(self.lax)
        self.log_lay.append(self.lay)
        self.log_laz.append(self.laz)

        self.log_wheel_rpm.append(self.wheel_rpm)
        self.log_linear_velocity.append(self.linear_velocity)
        self.log_distance.append(self.distance)

        self.log_count += 1

        if self.log_count >= self.log_limit:
            self.get_logger().info("Log limit reached. Generating plot...")
            self.log_enabled = False
            self.generate_plot()

    def generate_plot(self):
        # Common x-axis: sample index
        n = len(self.log_motor_rpm)
        if n == 0:
            self.get_logger().info("No logged data to plot.")
            return
        t = range(n)

        save_dir = "/workspaces/isaac_ros-dev/ros2/src/qcar2_pose_custom/logs"
        os.makedirs(save_dir, exist_ok=True)
        ts = int(time.time())

        # -----------------------------
        # Plot 1: Gyro (avx, avy, avz)
        # -----------------------------
        fig1 = plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, self.log_avx)
        plt.title("Angular Velocity X (rad/s)")

        plt.subplot(3, 1, 2)
        plt.plot(t, self.log_avy)
        plt.title("Angular Velocity Y (rad/s)")

        plt.subplot(3, 1, 3)
        plt.plot(t, self.log_avz)
        plt.title("Angular Velocity Z (rad/s)")

        plt.tight_layout()
        f1 = os.path.join(save_dir, f"qcar2_gyro_{ts}.png")
        plt.savefig(f1)
        plt.close(fig1)

        # -----------------------------
        # Plot 2: Accel (lax, lay, laz)
        # -----------------------------
        fig2 = plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, self.log_lax)
        plt.title("Linear Acceleration X (m/s^2)")

        plt.subplot(3, 1, 2)
        plt.plot(t, self.log_lay)
        plt.title("Linear Acceleration Y (m/s^2)")

        plt.subplot(3, 1, 3)
        plt.plot(t, self.log_laz)
        plt.title("Linear Acceleration Z (m/s^2)")

        plt.tight_layout()
        f2 = os.path.join(save_dir, f"qcar2_accel_{ts}.png")
        plt.savefig(f2)
        plt.close(fig2)

        # -----------------------------------------
        # Plot 3: Motor (motor_rev, motor_rpm)
        # -----------------------------------------
        fig3 = plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, self.log_motor_rev)
        plt.title("Motor Revolutions (rev)")

        plt.subplot(2, 1, 2)
        plt.plot(t, self.log_motor_rpm)
        plt.title("Motor RPM")

        plt.tight_layout()
        f3 = os.path.join(save_dir, f"qcar2_motor_{ts}.png")
        plt.savefig(f3)
        plt.close(fig3)

        # -----------------------------------------
        # Plot 4: Vehicle State (wheel + velocity + distance)
        # -----------------------------------------
        fig4 = plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, self.log_wheel_rpm)
        plt.title("Wheel RPM")

        plt.subplot(3, 1, 2)
        plt.plot(t, self.log_linear_velocity)
        plt.title("Linear Velocity (m/s)")

        plt.subplot(3, 1, 3)
        plt.plot(t, self.log_distance)
        plt.title("Distance Traveled (m)")

        plt.tight_layout()
        f4 = os.path.join(save_dir, f"qcar2_vehicle_{ts}.png")
        plt.savefig(f4)
        plt.close(fig4)

        self.get_logger().info(f"Saved plots:\n  {f1}\n  {f2}\n  {f3}\n  {f4}")
        self.save_csv(save_dir, ts)

    def save_csv(self, save_dir, timestamp):
        file_path = os.path.join(save_dir, f"qcar2_log_{timestamp}.csv")

        n = len(self.log_motor_rpm)

        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "index",
                "motor_rev",
                "motor_rpm",
                "wheel_rpm",
                "linear_velocity",
                "distance",
                "avx", "avy", "avz",
                "lax", "lay", "laz"
            ])

            # Rows
            for i in range(n):
                writer.writerow([
                i,
                self.log_motor_rev[i],
                self.log_motor_rpm[i],
                self.log_wheel_rpm[i],
                self.log_linear_velocity[i],
                self.log_distance[i],
                self.log_avx[i],
                self.log_avy[i],
                self.log_avz[i],
                self.log_lax[i],
                self.log_lay[i],
                self.log_laz[i]
            ])

        self.get_logger().info(f"CSV saved to {file_path}")

    
def main(args=None):
    rclpy.init(args=args)
    obj = car_pose()

    try:
        while rclpy.ok():
            rclpy.spin_once(obj, timeout_sec=0.1)
            obj.print_params()
    except KeyboardInterrupt:
        pass

    obj.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
