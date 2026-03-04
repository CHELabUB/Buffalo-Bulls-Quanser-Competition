#!/usr/bin/env python3
"""
QCar2 Auto/Manual Drive + YOLOv11 Traffic Sign Response
========================================================
Mode switch:
  TAB       -> Toggle Auto/Manual mode (press in terminal)

Manual mode:
  W / UP    -> Forward
  S / DOWN  -> Backward
  A / LEFT  -> Turn left
  D / RIGHT -> Turn right
  SPACE     -> Emergency stop
 
Auto mode:
  Auto straight driving + traffic sign response
  Redlight    -> Stop and wait for green
  Greenlight  -> Resume driving
  Stop        -> Stop 1s then continue
  Yellowlight -> Speed up
  Yield       -> Slow down 40%
  Roundabout  -> Slow down 40%
 
Quit:
  Q           -> Exit program
"""
 
import rclpy
from rclpy.node import Node
from qcar2_interfaces.msg import MotorCommands
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pynput import keyboard
from ultralytics import YOLO
import cv2
import time
import threading
 
# ============================================================
# Configuration
# ============================================================
TOPIC_CAMERA   = '/camera/color_image'
MODEL_PATH     = '/workspaces/isaac_ros-dev/data.pt'
 
LINEAR_SPEED   = 0.5
ANGULAR_SPEED  = 0.5
 
AUTO_SPEED     = 0.5
SPEED_SLOW     = 0.2
SPEED_FAST     = 1.5
SPEED_STOP     = 0.0
 
FAST_HOLD_SEC  = 1.5
STOP_HOLD_SEC  = 1.0
SLOW_HOLD_SEC  = 2.0
 
CONFIDENCE     = 0.5
 
STOP_COOLDOWN_SEC = 3.0
SLOW_COOLDOWN_SEC = 3.0
RED_COOLDOWN_SEC  = 0.5
 
DEFAULT_BOX_AREA = 3000
 
LIGHT_X_MIN  = 0.35
LIGHT_X_MAX  = 0.65
LIGHT_LABELS = {'Redlight', 'Greenlight', 'Yellowlight'}
 
LIGHT_AREA_MIN = 650
LIGHT_AREA_MAX = 850
 
MIN_BOX_AREA = {
    'Stop'      : 2000,
    'Yield'     : 2000,
    'Roundabout': 2000,
    'Crosswalk' : 2000,
}
 
CLASS_COLORS = {
    'Greenlight' : (0, 255, 0),
    'Redlight'   : (0, 0, 255),
    'Stop'       : (0, 0, 200),
    'Yellowlight': (0, 200, 255),
    'Crosswalk'  : (255, 200, 0),
    'Roundabout' : (0, 165, 255),
    'Yield'      : (180, 0, 180),
}
 
 
# ============================================================
# Drive States
# ============================================================
class DriveState:
    NORMAL    = 'Normal'
    SLOW      = 'Slow'
    FAST      = 'Fast'
    STOP_SIGN = 'Stop(1s)'
    RED_LIGHT = 'RedLight'
    COOLDOWN  = 'Cooldown'
 
 
# ============================================================
# Main Node
# ============================================================
class QCar2Controller(Node):
 
    def __init__(self):
        super().__init__('qcar2_controller')
 
        self.get_logger().info('Loading YOLO model...')
        self.model = YOLO(MODEL_PATH)
        self.get_logger().info(f'Classes: {list(self.model.names.values())}')
 
        self.cmd_pub = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)
        self.camera_sub = self.create_subscription(
            Image, TOPIC_CAMERA, self.camera_callback, 10)
 
        self.bridge = CvBridge()
        self.lock   = threading.Lock()
 
        self.auto_mode = False
 
        # Manual mode state
        self.keys_pressed    = set()
        self.manual_throttle = 0.0
        self.manual_steering = 0.0
 
        # Auto mode state
        self.auto_state   = DriveState.NORMAL
        self.auto_scale   = 1.0
        self.state_timer  = None
        self.cooldown_end = 0.0
 
        # Display
        self.latest_detections = []
        self.frame_count   = 0
        self.fps           = 0.0
        self.last_fps_time = time.time()
        self.fps_counter   = 0
        self.display_frame = None
        self.frame_lock    = threading.Lock()
        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()
 
        self.timer = self.create_timer(0.05, self.publish_command)
 
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()
 
        self.print_instructions()
 
    # ============================================================
    # Display thread
    # ============================================================
    def display_loop(self):
        while True:
            with self.frame_lock:
                frame = self.display_frame.copy() if self.display_frame is not None else None
            if frame is not None:
                cv2.imshow('QCar2 Controller', frame)
                cv2.waitKey(33)
            time.sleep(0.001)
 
    # ============================================================
    # Print instructions
    # ============================================================
    def print_instructions(self):
        print('\n' + '='*55)
        print('   QCar2 Auto/Manual Controller + Traffic Sign Response')
        print('='*55)
        print('  NOTE: Press keys in terminal, not in camera window')
        print('  TAB      -> Toggle Auto/Manual mode')
        print('  W/S/A/D  -> Manual control')
        print('  SPACE    -> Emergency stop')
        print('  Q        -> Quit')
        print('='*55)
        print(f'  STOP_COOLDOWN = {STOP_COOLDOWN_SEC}s')
        print(f'  SLOW_COOLDOWN = {SLOW_COOLDOWN_SEC}s')
        print('  Current mode: Manual\n')
 
    # ============================================================
    # Publish command at 20Hz
    # ============================================================
    def publish_command(self):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        if self.auto_mode:
            with self.lock:
                scale = self.auto_scale
            msg.values = [0.0, float(AUTO_SPEED * scale)]
        else:
            msg.values = [float(self.manual_steering), float(self.manual_throttle)]
        self.cmd_pub.publish(msg)
 
    # ============================================================
    # Camera callback
    # ============================================================
    def camera_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if frame is None or frame.size == 0:
                return
 
            self.frame_count += 1
            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.fps = self.fps_counter / (now - self.last_fps_time)
                self.fps_counter   = 0
                self.last_fps_time = now
 
            if self.frame_count % 3 == 0:
                # Crop bottom 5% to ignore onboard camera
                crop_h = int(frame.shape[0] * 0.95)
                detections = self.run_yolo(frame[:crop_h, :])
                self.latest_detections = detections
                if self.auto_mode:
                    self.update_auto_state(detections)
 
            display = self.draw_frame(frame)
            with self.frame_lock:
                self.display_frame = display
 
        except Exception as e:
            self.get_logger().error(f'Camera error: {e}')
 
    # ============================================================
    # YOLO inference: returns ALL detections (for display)
    # ============================================================
    def run_yolo(self, frame):
        detections = []
        results = self.model(frame, conf=CONFIDENCE, verbose=False)
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                confidence = float(box.conf[0])
                label      = self.model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                detections.append({
                    'label'     : label,
                    'confidence': confidence,
                    'bbox'      : (x1, y1, x2, y2),
                    'area'      : area,
                    'valid'     : self._is_valid_detection(label, area, x1, x2, frame),
                })
        return detections
 
    def _is_valid_detection(self, label, area, x1, x2, frame):
        """Check if detection meets area and position requirements for response"""
        if label in LIGHT_LABELS:
            if area < LIGHT_AREA_MIN or area > LIGHT_AREA_MAX:
                return False
            frame_w  = frame.shape[1]
            cx_ratio = (x1 + x2) / 2 / frame_w
            if cx_ratio < LIGHT_X_MIN or cx_ratio > LIGHT_X_MAX:
                return False
        else:
            min_area = MIN_BOX_AREA.get(label, DEFAULT_BOX_AREA)
            if area < min_area:
                return False
        return True
 
    # ============================================================
    # Auto mode state machine
    # ============================================================
    def update_auto_state(self, detections):
        now = time.time()
        with self.lock:
            current      = self.auto_state
            cooldown_end = self.cooldown_end
 
        if now < cooldown_end:
            return
 
        labels = [d['label'] for d in detections if d.get('valid', False)]
 
        if 'Redlight' in labels:
            if current != DriveState.RED_LIGHT:
                self._cancel_recovery()
                self._set_auto_state(DriveState.RED_LIGHT, SPEED_STOP)
                self.get_logger().info('Redlight -> Stop')
            return
 
        if current == DriveState.RED_LIGHT:
            if 'Greenlight' in labels:
                self._set_auto_state(DriveState.NORMAL, 1.0)
                with self.lock:
                    self.cooldown_end = now + RED_COOLDOWN_SEC
                self.get_logger().info('Greenlight -> Resuming')
            return
 
        if current != DriveState.NORMAL:
            return
 
        if 'Stop' in labels:
            self._set_auto_state(DriveState.STOP_SIGN, SPEED_STOP)
            with self.lock:
                self.cooldown_end = now + STOP_COOLDOWN_SEC
            self._schedule_recovery(STOP_HOLD_SEC)
            self.get_logger().info(f'Stop -> Stopping {STOP_HOLD_SEC}s')
            return
 
        if 'Yellowlight' in labels:
            self._set_auto_state(DriveState.FAST, SPEED_FAST)
            with self.lock:
                self.cooldown_end = now + SLOW_COOLDOWN_SEC
            self._schedule_recovery(FAST_HOLD_SEC)
            self.get_logger().info('Yellowlight -> Speed up')
            return
 
        slow_triggers = {'Yield', 'Roundabout'}
        if any(s in labels for s in slow_triggers):
            triggered = next(s for s in slow_triggers if s in labels)
            self._set_auto_state(DriveState.SLOW, SPEED_SLOW)
            with self.lock:
                self.cooldown_end = now + SLOW_COOLDOWN_SEC
            self._schedule_recovery(SLOW_HOLD_SEC)
            self.get_logger().info(f'{triggered} -> Slow')
            return
 
    def _set_auto_state(self, state, scale):
        with self.lock:
            self.auto_state = state
            self.auto_scale = scale
 
    def _schedule_recovery(self, delay):
        self._cancel_recovery()
        def _recover():
            self.get_logger().info('Recovering to Normal')
            self._set_auto_state(DriveState.NORMAL, 1.0)
        self.state_timer = threading.Timer(delay, _recover)
        self.state_timer.start()
 
    def _cancel_recovery(self):
        if self.state_timer and self.state_timer.is_alive():
            self.state_timer.cancel()
            self.state_timer = None
 
    # ============================================================
    # Keyboard events
    # ============================================================
    def on_press(self, key):
        if key == keyboard.Key.tab:
            self.auto_mode = not self.auto_mode
            mode_str = 'AUTO' if self.auto_mode else 'MANUAL'
            print(f'\n  *** Switched to {mode_str} mode ***\n')
            if self.auto_mode:
                self._cancel_recovery()
                with self.lock:
                    self.auto_state   = DriveState.NORMAL
                    self.auto_scale   = 1.0
                    self.cooldown_end = 0.0
            else:
                self.manual_throttle = 0.0
                self.manual_steering = 0.0
            return
 
        try:
            if hasattr(key, 'char') and key.char == 'q':
                self.stop()
                cv2.destroyAllWindows()
                rclpy.shutdown()
                return
        except AttributeError:
            pass
 
        if not self.auto_mode:
            try:
                if hasattr(key, 'char') and key.char:
                    self.keys_pressed.add(key.char.lower())
            except AttributeError:
                pass
            if key == keyboard.Key.up:      self.keys_pressed.add('up')
            elif key == keyboard.Key.down:  self.keys_pressed.add('down')
            elif key == keyboard.Key.left:  self.keys_pressed.add('left')
            elif key == keyboard.Key.right: self.keys_pressed.add('right')
            elif key == keyboard.Key.space: self.keys_pressed.add('space')
            self._update_manual_velocity()
 
    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.keys_pressed.discard(key.char.lower())
        except AttributeError:
            pass
        if key == keyboard.Key.up:      self.keys_pressed.discard('up')
        elif key == keyboard.Key.down:  self.keys_pressed.discard('down')
        elif key == keyboard.Key.left:  self.keys_pressed.discard('left')
        elif key == keyboard.Key.right: self.keys_pressed.discard('right')
        elif key == keyboard.Key.space: self.keys_pressed.discard('space')
        if not self.auto_mode:
            self._update_manual_velocity()
 
    def _update_manual_velocity(self):
        t = s = 0.0
        if 'w' in self.keys_pressed or 'up' in self.keys_pressed:
            t = LINEAR_SPEED
        elif 's' in self.keys_pressed or 'down' in self.keys_pressed:
            t = -LINEAR_SPEED
        if 'a' in self.keys_pressed or 'left' in self.keys_pressed:
            s = ANGULAR_SPEED
        elif 'd' in self.keys_pressed or 'right' in self.keys_pressed:
            s = -ANGULAR_SPEED
        if 'space' in self.keys_pressed:
            t = s = 0.0
        self.manual_throttle = t
        self.manual_steering = s
 
    # ============================================================
    # Draw frame
    # ============================================================
    def draw_frame(self, frame):
        display = frame.copy()
        h, w    = display.shape[:2]
 
        for det in self.latest_detections:
            label      = det['label']
            confidence = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            area  = det.get('area', 0)
            valid = det.get('valid', False)
            color = CLASS_COLORS.get(label, (0, 165, 255))
 
            if valid:
                # Valid detection: solid box, will trigger response
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                text = f'{label} {confidence:.2f} ({area})'
            else:
                # Invalid detection: dashed/grey box, display only
                gray = (160, 160, 160)
                cv2.rectangle(display, (x1, y1), (x2, y2), gray, 1)
                text = f'[{label}] {confidence:.2f} ({area})'
                color = gray
 
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(display, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
 
        with self.lock:
            auto_state   = self.auto_state
            auto_scale   = self.auto_scale
            cooldown_end = self.cooldown_end
 
        now           = time.time()
        cooldown_left = max(0.0, cooldown_end - now)
 
        mode_color = (0, 255, 255) if self.auto_mode else (255, 200, 0)
        mode_str   = 'AUTO' if self.auto_mode else 'MANUAL'
 
        if self.auto_mode:
            state_color = {
                DriveState.NORMAL    : (0, 255, 0),
                DriveState.SLOW      : (0, 200, 255),
                DriveState.FAST      : (0, 255, 128),
                DriveState.STOP_SIGN : (0, 0, 255),
                DriveState.RED_LIGHT : (0, 0, 255),
                DriveState.COOLDOWN  : (180, 180, 0),
            }.get(auto_state, (255, 255, 255))
            action_str = auto_state
            speed_val  = AUTO_SPEED * auto_scale
        else:
            state_color = (255, 200, 0)
            parts = []
            if self.manual_throttle > 0: parts.append('Fwd')
            if self.manual_throttle < 0: parts.append('Bwd')
            if self.manual_steering > 0: parts.append('L')
            if self.manual_steering < 0: parts.append('R')
            action_str = ' '.join(parts) if parts else 'Stop'
            speed_val  = self.manual_throttle
 
        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.42
        line_h    = 20
        lines = [
            (f'FPS:{self.fps:.0f} Mode:{mode_str}',              mode_color),
            (f'State: {action_str}',                              state_color),
            (f'Speed: {speed_val:.2f}m/s',                       (255, 255, 255)),
            (f'CD:{cooldown_left:.1f}s Signs:{len(self.latest_detections)}',
             (180, 180, 0) if cooldown_left > 0 else (200, 200, 200)),
        ]
        panel_w = 240
        panel_h = len(lines) * line_h + 10
        overlay = display.copy()
        cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        for i, (text, color) in enumerate(lines):
            cv2.putText(display, text, (10, 20 + i * line_h),
                        font, font_size, color, 1)
 
        bar_color = (0, 100, 180) if self.auto_mode else (80, 60, 0)
        cv2.rectangle(display, (0, h - 30), (w, h), bar_color, -1)
        hint = 'AUTO: sign response active' if self.auto_mode else \
               'MANUAL: W/S=Fwd/Bwd  A/D=Turn  SPACE=Stop'
        cv2.putText(display, f'[TAB] Switch | {hint} | Q=Quit',
                    (8, h - 9), font, 0.38, (255, 255, 255), 1)
 
        return display
 
    # ============================================================
    # Stop vehicle
    # ============================================================
    def stop(self):
        msg = MotorCommands()
        msg.motor_names = ['steering_angle', 'motor_throttle']
        msg.values = [0.0, 0.0]
        self.cmd_pub.publish(msg)
 
    def destroy_node(self):
        self._cancel_recovery()
        self.stop()
        self.listener.stop()
        super().destroy_node()
 
 
# ============================================================
# Entry Point
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = QCar2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nStopping...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()