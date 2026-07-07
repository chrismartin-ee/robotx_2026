import serial
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from threading import Lock


class ArduinoNano:
    def __init__(self, port, baudrate=115200):
        self.serial_conn = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Arduino resets on connect; wait for it to boot

    def send_command(self, msg):
        if isinstance(msg, bytes):
            data = msg
        else:
            data = f"{msg}\n".encode("ascii")
        self.serial_conn.write(data)
        self.serial_conn.flush()

    def close(self):
        if self.serial_conn.is_open:
            self.serial_conn.close()


class LEDNode(Node):
    """Forwards /led_state (Int32 0-3) to the LED Arduino over serial.
    0=off, 1=red (disarmed/e-stop), 2=yellow (manual), 3=green (auto).
    Reconnects automatically if the serial link drops (e.g. EMI events
    knock the CH340 off the bus and it re-enumerates)."""

    def __init__(self):
        super().__init__('led_node')

        self.declare_parameter('port', '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0')
        self.declare_parameter('baud', 115200)
        self.port = self.get_parameter('port').value
        self.baud = self.get_parameter('baud').value

        self.LED = None
        self.last_state = None
        self.lock = Lock()

        self._connect()

        self.request_sub = self.create_subscription(
            Int32, '/led_state', self.request_callback, 10)
        # manual test: ros2 topic pub /led_state std_msgs/msg/Int32 "{data: 1}" --once

        self.reconnect_timer = self.create_timer(3.0, self._reconnect_check)

    def _connect(self):
        try:
            self.LED = ArduinoNano(self.port, self.baud)
            self.get_logger().info(f"LED Arduino connected on {self.port}")
            return True
        except Exception as e:
            self.LED = None
            self.get_logger().warn(f"LED serial connect failed: {e}")
            return False

    def _mark_dead(self):
        try:
            self.LED.close()
        except Exception:
            pass
        self.LED = None
        self.get_logger().warn("LED serial link lost - will retry every 3s")

    def _reconnect_check(self):
        if self.LED is not None:
            return
        if self._connect() and self.last_state is not None:
            # restore the color that should be showing
            self._send(self.last_state)

    def _send(self, state):
        try:
            with self.lock:
                self.LED.send_command(state)
            return True
        except Exception:
            self._mark_dead()
            return False

    def request_callback(self, msg):
        state = msg.data
        if state not in (0, 1, 2, 3):
            self.get_logger().warn(f"Invalid LED state: {state}")
            return
        self.last_state = state
        labels = {0: "off", 1: "RED disarmed/e-stop", 2: "YELLOW manual", 3: "GREEN auto"}
        self.get_logger().info(f"LED -> {labels[state]}")
        if self.LED is None:
            self.get_logger().warn("LED not connected - state saved, will apply on reconnect")
            return
        self._send(state)

    def destroy_node(self):
        if self.LED is not None:
            try:
                self.LED.send_command(0)
                self.LED.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LEDNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
