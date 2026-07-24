import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool
from pymavlink import mavutil


class PixhawkLEDStatusNode(Node):
    """Watches Pixhawk state via MAVProxy's UDP stream and publishes /led_state.
    E-stop keeps the vehicle ARMED (RC option 165), so it's invisible in
    HEARTBEAT; we read the switch position from RC_CHANNELS instead."""

    def __init__(self):
        super().__init__("pixhawk_led_status")

        self.declare_parameter('endpoint', 'udpin:127.0.0.1:14550')
        self.declare_parameter('estop_channel', 7)
        endpoint = self.get_parameter('endpoint').value
        self.estop_channel = self.get_parameter('estop_channel').value

        self.led_pub = self.create_publisher(Int32, "/led_state", 10)
        self.autonomy_t = 0.0
        self.create_subscription(Bool, '/autonomy_active', self._auto_cb, 10)
        self.last_led_state = None
        self.estop_active = True   # assume killed until real data arrives
        self.armed = False
        self.mode = None

        self.get_logger().info(f"Connecting to Pixhawk on {endpoint}")
        self.master = mavutil.mavlink_connection(endpoint)
        # Non-blocking startup: don't wait_heartbeat() (it would freeze init if
        # the Pixhawk is late). Lock on in check_pixhawk_state and request the
        # RC stream then.
        self.streams_requested = False
        self.get_logger().info(
            "LED status started; waiting for Pixhawk heartbeat (non-blocking)")

        # 25 Hz RC + a 50 ms timer keep e-stop -> RED reporting under ~90 ms.
        self.timer = self.create_timer(0.05, self.check_pixhawk_state)

    def _request_streams(self):
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
            40000,  # microseconds -> 25 Hz
            0, 0, 0, 0, 0)
        self.streams_requested = True
        self.get_logger().info("Pixhawk heartbeat received")

    def _auto_cb(self, msg):
        if msg.data:
            self.autonomy_t = time.time()

    def publish_led_state(self, state):
        if state == self.last_led_state:
            return
        msg = Int32()
        msg.data = state
        self.led_pub.publish(msg)
        self.last_led_state = state
        labels = {1: "RED - Disarmed / E-stop", 2: "YELLOW - Manual", 3: "GREEN - Auto"}
        self.get_logger().info(f"LED: {labels.get(state, state)}")

    def check_pixhawk_state(self):
        while True:
            msg = self.master.recv_match(
                type=["HEARTBEAT", "RC_CHANNELS"], blocking=False)
            if msg is None:
                break
            if not self.streams_requested:
                # Lock onto the first autopilot HEARTBEAT, then request streams.
                if (msg.get_type() == "HEARTBEAT"
                        and msg.get_srcComponent()
                        == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1):
                    self.master.target_system = msg.get_srcSystem()
                    self.master.target_component = msg.get_srcComponent()
                    self._request_streams()
                else:
                    continue  # ignore GCS/other components until locked
            if msg.get_srcSystem() != self.master.target_system:
                continue
            if msg.get_type() == "HEARTBEAT":
                self.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = mavutil.mode_string_v10(msg)
            else:  # RC_CHANNELS
                pwm = getattr(msg, f"chan{self.estop_channel}_raw")
                self.estop_active = pwm < 1200  # SB low ~994 = e-stop; RC loss reads 0

        if self.mode is None:
            return  # no heartbeat yet

        if not self.armed or self.estop_active:
            self.publish_led_state(1)
        elif self.mode in ("AUTO", "GUIDED") or (time.time() - self.autonomy_t) < 1.0:
            self.publish_led_state(3)
        else:
            self.publish_led_state(2)


def main(args=None):
    rclpy.init(args=args)
    node = PixhawkLEDStatusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
