import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

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

        self.last_led_state = None
        self.estop_active = True   # assume killed until real data arrives
        self.armed = False
        self.mode = None

        self.get_logger().info(f"Connecting to Pixhawk on {endpoint}")
        self.master = mavutil.mavlink_connection(endpoint)
        self.master.wait_heartbeat()
        self.get_logger().info("Pixhawk heartbeat received")

        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS,
            200000,  # microseconds -> 5 Hz
            0, 0, 0, 0, 0)

        self.timer = self.create_timer(0.2, self.check_pixhawk_state)

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
        elif self.mode in ("AUTO", "GUIDED"):
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
