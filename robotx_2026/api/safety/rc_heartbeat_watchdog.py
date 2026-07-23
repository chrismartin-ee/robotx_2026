import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from pymavlink import mavutil


class RCHeartbeatWatchdog(Node):
    """Independent software failsafe that stops the boat when the remote
    controller (RC transmitter) link dies, WITHOUT anyone pressing the physical
    kill switch.

    The RC link is the controller's "heartbeat." We watch its health through
    the Pixhawk's MAVLink stream (rebroadcast by MAVProxy). Evidence the link
    is alive is either of:
      - an RC_CHANNELS message whose monitored channel sits at a valid PWM
        (on this hardware ArduPilot reports 0 for that channel when the
        transmitter is out of range or powered off), or
      - a SYS_STATUS message reporting the RC receiver sensor healthy.
    The receiver going unhealthy in SYS_STATUS is also treated as an immediate
    loss, so a single stale signal can never mask a real dropout (a missed kill
    is worse than a false one).

    If nothing confirms the link for `heartbeat_timeout` seconds while the
    vehicle is ARMED, the watchdog force-disarms (motors off, same as the kill
    button) and LATCHES: it keeps the vehicle disarmed until an operator calls
    the `~/reset` service AND the RC link has recovered.

    This complements, and does not replace, ArduPilot's own FS_THR / FS_GCS
    failsafes -- run both for defense in depth.
    """

    # Magic value ArduPilot requires in param2 to force-disarm even while the
    # vehicle is moving.
    FORCE_DISARM_MAGIC = 21196

    def __init__(self):
        super().__init__("rc_heartbeat_watchdog")

        self.declare_parameter("endpoint", "udpin:127.0.0.1:14552")
        self.declare_parameter("heartbeat_timeout", 1.0)
        self.declare_parameter("rc_channel", 7)        # RC7 = arm/e-stop
        self.declare_parameter("min_valid_pwm", 900)   # below this = no pulses
        self.declare_parameter("use_rc_health_bit", True)
        self.declare_parameter("require_armed", True)
        self.declare_parameter("link_timeout", 3.0)    # Pixhawk MAVLink silence

        endpoint = self.get_parameter("endpoint").value
        self.heartbeat_timeout = float(self.get_parameter("heartbeat_timeout").value)
        self.rc_channel = int(self.get_parameter("rc_channel").value)
        self.min_valid_pwm = int(self.get_parameter("min_valid_pwm").value)
        self.use_rc_health_bit = bool(self.get_parameter("use_rc_health_bit").value)
        self.require_armed = bool(self.get_parameter("require_armed").value)
        self.link_timeout = float(self.get_parameter("link_timeout").value)

        self.kill_pub = self.create_publisher(Bool, "/kill_active", 10)
        self.reset_srv = self.create_service(Trigger, "~/reset", self._reset_cb)

        self.get_logger().info(f"Connecting to Pixhawk on {endpoint}")
        self.master = mavutil.mavlink_connection(endpoint)
        self.master.wait_heartbeat()
        self.get_logger().info("Pixhawk heartbeat received; RC watchdog active")

        # Make sure the streams we depend on are actually flowing.
        for msg_id, rate_hz in (
            (mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, 5),
            (mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS, 4),
        ):
            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                msg_id, int(1e6 / rate_hz), 0, 0, 0, 0, 0)

        # State
        now = time.monotonic()
        self.armed = False
        self.killed = False
        self.last_rc_ok = now          # last valid monitored-channel PWM
        self.last_sys_status = 0.0
        self.rc_health_bad = False     # receiver present but unhealthy
        self.last_vehicle_hb = 0.0
        self.last_disarm_sent = 0.0
        self.last_kill_published = None
        self.last_status_log = 0.0

        self.timer = self.create_timer(0.1, self.tick)

    # ---------------- MAVLink intake ----------------
    def _drain(self):
        now = time.monotonic()
        while True:
            msg = self.master.recv_match(
                type=["HEARTBEAT", "RC_CHANNELS", "SYS_STATUS"], blocking=False)
            if msg is None:
                break
            if msg.get_srcSystem() != self.master.target_system:
                continue

            mtype = msg.get_type()
            if mtype == "HEARTBEAT":
                self.last_vehicle_hb = now
                self.armed = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            elif mtype == "RC_CHANNELS":
                pwm = getattr(msg, f"chan{self.rc_channel}_raw", 0)
                if pwm >= self.min_valid_pwm:
                    self.last_rc_ok = now
            elif mtype == "SYS_STATUS":
                self.last_sys_status = now
                bit = mavutil.mavlink.MAV_SYS_STATUS_SENSOR_RC_RECEIVER
                present = bool(msg.onboard_control_sensors_present & bit)
                healthy = bool(msg.onboard_control_sensors_health & bit)
                self.rc_health_bad = present and not healthy

    def _link_lost(self, now):
        """True when the RC link should be considered dead."""
        rc_timed_out = (now - self.last_rc_ok) > self.heartbeat_timeout
        health_bad = (
            self.use_rc_health_bit
            and (now - self.last_sys_status) < 2.0 * self.heartbeat_timeout
            and self.rc_health_bad
        )
        return rc_timed_out or health_bad

    # ---------------- Watchdog loop ----------------
    def tick(self):
        self._drain()
        now = time.monotonic()

        mavlink_ok = (now - self.last_vehicle_hb) < self.link_timeout
        link_lost = self._link_lost(now)

        if self.killed:
            # Latched: keep enforcing the kill for as long as it is engaged.
            if self.armed and mavlink_ok:
                self._disarm("re-enforcing latched kill")
            self._publish_kill(True)
            self._status_log(now, link_lost, mavlink_ok)
            return

        if not link_lost:
            self._publish_kill(False)
            self._status_log(now, link_lost, mavlink_ok)
            return

        # RC link is down.
        if self.require_armed and not self.armed:
            # Nothing to stop, and we do not want to latch a kill on a boat
            # that is already safe on the bench.
            self._publish_kill(False)
            self._status_log(now, link_lost, mavlink_ok)
            return

        if not mavlink_ok:
            # We have lost the Pixhawk link too and cannot command a disarm;
            # ArduPilot's onboard failsafe must handle this case.
            self.get_logger().error(
                "RC link lost AND no MAVLink to Pixhawk - cannot force-disarm "
                "from software; relying on ArduPilot failsafe",
                throttle_duration_sec=2.0)
            return

        # Trigger the latched kill.
        self.killed = True
        self.get_logger().error(
            f"RC heartbeat lost for >{self.heartbeat_timeout:.1f}s "
            "-> FORCE-DISARMING (latched). Call ~/reset to recover.")
        self._disarm("RC heartbeat lost")
        self._publish_kill(True)

    def _disarm(self, reason):
        now = time.monotonic()
        if now - self.last_disarm_sent < 0.4:
            return
        self.last_disarm_sent = now
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            0,                        # param1: 0 = disarm
            self.FORCE_DISARM_MAGIC,  # param2: force even while moving
            0, 0, 0, 0, 0)
        self.get_logger().warn(
            f"Force-disarm sent ({reason})", throttle_duration_sec=1.0)

    def _publish_kill(self, active):
        if active == self.last_kill_published:
            return
        self.kill_pub.publish(Bool(data=active))
        self.last_kill_published = active

    def _status_log(self, now, link_lost, mavlink_ok):
        if now - self.last_status_log < 5.0:
            return
        self.last_status_log = now
        self.get_logger().info(
            f"armed={self.armed} rc_lost={link_lost} killed={self.killed} "
            f"mavlink_ok={mavlink_ok} rc_age={now - self.last_rc_ok:.1f}s",
            throttle_duration_sec=5.0)

    # ---------------- Reset service ----------------
    def _reset_cb(self, request, response):
        if self._link_lost(time.monotonic()):
            response.success = False
            response.message = (
                "Refused: RC link still down. Restore the transmitter first.")
            self.get_logger().warn("Kill reset refused - RC link still down")
            return response
        self.killed = False
        self._publish_kill(False)
        response.success = True
        response.message = "Kill latch reset. Re-arm from the transmitter/GCS."
        self.get_logger().info("Kill latch reset by operator")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RCHeartbeatWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
