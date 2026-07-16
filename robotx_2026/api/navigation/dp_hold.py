import math
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import cv2
import depthai as dai
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from pymavlink import mavutil
from ultralytics import YOLO

ENGINE = "/root/robotx_ws/src/robotx_2026/models/buoy_v16.engine"
LABELS = ["black_buoy", "black_cross", "black_target_boat", "black_triangle",
          "blue_buoy", "green_buoy", "green_light_buoy", "green_pole_buoy",
          "red_buoy", "red_light_buoy", "red_pole_buoy", "yellow_buoy",
          "yellow_target_boat"]
TARGET_CLASSES = {"green_buoy", "green_pole_buoy"}   # what to lock onto

TARGET_YAW = 240.0     # deg - hardcoded hold heading
TARGET_Z = 2.0         # m  - hold buoy this far ahead
TARGET_X = 0.0         # m  - buoy centered
CONF_MIN = 0.5
LOST_TIMEOUT = 1.0     # s without target -> neutral controls

KP_YAW = 4.0
KD_YAW = 3.0    # PWM per deg/s of rotation - the damper
KP_FWD = 90.0         # PWM per meter of forward error
KP_LAT = 90.0         # PWM per meter of lateral error
PWM_LIMIT = 120        # max deflection from 1500
CTRL_HZ = 10.0
DEAD_YAW = 5.0     # deg - ignore smaller yaw errors
DEAD_POS = 0.15    # m - ignore smaller position errors
GRACE_S = 20.0

def clamp(v, lim=PWM_LIMIT):
    return int(max(-lim, min(lim, v)))


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


class DPHold(Node):
    def __init__(self):
        super().__init__("dp_hold")
        self.mav = mavutil.mavlink_connection('udpin:127.0.0.1:14551')
        self.mav.wait_heartbeat()
        self.get_logger().info("Heartbeat OK")
        self.auto_pub = self.create_publisher(Bool, '/autonomy_active', 10)
        self.heading = None
        self.mode = ""
        self.target = None        # (t, z, x)
        self.hold_yaw = 240.0
        self.yaw_rate = 0.0
        self.engaged = True
        self.last_seen = float('inf')
        self.frame = None
        self.lock = threading.Lock()
        self.depth_img = None
        self.fx = self.cx = None

        self.model = YOLO(ENGINE, task="detect")
        self._start_camera()
        self._start_stream()
        self.get_logger().info(
            f"DP hold up: yaw {TARGET_YAW} deg, buoy at {TARGET_Z}m ahead")
        self.timer = self.create_timer(1.0 / CTRL_HZ, self.tick)

    def _start_camera(self):
        p = dai.Pipeline()
        cam = p.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        cam.setIspScale(2, 3)
        cam.setPreviewSize(640, 352)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(30)
        left = p.create(dai.node.ColorCamera)
        left.setCamera("left")
        left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        left.setIspScale(1, 3)
        left.setFps(30)
        right = p.create(dai.node.ColorCamera)
        right.setCamera("right")
        right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        right.setIspScale(1, 3)
        right.setFps(30)
        stereo = p.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        left.isp.link(stereo.left)
        right.isp.link(stereo.right)
        xr = p.create(dai.node.XLinkOut); xr.setStreamName("rgb")
        cam.preview.link(xr.input)
        xd = p.create(dai.node.XLinkOut); xd.setStreamName("depth")
        stereo.depth.link(xd.input)
        self.device = dai.Device(p)
        self.q_rgb = self.device.getOutputQueue("rgb", 4, False)
        self.q_depth = self.device.getOutputQueue("depth", 4, False)

    def _depth_at(self, u, v, half=8):
        dh, dw = self.depth_img.shape[:2]
        s = dw / 640.0
        off = (dh / s - 352.0) / 2.0
        du, dv = int(u * s), int((v + off) * s)
        roi = self.depth_img[max(0, dv-half):dv+half+1,
                             max(0, du-half):du+half+1].astype(np.float32)
        valid = roi[roi > 0]
        if valid.size == 0:
            return None, du
        return float(np.median(valid)) / 1000.0, du

    def tick(self):
        while True:
            m = self.mav.recv_match(blocking=False)
            if m is None:
                break
            t = m.get_type()
            if t == "GLOBAL_POSITION_INT":
                self.heading = m.hdg / 100.0
            elif t == "HEARTBEAT" and m.get_srcSystem() == self.mav.target_system:
                self.mode = mavutil.mode_string_v10(m)
            elif t == "ATTITUDE":
                self.yaw_rate = math.degrees(m.yawspeed)

        self._see()

        now = time.time()
        fresh = (self.target is not None and
                 now - self.target[0] < LOST_TIMEOUT)
        if fresh:
            self.engaged = True
            self.last_seen = now
        if (self.mode != "MANUAL" or self.heading is None or
                not self.engaged or now - self.last_seen > GRACE_S):
            self.engaged = False
            self._override(0, 0, 0, release=True)
            self.auto_pub.publish(Bool(data=False))
            return
        self.auto_pub.publish(Bool(data=True))

        yaw_err = wrap180(self.hold_yaw - self.heading)
        if abs(yaw_err) < DEAD_YAW:
            yaw_err = 0.0
        steer = clamp(KP_YAW * yaw_err - KD_YAW * self.yaw_rate)

        if fresh:
            _, z, x = self.target
            fwd_err = z - TARGET_Z
            lat_err = x - TARGET_X
            if abs(fwd_err) < DEAD_POS:
                fwd_err = 0.0
            if abs(lat_err) < DEAD_POS:
                lat_err = 0.0
            self._override(steer, clamp(KP_FWD * fwd_err),
                           clamp(KP_LAT * lat_err))
        else:
            self._override(steer, 0, 0)   # lost sight: hold yaw, wait
        self.get_logger().info(
            f"yaw_err {yaw_err:+.0f} lock {'Y' if fresh else 'lost'}",
            throttle_duration_sec=1.0)

    def _see(self):
        d = self.q_depth.tryGet()
        if d is not None:
            self.depth_img = d.getFrame()
            if self.fx is None:
                M = self.device.readCalibration().getCameraIntrinsics(
                    dai.CameraBoardSocket.CAM_A,
                    self.depth_img.shape[1], self.depth_img.shape[0])
                self.fx, self.cx = M[0][0], M[0][2]
        fr = None
        while True:
            f = self.q_rgb.tryGet()
            if f is None:
                break
            fr = f
        if fr is None:
            return
        img = fr.getCvFrame()
        res = self.model.predict(img, verbose=False, imgsz=(352, 640),
                                 conf=CONF_MIN)[0]
        best = None
        if self.depth_img is not None and self.fx is not None:
            for b in res.boxes:
                name = LABELS[int(b.cls[0])]
                if name not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                z, du = self._depth_at((x1+x2)//2, (y1+y2)//2)
                if z is None or z <= 0.3:
                    continue
                x_m = (du - self.cx) * z / self.fx
                if best is None or z < best[0]:
                    best = (z, x_m, (x1, y1, x2, y2), float(b.conf[0]))
        if best is not None:
            self.target = (time.time(), best[0], best[1])
            x1, y1, x2, y2 = best[2]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"LOCK {best[0]:.2f}m {best[1]:+.2f}m {int(best[3]*100)}%",
                        (x1, max(y1-6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
        hdg = f"{self.heading:.0f}" if self.heading is not None else "?"
        cv2.putText(img, f"mode {self.mode} hdg {hdg} tgt {TARGET_YAW:.0f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        with self.lock:
            self.frame = img

    def _override(self, steer, throttle, lateral, release=False):
        if release:
            vals = [0] * 8      # release all channels to the radio
        else:
            vals = [1489 + steer, 0, 1495 + throttle, 1495 + lateral,
                    0, 0, 0, 0]
        self.mav.mav.rc_channels_override_send(
            self.mav.target_system, self.mav.target_component, *vals)

    def _start_stream(self):
        node = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with node.lock:
                            fr = None if node.frame is None else node.frame.copy()
                        if fr is None:
                            time.sleep(0.1)
                            continue
                        ok, jpg = cv2.imencode(".jpg", fr,
                                               [cv2.IMWRITE_JPEG_QUALITY, 60])
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpg.tobytes())
                            self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *a):
                pass

        srv = HTTPServer(("0.0.0.0", 8080), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = DPHold()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    try:
        node._override(0, 0, 0, release=True)   # give sticks back on exit
    except Exception:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
