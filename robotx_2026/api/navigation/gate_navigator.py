import math
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import cv2
import depthai as dai
import rclpy
from rclpy.node import Node
from pymavlink import mavutil
from ultralytics import YOLO

ENGINE = "/root/robotx_ws/src/robotx_2026/models/buoy_v16.engine"
LABELS = ["black_buoy", "black_cross", "black_target_boat", "black_triangle",
          "blue_buoy", "green_buoy", "green_light_buoy", "green_pole_buoy",
          "red_buoy", "red_light_buoy", "red_pole_buoy", "yellow_buoy",
          "yellow_target_boat"]
RED_CLASSES = {"red_buoy", "red_pole_buoy", "red_light_buoy"}
GREEN_CLASSES = {"green_buoy", "green_pole_buoy", "green_light_buoy"}

CONF_MIN = 0.60          # min confidence for a buoy to drive corrections
CONF_SHOW = 0.40         # min confidence to display on the stream
RANGE_MAX = 25.0         # m
PAIR_FRESH_S = 1.5       # red+green must both be seen within this window
GATE_MIN_WIDTH = 1.0     # m - closer = same buoy misread as two colors
GATE_MAX_WIDTH = 8.0     # m - farther = not a gate
MAX_CORRECTION = 15.0    # m - corrected point must be near planned waypoint
ARRIVE_RADIUS = 1.0      # m - pool scale; enlarge for the lake
CMD_PERIOD = 1.0         # s between guided target sends
M_PER_DEG = 111111.0


def offset_latlon(lat, lon, north_m, east_m):
    return (lat + north_m / M_PER_DEG,
            lon + east_m / (M_PER_DEG * math.cos(math.radians(lat))))


def dist_m(lat1, lon1, lat2, lon2):
    dn = (lat2 - lat1) * M_PER_DEG
    de = (lon2 - lon1) * M_PER_DEG * math.cos(math.radians(lat1))
    return math.hypot(dn, de)


class GateNavigator(Node):
    def __init__(self):
        super().__init__("gate_navigator")

        self.declare_parameter('endpoint', 'udpin:127.0.0.1:14551')
        endpoint = self.get_parameter('endpoint').value

        self.get_logger().info(f"Connecting MAVLink on {endpoint}")
        self.mav = mavutil.mavlink_connection(endpoint)
        self.mav.wait_heartbeat()
        self.get_logger().info("Heartbeat OK")

        self.waypoints = self._fetch_mission()
        if not self.waypoints:
            raise RuntimeError("No waypoints on vehicle - upload a mission in QGC first")
        self.get_logger().info(f"Fetched {len(self.waypoints)} waypoints from vehicle")
        self.wp_index = 0

        self.lat = None
        self.lon = None
        self.heading = None
        self.mode = ""
        self.armed = False

        self.last_red = None      # (t, (lat, lon))
        self.last_green = None
        self.correction = None
        self.frame = None
        self.frame_lock = threading.Lock()

        self.depth_img = None
        self.fx = None
        self.cx = None

        self.get_logger().info("Loading TensorRT engine...")
        self.model = YOLO(ENGINE, task="detect")
        self._start_camera()
        self._start_stream()
        self.get_logger().info("Camera + GPU model up")

        self.last_cmd = 0.0
        self.done = False
        self.fps = 0.0
        self._t_prev = time.time()
        self.timer = self.create_timer(0.05, self.tick)

    # ---------- mission fetch ----------
    def _fetch_mission(self):
        self.mav.mav.mission_request_list_send(
            self.mav.target_system, self.mav.target_component)
        msg = self.mav.recv_match(type='MISSION_COUNT', blocking=True, timeout=5)
        if msg is None:
            return []
        wps = []
        for i in range(msg.count):
            self.mav.mav.mission_request_int_send(
                self.mav.target_system, self.mav.target_component, i)
            item = self.mav.recv_match(type='MISSION_ITEM_INT',
                                       blocking=True, timeout=5)
            if item and item.command == 16 and item.seq > 0:
                wps.append((item.x / 1e7, item.y / 1e7))
        self.mav.mav.mission_ack_send(self.mav.target_system,
                                      self.mav.target_component, 0)
        return wps

    # ---------- camera: RGB + depth only, inference on GPU ----------
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
        xo_rgb = p.create(dai.node.XLinkOut)
        xo_rgb.setStreamName("rgb")
        cam.preview.link(xo_rgb.input)
        xo_d = p.create(dai.node.XLinkOut)
        xo_d.setStreamName("depth")
        stereo.depth.link(xo_d.input)
        self.device = dai.Device(p)
        self.q_rgb = self.device.getOutputQueue("rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)

    def _depth_at(self, u, v, half=8):
        """Median depth (m) around preview pixel (u,v); returns (z, du, dv)."""
        dh, dw = self.depth_img.shape[:2]
        s = dw / 640.0
        crop_off = (dh / s - 352.0) / 2.0
        du, dv = int(u * s), int((v + crop_off) * s)
        u0, u1 = max(0, du - half), min(dw, du + half + 1)
        v0, v1 = max(0, dv - half), min(dh, dv + half + 1)
        roi = self.depth_img[v0:v1, u0:u1].astype(np.float32)
        valid = roi[roi > 0]
        if valid.size == 0:
            return None, du, dv
        return float(np.median(valid)) / 1000.0, du, dv

    # ---------- main loop ----------
    def tick(self):
        self._drain_mavlink()
        self._drain_camera()
        if self.done or self.lat is None or self.heading is None:
            return
        if self.mode not in ("GUIDED",):
            return

        self._update_correction()

        planned = self.waypoints[self.wp_index]
        target = self.correction if self.correction else planned

        if dist_m(self.lat, self.lon, *target) < ARRIVE_RADIUS:
            self.get_logger().info(f"Waypoint {self.wp_index + 1} reached")
            self.wp_index += 1
            self.correction = None
            self.last_red = self.last_green = None
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info("Mission complete - LOITER")
                self.mav.set_mode('LOITER')
                self.done = True
            return

        now = time.time()
        if now - self.last_cmd > CMD_PERIOD:
            self._send_target(*target)
            self.last_cmd = now

    def _drain_mavlink(self):
        while True:
            m = self.mav.recv_match(blocking=False)
            if m is None:
                break
            t = m.get_type()
            if t == "GLOBAL_POSITION_INT":
                self.lat = m.lat / 1e7
                self.lon = m.lon / 1e7
                self.heading = m.hdg / 100.0
            elif t == "HEARTBEAT" and m.get_srcSystem() == self.mav.target_system:
                self.armed = bool(m.base_mode &
                                  mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self.mode = mavutil.mode_string_v10(m)

    def _drain_camera(self):
        d = self.q_depth.tryGet()
        if d is not None:
            self.depth_img = d.getFrame()
            if self.fx is None:
                calib = self.device.readCalibration()
                M = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,
                                              self.depth_img.shape[1],
                                              self.depth_img.shape[0])
                self.fx, self.cx = M[0][0], M[0][2]

        frame = None
        while True:
            f = self.q_rgb.tryGet()
            if f is None:
                break
            frame = f
        if frame is None:
            return
        img = frame.getCvFrame()

        dets = []
        res = self.model.predict(img, verbose=False, imgsz=(352, 640),
                                 conf=CONF_SHOW)[0]
        if self.depth_img is not None and self.fx is not None:
            for b in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                name = LABELS[int(b.cls[0])]
                conf = float(b.conf[0])
                z, du, dv = self._depth_at((x1 + x2) // 2, (y1 + y2) // 2)
                x_m = None
                if z is not None:
                    x_m = (du - self.cx) * z / self.fx
                dets.append({"box": (x1, y1, x2, y2), "name": name,
                             "conf": conf, "z": z, "x": x_m})

        # nearest confident red + green in THIS frame -> world anchors
        if self.lat is not None and self.heading is not None:
            now = time.time()
            best = {"red": None, "green": None}
            for det in dets:
                if (det["conf"] < CONF_MIN or det["z"] is None or
                        det["x"] is None or det["z"] <= 0.3 or
                        det["z"] > RANGE_MAX):
                    continue
                col = ("red" if det["name"] in RED_CLASSES else
                       "green" if det["name"] in GREEN_CLASSES else None)
                if col and (best[col] is None or det["z"] < best[col]["z"]):
                    best[col] = det
            if best["red"]:
                self.last_red = (now, self._body_to_world(best["red"]["z"],
                                                          best["red"]["x"]))
            if best["green"]:
                self.last_green = (now, self._body_to_world(best["green"]["z"],
                                                            best["green"]["x"]))

        self._annotate(img, dets)
        with self.frame_lock:
            self.frame = img

    def _body_to_world(self, z, x):
        th = math.radians(self.heading)
        north = z * math.cos(th) - x * math.sin(th)
        east = z * math.sin(th) + x * math.cos(th)
        return offset_latlon(self.lat, self.lon, north, east)

    def _update_correction(self):
        now = time.time()
        if (self.last_red is None or self.last_green is None or
                now - self.last_red[0] > PAIR_FRESH_S or
                now - self.last_green[0] > PAIR_FRESH_S):
            return
        rlat, rlon = self.last_red[1]
        glat, glon = self.last_green[1]
        sep = dist_m(rlat, rlon, glat, glon)
        if sep < GATE_MIN_WIDTH or sep > GATE_MAX_WIDTH:
            return
        cand = ((rlat + glat) / 2, (rlon + glon) / 2)
        if dist_m(*cand, *self.waypoints[self.wp_index]) > MAX_CORRECTION:
            return
        if self.correction is None:
            self.correction = cand
        else:
            self.correction = ((self.correction[0] + cand[0]) / 2,
                               (self.correction[1] + cand[1]) / 2)
        self.get_logger().info(
            f"red@({rlat:.7f},{rlon:.7f}) green@({glat:.7f},{glon:.7f}) "
            f"sep {sep:.1f}m mid->({cand[0]:.7f},{cand[1]:.7f})",
            throttle_duration_sec=1.0)

    def _send_target(self, lat, lon):
        self.mav.mav.set_position_target_global_int_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000,
            int(lat * 1e7), int(lon * 1e7), 0,
            0, 0, 0, 0, 0, 0, 0, 0)
        tag = "corrected" if self.correction else "planned"
        self.get_logger().info(
            f"WP {self.wp_index + 1}/{len(self.waypoints)} ({tag}) "
            f"dist {dist_m(self.lat, self.lon, lat, lon):.1f}m")

    # ---------- debug stream ----------
    def _annotate(self, img, dets):
        for det in dets:
            x1, y1, x2, y2 = det["box"]
            name = det["name"]
            color = ((0, 255, 0) if name in GREEN_CLASSES else
                     (0, 0, 255) if name in RED_CLASSES else (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {int(det['conf']*100)}%"
            if det["z"] is not None and det["x"] is not None:
                label += f" | {det['z']:.1f}m {det['x']:+.1f}m"
            cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - self._t_prev, 1e-3))
        self._t_prev = now
        status = (f"{self.fps:.0f}fps  "
                  f"WP {min(self.wp_index + 1, len(self.waypoints))}"
                  f"/{len(self.waypoints)} "
                  f"{'CORRECTED' if self.correction else 'planned'} "
                  f"mode {self.mode}")
        cv2.putText(img, status, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

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
                        with node.frame_lock:
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
    node = GateNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
