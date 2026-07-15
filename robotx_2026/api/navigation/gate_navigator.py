import math
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import depthai as dai
import rclpy
from rclpy.node import Node
from pymavlink import mavutil

BLOB = "/root/robotx_ws/src/robotx_2026/models/buoy_v16.blob"
LABELS = ["black_buoy", "black_cross", "black_target_boat", "black_triangle",
          "blue_buoy", "green_buoy", "green_light_buoy", "green_pole_buoy",
          "red_buoy", "red_light_buoy", "red_pole_buoy", "yellow_buoy",
          "yellow_target_boat"]
RED_CLASSES = {"red_buoy", "red_pole_buoy", "red_light_buoy"}
GREEN_CLASSES = {"green_buoy", "green_pole_buoy", "green_light_buoy"}

CONF_MIN = 0.60          # min confidence to use a buoy for correction
RANGE_MAX = 25.0         # m - ignore detections beyond this
PAIR_FRESH_S = 1.5       # red+green must both be seen within this window
MAX_CORRECTION = 15.0    # m - corrected point must be this close to planned
ARRIVE_RADIUS = 0.75      # m
CMD_PERIOD = 1.0         # s between guided target sends
M_PER_DEG = 111111.0
GATE_MIN_WIDTH = 1.0    # m - red/green closer than this = same buoy misread
GATE_MAX_WIDTH = 8.0    # m - farther apart than this = not a gate



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

        # --- MAVLink ---
        self.get_logger().info(f"Connecting MAVLink on {endpoint}")
        self.mav = mavutil.mavlink_connection(endpoint)
        self.mav.wait_heartbeat()
        self.get_logger().info("Heartbeat OK")

        self.waypoints = self._fetch_mission()
        if not self.waypoints:
            raise RuntimeError("No waypoints on vehicle - upload a mission in QGC first")
        self.get_logger().info(f"Fetched {len(self.waypoints)} waypoints from vehicle")
        self.wp_index = 0

        # --- vehicle state ---
        self.lat = None
        self.lon = None
        self.heading = None   # degrees
        self.mode = ""
        self.armed = False

        # --- perception state ---
        self.last_red = None     # (t, z, x)
        self.last_green = None
        self.correction = None   # (lat, lon) of gate midpoint
        self.frame = None
        self.frame_lock = threading.Lock()
        self.last_dets = []

        self._start_camera()
        self._start_stream()

        self.last_cmd = 0.0
        self.guided_set = False
        self.done = False
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
            if item and item.command == 16 and item.seq > 0:  # NAV_WAYPOINT, skip home
                wps.append((item.x / 1e7, item.y / 1e7))
        self.mav.mav.mission_ack_send(self.mav.target_system,
                                      self.mav.target_component, 0)
        return wps

    # ---------- camera ----------
    def _start_camera(self):
        p = dai.Pipeline()
        cam = p.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        cam.setIspScale(2, 3)
        cam.setPreviewSize(640, 352)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(15)
        left = p.create(dai.node.ColorCamera)
        left.setCamera("left")
        left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        left.setIspScale(1, 3)
        left.setFps(10)
        right = p.create(dai.node.ColorCamera)
        right.setCamera("right")
        right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        right.setIspScale(1, 3)
        right.setFps(10)
        stereo = p.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        left.isp.link(stereo.left)
        right.isp.link(stereo.right)
        nn = p.create(dai.node.YoloSpatialDetectionNetwork)
        nn.setBlobPath(BLOB)
        nn.setNumClasses(len(LABELS))
        nn.setCoordinateSize(4)
        nn.setIouThreshold(0.5)
        nn.setConfidenceThreshold(0.5)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        nn.setBoundingBoxScaleFactor(0.5)
        nn.setDepthLowerThreshold(300)
        nn.setDepthUpperThreshold(35000)
        cam.preview.link(nn.input)
        stereo.depth.link(nn.inputDepth)
        xo_rgb = p.create(dai.node.XLinkOut)
        xo_rgb.setStreamName("rgb")
        cam.preview.link(xo_rgb.input)
        xo_nn = p.create(dai.node.XLinkOut)
        xo_nn.setStreamName("nn")
        nn.out.link(xo_nn.input)
        self.device = dai.Device(p)
        self.q_rgb = self.device.getOutputQueue("rgb", 4, False)
        self.q_nn = self.device.getOutputQueue("nn", 4, False)

    # ---------- main loop ----------
    def tick(self):
        self._drain_mavlink()
        self._drain_camera()
        if self.done or self.lat is None or self.heading is None:
            return

        if self.mode not in ("GUIDED",):
            return  # pilot took over; hold fire until back in GUIDED

        self._update_correction()

        planned = self.waypoints[self.wp_index]
        target = self.correction if self.correction else planned

        if dist_m(self.lat, self.lon, *target) < ARRIVE_RADIUS:
            self.get_logger().info(f"Waypoint {self.wp_index + 1} reached")
            self.wp_index += 1
            self.correction = None
            self.last_red = self.last_green = None
            if self.wp_index >= len(self.waypoints):
                self.get_logger().info("Mission complete - HOLD")
                self.mav.set_mode('HOLD')
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
        dets = self.q_nn.tryGet()
        if dets is not None:
            self.last_dets = dets.detections
            if self.lat is not None and self.heading is not None:
                now = time.time()
                best = {"red": None, "green": None}   # nearest valid per color, this frame
                for d in dets.detections:
                    name = LABELS[d.label]
                    z = d.spatialCoordinates.z / 1000.0
                    x = d.spatialCoordinates.x / 1000.0
                    if d.confidence < CONF_MIN or z <= 0.3 or z > RANGE_MAX:
                        continue
                    col = ("red" if name in RED_CLASSES else
                           "green" if name in GREEN_CLASSES else None)
                    if col and (best[col] is None or z < best[col][0]):
                        best[col] = (z, x)
                if best["red"]:
                    self.last_red = (now, self._body_to_world(*best["red"]))
                if best["green"]:
                    self.last_green = (now, self._body_to_world(*best["green"]))
        f = self.q_rgb.tryGet()
        if f is not None:
            frame = f.getCvFrame()
            self._annotate(frame)
            with self.frame_lock:
                self.frame = frame

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
            f"red@({self.last_red[1][0]:.7f},{self.last_red[1][1]:.7f}) "
            f"green@({self.last_green[1][0]:.7f},{self.last_green[1][1]:.7f}) "
            f"sep {sep:.1f}m mid->({cand[0]:.7f},{cand[1]:.7f})",
            throttle_duration_sec=1.0)

    def _send_target(self, lat, lon):
        self.mav.mav.set_position_target_global_int_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b110111111000,  # position only
            int(lat * 1e7), int(lon * 1e7), 0,
            0, 0, 0, 0, 0, 0, 0, 0)
        tag = "corrected" if self.correction else "planned"
        self.get_logger().info(
            f"WP {self.wp_index + 1}/{len(self.waypoints)} ({tag}) "
            f"dist {dist_m(self.lat, self.lon, lat, lon):.1f}m")

    # ---------- debug stream ----------
    def _annotate(self, frame):
        h, w = frame.shape[:2]
        for d in self.last_dets:
            name = LABELS[d.label]
            color = ((0, 255, 0) if name in GREEN_CLASSES else
                     (0, 0, 255) if name in RED_CLASSES else (0, 255, 255))
            x1, y1 = int(d.xmin * w), int(d.ymin * h)
            x2, y2 = int(d.xmax * w), int(d.ymax * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{name} {d.spatialCoordinates.z/1000.0:.1f}m",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        status = (f"WP {min(self.wp_index + 1, len(self.waypoints))}"
                  f"/{len(self.waypoints)} "
                  f"{'CORRECTED' if self.correction else 'planned'} "
                  f"mode {self.mode}")
        cv2.putText(frame, status, (6, 20),
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
                        time.sleep(0.08)
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
