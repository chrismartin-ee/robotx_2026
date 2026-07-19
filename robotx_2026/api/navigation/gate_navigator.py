#!/usr/bin/env python3

import math
import time
import threading
from collections import deque
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from statistics import median

import cv2
import depthai as dai
import numpy as np
import rclpy
from pymavlink import mavutil
from rclpy.node import Node
from ultralytics import YOLO


ENGINE = "/root/robotx_ws/src/robotx_2026/models/buoy_v16.engine"

LABELS = [
    "black_buoy",
    "black_cross",
    "black_target_boat",
    "black_triangle",
    "blue_buoy",
    "green_buoy",
    "green_light_buoy",
    "green_pole_buoy",
    "red_buoy",
    "red_light_buoy",
    "red_pole_buoy",
    "yellow_buoy",
    "yellow_target_boat",
]

RED_CLASSES = {"red_buoy", "red_pole_buoy", "red_light_buoy"}
GREEN_CLASSES = {"green_buoy", "green_pole_buoy", "green_light_buoy"}

# ---------------- Detection and gate settings ----------------
CONF_MIN = 0.60
CONF_SHOW = 0.40
RANGE_MIN = 0.30
RANGE_MAX = 25.0

GATE_WP_INDICES = {0, 1}    # zero-based waypoints treated as gates (wp1, wp2)
GATE_MIN_WIDTH = 1.0        # meters
GATE_MAX_WIDTH = 8.0        # meters
PAIR_MAX_DEPTH_DIFF = 5.0   # meters; rejects unlikely red/green pairings
PAIR_MIN_PIXEL_SEP = 12     # pixels; rejects duplicate boxes on one buoy
MAX_CORRECTION = 15.0       # midpoint must be this close to planned waypoint
MIDPOINT_SHIFT_RATIO = 0.5   # shift target toward the RIGHT-side buoy by this fraction of the gap

PAIR_CONFIRM_FRAMES = 6
PAIR_SAMPLE_TIMEOUT = 0.60  # clear samples if detections stop being consecutive
PAIR_MAX_SPREAD = 1.00      # maximum world-position spread before locking

# The boat first drives to the gate midpoint, then to a point beyond the gate.
# This prevents it from turning toward waypoint 2 while still between the buoys.
GATE_EXIT_DISTANCE = 0.2
MIDPOINT_SWITCH_RADIUS = 1.0
EXIT_ARRIVE_RADIUS = 1.0
NORMAL_WP_ARRIVE_RADIUS = 1.0

# Positive yaw offset rotates camera measurements clockwise/right relative to
# the Pixhawk heading. Leave at 0 if the OAK-D points exactly straight ahead.
CAMERA_YAW_OFFSET_DEG = 0.0

# Optional camera position relative to the GPS/Pixhawk reference point.
# Positive forward is toward the bow; positive right is starboard.
CAMERA_FORWARD_OFFSET_M = 0.0
CAMERA_RIGHT_OFFSET_M = 0.0

CMD_PERIOD = 1.0
M_PER_DEG = 111111.0

# ArduPilot's documented position-only SET_POSITION_TARGET_GLOBAL_INT mask.
POSITION_ONLY_TYPE_MASK = 0b110111111100  # 3580

# Camera/depth output dimensions. 1920x1200 RGB scaled by 1/3 is 640x400,
# matching the OAK-D 400p stereo stream without aspect-ratio cropping.
CAM_WIDTH = 640
CAM_HEIGHT = 400
CAM_FPS = 30
SYNC_THRESHOLD_MS = 50


def offset_latlon(lat, lon, north_m, east_m):
    cos_lat = max(abs(math.cos(math.radians(lat))), 1e-6)
    return (
        lat + north_m / M_PER_DEG,
        lon + east_m / (M_PER_DEG * cos_lat),
    )


def dist_m(lat1, lon1, lat2, lon2):
    dn = (lat2 - lat1) * M_PER_DEG
    de = (
        (lon2 - lon1)
        * M_PER_DEG
        * math.cos(math.radians((lat1 + lat2) / 2.0))
    )
    return math.hypot(dn, de)


class GateNavigator(Node):
    def __init__(self):
        super().__init__("gate_navigator")

        self.declare_parameter("endpoint", "udpin:127.0.0.1:14551")
        endpoint = self.get_parameter("endpoint").value

        self.get_logger().info(f"Connecting MAVLink on {endpoint}")
        self.mav = mavutil.mavlink_connection(endpoint)
        self.mav.wait_heartbeat()
        self.get_logger().info("Heartbeat OK")

        self.waypoints = self._fetch_mission()
        if not self.waypoints:
            raise RuntimeError(
                "No NAV_WAYPOINT items found on vehicle. Upload a mission in QGC first."
            )

        self.get_logger().info(
            f"Fetched {len(self.waypoints)} navigation waypoints from vehicle"
        )
        for index, waypoint in enumerate(self.waypoints, start=1):
            self.get_logger().info(
                f"WP {index}: {waypoint[0]:.7f}, {waypoint[1]:.7f}"
            )

        if max(GATE_WP_INDICES) >= len(self.waypoints):
            raise RuntimeError(
                f"GATE_WP_INDICES={GATE_WP_INDICES} but only "
                f"{len(self.waypoints)} waypoint(s) were fetched"
            )

        self.wp_index = 0

        self.lat = None
        self.lon = None
        self.heading = None
        self.mode = ""
        self.armed = False

        # Gate state
        self.gate_samples = deque(maxlen=PAIR_CONFIRM_FRAMES)
        self.last_pair_sample_time = 0.0
        self.gate_midpoint = None
        self.gate_exit = None
        self.gate_locked = False
        self.gate_clearing = False

        # Camera/debug state
        self.frame = None
        self.frame_lock = threading.Lock()
        self.depth_img = None
        self.fx = None
        self.cx = None
        self.last_gate_debug = None

        self.get_logger().info("Loading TensorRT engine...")
        self.model = YOLO(ENGINE, task="detect")
        self._start_camera()
        self._start_stream()
        self.get_logger().info("Camera, synchronized depth, and GPU model ready")

        self.last_cmd = 0.0
        self.last_waiting_log = 0.0
        self.done = False
        self.fps = 0.0
        self._t_prev = time.time()

        self.timer = self.create_timer(0.05, self.tick)

    # ---------------- Mission fetch ----------------
    def _fetch_mission(self):
        self.mav.mav.mission_request_list_send(
            self.mav.target_system,
            self.mav.target_component,
        )

        count_msg = self.mav.recv_match(
            type="MISSION_COUNT",
            blocking=True,
            timeout=5,
        )
        if count_msg is None:
            return []

        waypoints = []

        for sequence in range(count_msg.count):
            self.mav.mav.mission_request_int_send(
                self.mav.target_system,
                self.mav.target_component,
                sequence,
            )

            item = self.mav.recv_match(
                type=["MISSION_ITEM_INT", "MISSION_ITEM"],
                blocking=True,
                timeout=5,
            )
            if item is None:
                self.get_logger().warning(
                    f"Timed out while requesting mission item {sequence}"
                )
                continue

            # ArduPilot mission downloads commonly expose home as sequence 0.
            # Keep the original behavior of ignoring that entry.
            if item.command != mavutil.mavlink.MAV_CMD_NAV_WAYPOINT or item.seq <= 0:
                continue

            if item.get_type() == "MISSION_ITEM_INT":
                lat = item.x / 1e7
                lon = item.y / 1e7
            else:
                lat = float(item.x)
                lon = float(item.y)

            waypoints.append((lat, lon))

        self.mav.mav.mission_ack_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_MISSION_ACCEPTED,
        )

        return waypoints

    # ---------------- OAK-D RGB + aligned synchronized depth ----------------
    def _start_camera(self):
        pipeline = dai.Pipeline()

        rgb = pipeline.create(dai.node.ColorCamera)
        rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1200_P
        )
        rgb.setIspScale(1, 3)  # 1920x1200 -> 640x400
        rgb.setInterleaved(False)
        rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        rgb.setFps(CAM_FPS)

        left = pipeline.create(dai.node.ColorCamera)
        left.setCamera("left")
        left.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1200_P
        )
        left.setIspScale(1, 3)
        left.setFps(CAM_FPS)

        right = pipeline.create(dai.node.ColorCamera)
        right.setCamera("right")
        right.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1200_P
        )
        right.setIspScale(1, 3)
        right.setFps(CAM_FPS)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(
            dai.node.StereoDepth.PresetMode.DEFAULT
        )
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(CAM_WIDTH, CAM_HEIGHT)

        left.isp.link(stereo.left)
        right.isp.link(stereo.right)

        sync = pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=SYNC_THRESHOLD_MS))

        rgb.isp.link(sync.inputs["rgb"])
        stereo.depth.link(sync.inputs["depth"])

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgbd")
        sync.out.link(xout.input)

        self.device = dai.Device(pipeline)
        self.q_rgbd = self.device.getOutputQueue(
            "rgbd",
            maxSize=4,
            blocking=False,
        )

    def _depth_for_box(self, box):
        """Return depth and lateral position for one RGB bounding box.

        The depth frame is aligned and synchronized to the 640x400 RGB frame,
        so RGB pixel coordinates can be used directly on the depth image.
        """
        if self.depth_img is None or self.fx is None or self.cx is None:
            return None, None, None, None

        x1, y1, x2, y2 = box
        height, width = self.depth_img.shape[:2]

        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width - 1, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            return None, None, None, None

        # Use a narrow central strip so the median is more likely to land on
        # the buoy rather than the water/background inside the full box.
        u = int((x1 + x2) / 2)
        v = int(y1 + 0.55 * (y2 - y1))

        half_w = max(4, min(12, (x2 - x1) // 8))
        half_h = max(5, min(20, (y2 - y1) // 6))

        u0 = max(0, u - half_w)
        u1 = min(width, u + half_w + 1)
        v0 = max(0, v - half_h)
        v1 = min(height, v + half_h + 1)

        roi = self.depth_img[v0:v1, u0:u1].astype(np.float32)
        valid = roi[
            (roi >= RANGE_MIN * 1000.0)
            & (roi <= RANGE_MAX * 1000.0)
        ]

        if valid.size < 5:
            return None, None, u, v

        forward_m = float(np.median(valid)) / 1000.0
        right_m = (u - self.cx) * forward_m / self.fx

        return forward_m, right_m, u, v

    # ---------------- Main control loop ----------------
    def tick(self):
        self._drain_mavlink()
        self._drain_camera()

        if self.done or self.lat is None or self.lon is None or self.heading is None:
            return

        if self.mode != "GUIDED":
            return

        if self.wp_index >= len(self.waypoints):
            self._finish_mission()
            return

        target = self._current_target()
        if target is None:
            return

        distance = dist_m(self.lat, self.lon, *target)

        if self.wp_index in GATE_WP_INDICES:
            if self.gate_locked:
                if not self.gate_clearing and distance < MIDPOINT_SWITCH_RADIUS:
                    self.gate_clearing = True
                    self.get_logger().info(
                        "Gate midpoint reached; commanding the gate-exit point"
                    )
                    target = self.gate_exit
                    distance = dist_m(self.lat, self.lon, *target)

                elif self.gate_clearing and distance < EXIT_ARRIVE_RADIUS:
                    self.get_logger().info(
                        f"Gate cleared at waypoint {self.wp_index + 1}"
                    )
                    self._advance_waypoint()
                    return
            else:
                planned = self.waypoints[self.wp_index]
                planned_distance = dist_m(self.lat, self.lon, *planned)

                # Do not skip the gate merely because the approximate QGC
                # waypoint was reached. Hold there until a stable gate is found.
                if planned_distance < NORMAL_WP_ARRIVE_RADIUS:
                    now = time.time()
                    if now - self.last_waiting_log > 2.0:
                        self.get_logger().info(
                            "At planned gate waypoint; waiting for a stable "
                            "same-frame red/green detection"
                        )
                        self.last_waiting_log = now
        else:
            if distance < NORMAL_WP_ARRIVE_RADIUS:
                self.get_logger().info(
                    f"Waypoint {self.wp_index + 1} reached"
                )
                self._advance_waypoint()
                return

        now = time.time()
        if now - self.last_cmd >= CMD_PERIOD:
            self._send_target(*target)
            self.last_cmd = now

    def _current_target(self):
        if self.wp_index in GATE_WP_INDICES and self.gate_locked:
            if self.gate_clearing:
                return self.gate_exit
            return self.gate_midpoint

        return self.waypoints[self.wp_index]

    def _advance_waypoint(self):
        self.wp_index += 1
        self._reset_gate_state()

        if self.wp_index >= len(self.waypoints):
            self._finish_mission()

    def _finish_mission(self):
        if self.done:
            return

        self.get_logger().info("Mission complete - switching to LOITER")
        self.mav.set_mode("LOITER")
        self.done = True

    def _reset_gate_state(self):
        self.gate_samples.clear()
        self.last_pair_sample_time = 0.0
        self.gate_midpoint = None
        self.gate_exit = None
        self.gate_locked = False
        self.gate_clearing = False
        self.last_gate_debug = None

    # ---------------- MAVLink input ----------------
    def _drain_mavlink(self):
        while True:
            message = self.mav.recv_match(blocking=False)
            if message is None:
                break

            message_type = message.get_type()

            if message_type == "GLOBAL_POSITION_INT":
                self.lat = message.lat / 1e7
                self.lon = message.lon / 1e7

                # 65535 means heading is unknown.
                if message.hdg != 65535:
                    self.heading = message.hdg / 100.0

            elif message_type == "VFR_HUD" and self.heading is None:
                self.heading = float(message.heading)

            elif (
                message_type == "HEARTBEAT"
                and message.get_srcSystem() == self.mav.target_system
            ):
                self.armed = bool(
                    message.base_mode
                    & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.mode = mavutil.mode_string_v10(message)

    # ---------------- Camera inference and gate pairing ----------------
    def _drain_camera(self):
        group = None

        # Keep the newest complete synchronized RGB/depth group.
        while True:
            candidate_group = self.q_rgbd.tryGet()
            if candidate_group is None:
                break
            group = candidate_group

        if group is None:
            return

        messages = {name: message for name, message in group}
        rgb_message = messages.get("rgb")
        depth_message = messages.get("depth")

        if rgb_message is None or depth_message is None:
            return

        image = rgb_message.getCvFrame()
        self.depth_img = depth_message.getFrame()

        if image.shape[:2] != self.depth_img.shape[:2]:
            self.get_logger().error(
                "RGB/depth dimensions do not match: "
                f"RGB={image.shape[:2]}, depth={self.depth_img.shape[:2]}"
            )
            return

        if self.fx is None:
            height, width = self.depth_img.shape[:2]
            calibration = self.device.readCalibration()
            intrinsics = calibration.getCameraIntrinsics(
                dai.CameraBoardSocket.CAM_A,
                width,
                height,
            )
            self.fx = float(intrinsics[0][0])
            self.cx = float(intrinsics[0][2])
            self.get_logger().info(
                f"RGB intrinsics loaded: fx={self.fx:.2f}, cx={self.cx:.2f}"
            )

        detections = []
        result = self.model.predict(
            image,
            verbose=False,
            imgsz=(352, 640),
            conf=CONF_SHOW,
        )[0]

        for box in result.boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0])
            class_index = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_index < 0 or class_index >= len(LABELS):
                continue

            name = LABELS[class_index]
            forward_m, right_m, u, v = self._depth_for_box(
                (x1, y1, x2, y2)
            )

            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "name": name,
                    "conf": confidence,
                    "z": forward_m,
                    "x": right_m,
                    "u": u,
                    "v": v,
                }
            )

        gate_debug = None

        if (
            self.wp_index in GATE_WP_INDICES
            and not self.gate_locked
            and self.lat is not None
            and self.lon is not None
            and self.heading is not None
        ):
            gate_candidate = self._select_gate_candidate(detections)

            if gate_candidate is not None:
                gate_debug = gate_candidate
                self._record_gate_candidate(gate_candidate)
            elif (
                self.gate_samples
                and time.time() - self.last_pair_sample_time
                > PAIR_SAMPLE_TIMEOUT
            ):
                self.gate_samples.clear()

        self.last_gate_debug = gate_debug
        self._annotate(image, detections, gate_debug)

        with self.frame_lock:
            self.frame = image

    def _select_gate_candidate(self, detections):
        red_detections = [
            detection
            for detection in detections
            if detection["name"] in RED_CLASSES
            and detection["conf"] >= CONF_MIN
            and detection["z"] is not None
            and detection["x"] is not None
        ]

        green_detections = [
            detection
            for detection in detections
            if detection["name"] in GREEN_CLASSES
            and detection["conf"] >= CONF_MIN
            and detection["z"] is not None
            and detection["x"] is not None
        ]

        if not red_detections or not green_detections:
            return None

        planned = self.waypoints[self.wp_index]
        best_candidate = None
        best_score = float("inf")

        # Test every red/green combination from this one synchronized frame.
        for red in red_detections:
            for green in green_detections:
                if red["u"] is None or green["u"] is None:
                    continue

                pixel_separation = abs(red["u"] - green["u"])
                if pixel_separation < PAIR_MIN_PIXEL_SEP:
                    continue

                red_forward = red["z"]
                red_right = red["x"]
                green_forward = green["z"]
                green_right = green["x"]

                depth_difference = abs(red_forward - green_forward)
                if depth_difference > PAIR_MAX_DEPTH_DIFF:
                    continue

                gate_forward = green_forward - red_forward
                gate_right = green_right - red_right
                gate_width = math.hypot(gate_forward, gate_right)

                if not GATE_MIN_WIDTH <= gate_width <= GATE_MAX_WIDTH:
                    continue

                midpoint_forward = (red_forward + green_forward) / 2.0
                midpoint_right = (red_right + green_right) / 2.0
                # Empirical correction: computed target consistently lands on
                # the left buoy, so shift toward whichever buoy is on the right.
                if red_right >= green_right:
                    toward_right_f = red_forward - green_forward
                    toward_right_r = red_right - green_right
                else:
                    toward_right_f = green_forward - red_forward
                    toward_right_r = green_right - red_right
                midpoint_forward += MIDPOINT_SHIFT_RATIO * toward_right_f
                midpoint_right += MIDPOINT_SHIFT_RATIO * toward_right_r

                if midpoint_forward <= RANGE_MIN:
                    continue

                midpoint_world = self._body_to_world(
                    midpoint_forward,
                    midpoint_right,
                )

                correction_distance = dist_m(
                    *midpoint_world,
                    *planned,
                )
                if correction_distance > MAX_CORRECTION:
                    continue

                # The line between the buoys defines the gate. Use the
                # perpendicular that points away from the current boat position
                # to create a safe exit target beyond the gate.
                normal_forward = -gate_right / gate_width
                normal_right = gate_forward / gate_width

                if (
                    normal_forward * midpoint_forward
                    + normal_right * midpoint_right
                    < 0.0
                ):
                    normal_forward *= -1.0
                    normal_right *= -1.0

                exit_forward = (
                    midpoint_forward
                    + normal_forward * GATE_EXIT_DISTANCE
                )
                exit_right = (
                    midpoint_right
                    + normal_right * GATE_EXIT_DISTANCE
                )

                exit_world = self._body_to_world(
                    exit_forward,
                    exit_right,
                )

                # Prefer the pair whose midpoint is closest to the approximate
                # QGC waypoint, then prefer buoys at similar depth.
                score = correction_distance + 0.20 * depth_difference

                if score < best_score:
                    best_score = score
                    best_candidate = {
                        "red": red,
                        "green": green,
                        "width": gate_width,
                        "mid_body": (midpoint_forward, midpoint_right),
                        "mid_world": midpoint_world,
                        "exit_body": (exit_forward, exit_right),
                        "exit_world": exit_world,
                        "planned_error": correction_distance,
                    }

        return best_candidate

    def _record_gate_candidate(self, candidate):
        now = time.time()

        if (
            self.gate_samples
            and now - self.last_pair_sample_time > PAIR_SAMPLE_TIMEOUT
        ):
            self.gate_samples.clear()

        self.last_pair_sample_time = now

        midpoint = candidate["mid_world"]
        exit_point = candidate["exit_world"]

        self.gate_samples.append(
            (
                midpoint[0],
                midpoint[1],
                exit_point[0],
                exit_point[1],
            )
        )

        midpoint_forward, midpoint_right = candidate["mid_body"]
        self.get_logger().info(
            "Gate pair: "
            f"width={candidate['width']:.2f}m, "
            f"mid=({midpoint_forward:.2f}m forward, "
            f"{midpoint_right:+.2f}m right), "
            f"samples={len(self.gate_samples)}/{PAIR_CONFIRM_FRAMES}",
            throttle_duration_sec=0.5,
        )

        if len(self.gate_samples) < PAIR_CONFIRM_FRAMES:
            return

        midpoint_lat = median(sample[0] for sample in self.gate_samples)
        midpoint_lon = median(sample[1] for sample in self.gate_samples)
        exit_lat = median(sample[2] for sample in self.gate_samples)
        exit_lon = median(sample[3] for sample in self.gate_samples)

        midpoint_spread = max(
            dist_m(midpoint_lat, midpoint_lon, sample[0], sample[1])
            for sample in self.gate_samples
        )
        exit_spread = max(
            dist_m(exit_lat, exit_lon, sample[2], sample[3])
            for sample in self.gate_samples
        )

        if max(midpoint_spread, exit_spread) > PAIR_MAX_SPREAD:
            self.get_logger().warning(
                "Gate estimate is not stable yet: "
                f"midpoint spread={midpoint_spread:.2f}m, "
                f"exit spread={exit_spread:.2f}m"
            )
            return

        self.gate_midpoint = (midpoint_lat, midpoint_lon)
        self.gate_exit = (exit_lat, exit_lon)
        self.gate_locked = True
        self.gate_clearing = False

        self.get_logger().info(
            "Gate correction locked: "
            f"midpoint=({midpoint_lat:.7f}, {midpoint_lon:.7f}), "
            f"exit=({exit_lat:.7f}, {exit_lon:.7f})"
        )

    def _body_to_world(self, forward_m, right_m):
        # Translate the camera measurement to the boat/GPS reference point.
        forward_m += CAMERA_FORWARD_OFFSET_M
        right_m += CAMERA_RIGHT_OFFSET_M

        camera_heading = (
            self.heading + CAMERA_YAW_OFFSET_DEG
        ) % 360.0
        heading_rad = math.radians(camera_heading)

        north_m = (
            forward_m * math.cos(heading_rad)
            - right_m * math.sin(heading_rad)
        )
        east_m = (
            forward_m * math.sin(heading_rad)
            + right_m * math.cos(heading_rad)
        )

        return offset_latlon(
            self.lat,
            self.lon,
            north_m,
            east_m,
        )

    # ---------------- MAVLink output ----------------
    def _send_target(self, lat, lon):
        time_boot_ms = int(time.monotonic() * 1000.0) & 0xFFFFFFFF

        self.mav.mav.set_position_target_global_int_send(
            time_boot_ms,
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            POSITION_ONLY_TYPE_MASK,
            int(lat * 1e7),
            int(lon * 1e7),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )

        if self.wp_index in GATE_WP_INDICES and self.gate_locked:
            tag = "gate-exit" if self.gate_clearing else "gate-midpoint"
        else:
            tag = "planned"

        self.get_logger().info(
            f"WP {self.wp_index + 1}/{len(self.waypoints)} "
            f"({tag}) dist={dist_m(self.lat, self.lon, lat, lon):.1f}m"
        )

    # ---------------- Debug video stream ----------------
    def _annotate(self, image, detections, gate_candidate):
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            name = detection["name"]

            if name in GREEN_CLASSES:
                color = (0, 255, 0)
            elif name in RED_CLASSES:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {int(detection['conf'] * 100)}%"
            if detection["z"] is not None and detection["x"] is not None:
                label += (
                    f" | {detection['z']:.1f}m "
                    f"{detection['x']:+.1f}m"
                )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        if gate_candidate is not None:
            red = gate_candidate["red"]
            green = gate_candidate["green"]

            red_point = (red["u"], red["v"])
            green_point = (green["u"], green["v"])
            midpoint_point = (
                int((red["u"] + green["u"]) / 2),
                int((red["v"] + green["v"]) / 2),
            )

            cv2.line(image, red_point, green_point, (255, 255, 255), 2)
            cv2.circle(image, midpoint_point, 6, (255, 255, 255), -1)
            cv2.putText(
                image,
                f"gate {gate_candidate['width']:.2f}m",
                (midpoint_point[0] + 8, midpoint_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

        now = time.time()
        instantaneous_fps = 1.0 / max(now - self._t_prev, 1e-3)
        self.fps = 0.9 * self.fps + 0.1 * instantaneous_fps
        self._t_prev = now

        if self.gate_clearing:
            phase = "GATE EXIT"
        elif self.gate_locked:
            phase = "GATE MIDPOINT"
        elif self.wp_index in GATE_WP_INDICES:
            phase = f"GATE SEARCH {len(self.gate_samples)}/{PAIR_CONFIRM_FRAMES}"
        else:
            phase = "PLANNED"

        waypoint_number = min(self.wp_index + 1, len(self.waypoints))
        status = (
            f"{self.fps:.0f}fps  "
            f"WP {waypoint_number}/{len(self.waypoints)}  "
            f"{phase}  mode {self.mode}"
        )

        cv2.putText(
            image,
            status,
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    def _start_stream(self):
        node = self

        class StreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()

                try:
                    while True:
                        with node.frame_lock:
                            frame = (
                                None
                                if node.frame is None
                                else node.frame.copy()
                            )

                        if frame is None:
                            time.sleep(0.1)
                            continue

                        ok, jpeg = cv2.imencode(
                            ".jpg",
                            frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 60],
                        )

                        if ok:
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                            )
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b"\r\n")

                        time.sleep(0.05)

                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        server = HTTPServer(("0.0.0.0", 8080), StreamHandler)
        threading.Thread(
            target=server.serve_forever,
            daemon=True,
        ).start()


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
