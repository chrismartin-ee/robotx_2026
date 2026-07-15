import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import cv2
import depthai as dai
from ultralytics import YOLO

ENGINE = str((Path(__file__).parent / "../models/buoy_v16.engine").resolve())
LABELS = ["black_buoy", "black_cross", "black_target_boat", "black_triangle",
          "blue_buoy", "green_buoy", "green_light_buoy", "green_pole_buoy",
          "red_buoy", "red_light_buoy", "red_pole_buoy", "yellow_buoy",
          "yellow_target_boat"]
CONF_MIN = 0.5

# ---- camera pipeline: RGB + depth only, no NN on-camera ----
pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
cam.setIspScale(2, 3)
cam.setPreviewSize(640, 352)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setFps(30)

left = pipeline.create(dai.node.ColorCamera)
left.setCamera("left")
left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
left.setIspScale(1, 3)
left.setFps(30)
right = pipeline.create(dai.node.ColorCamera)
right.setCamera("right")
right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
right.setIspScale(1, 3)
right.setFps(30)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
left.isp.link(stereo.left)
right.isp.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

device = dai.Device(pipeline)
q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

model = YOLO(ENGINE, task="detect")
print("Camera + GPU model up. Open http://<JETSON_IP>:8080")

latest = {"frame": None}
lock = threading.Lock()


def depth_at(depth_img, u, v, half=8):
    """Median depth (m) in a small ROI around preview pixel (u, v)."""
    dh, dw = depth_img.shape[:2]
    s = dw / 640.0
    crop_off = (dh / s - 352.0) / 2.0     # preview is a vertical center-crop
    du, dv = int(u * s), int((v + crop_off) * s)
    u0, u1 = max(0, du - half), min(dw, du + half + 1)
    v0, v1 = max(0, dv - half), min(dh, dv + half + 1)
    roi = depth_img[v0:v1, u0:u1].astype(np.float32)
    valid = roi[roi > 0]
    if valid.size == 0:
        return None, du, dv
    return float(np.median(valid)) / 1000.0, du, dv


# intrinsics of CAM_A at the depth frame's geometry (queried on first frame)
fx = cx = None


def main_loop():
    global fx, cx
    depth_img = None
    t_prev, fps = time.time(), 0.0
    while True:
        d = q_depth.tryGet()
        if d is not None:
            depth_img = d.getFrame()
            if fx is None:
                calib = device.readCalibration()
                M = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A,
                                              depth_img.shape[1],
                                              depth_img.shape[0])
                fx, cx = M[0][0], M[0][2]
        f = q_rgb.get()
        frame = f.getCvFrame()

        res = model.predict(frame, verbose=False, imgsz=(352, 640),
                            conf=CONF_MIN)[0]
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
            name = LABELS[int(b.cls[0])]
            conf = float(b.conf[0])
            color = ((0, 255, 0) if "green" in name else
                     (0, 0, 255) if "red" in name else (0, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {int(conf*100)}%"
            if depth_img is not None and fx is not None:
                z, du, dv = depth_at(depth_img, (x1 + x2) // 2, (y1 + y2) // 2)
                if z is not None and z > 0.3:
                    x_m = (du - cx) * z / fx
                    label += f" | {z:.1f}m, {x_m:+.1f}m side"
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
        t_prev = now
        cv2.putText(frame, f"{fps:.0f} fps", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        with lock:
            latest["frame"] = frame


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with lock:
                    fr = None if latest["frame"] is None else latest["frame"].copy()
                if fr is None:
                    time.sleep(0.05)
                    continue
                ok, jpg = cv2.imencode(".jpg", fr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])
                if ok:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *a):
        pass


threading.Thread(target=main_loop, daemon=True).start()
HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
