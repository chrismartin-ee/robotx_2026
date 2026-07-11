import depthai as dai
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

BLOB = str((Path(__file__).parent / "../models/buoy_v16.blob").resolve())
LABELS = ["black_buoy", "black_cross", "black_target_boat", "black_triangle",
          "blue_buoy", "green_buoy", "green_light_buoy", "green_pole_buoy",
          "red_buoy", "red_light_buoy", "red_pole_buoy", "yellow_buoy",
          "yellow_target_boat"]

pipeline = dai.Pipeline()

# center camera feeds the model
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
cam.setIspScale(2, 3)
cam.setPreviewSize(640, 352)
cam.setInterleaved(False)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setFps(30)

# left/right cameras feed stereo depth
left = pipeline.create(dai.node.ColorCamera)
left.setCamera("left")
left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
left.setIspScale(1, 3)
left.setFps(10)

right = pipeline.create(dai.node.ColorCamera)
right.setCamera("right")
right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
right.setIspScale(1, 3)
right.setFps(10)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(False)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
left.isp.link(stereo.left)
right.isp.link(stereo.right)

# spatial detection: model + depth fused on-camera
nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
nn.setBlobPath(BLOB)
nn.setNumClasses(len(LABELS))
nn.setCoordinateSize(4)
nn.setIouThreshold(0.5)
nn.setConfidenceThreshold(0.5)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
nn.setBoundingBoxScaleFactor(0.5)   # sample depth from the middle of the box
nn.setDepthLowerThreshold(300)      # mm
nn.setDepthUpperThreshold(35000)    # mm
cam.preview.link(nn.input)
stereo.depth.link(nn.inputDepth)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam.preview.link(xout_rgb.input)
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

device = dai.Device(pipeline)
q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)
last_dets = []
print("Camera + buoy model + depth up. Open http://<JETSON_IP>:8080")

def get_annotated_frame():
    global last_dets
    frame = q_rgb.get().getCvFrame()
    dets = q_nn.tryGet()
    if dets is not None:
        last_dets = dets.detections
    h, w = frame.shape[:2]
    for d in last_dets:
        x1, y1 = int(d.xmin * w), int(d.ymin * h)
        x2, y2 = int(d.xmax * w), int(d.ymax * h)
        name = LABELS[d.label]
        if "green" in name:
            color = (0, 255, 0)
        elif "red" in name:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)
        z_m = d.spatialCoordinates.z / 1000.0
        x_m = d.spatialCoordinates.x / 1000.0
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {int(d.confidence*100)}%",
                    (x1, max(y1 - 20, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"{z_m:.1f}m ahead, {x_m:+.1f}m side",
                    (x1, max(y1 - 6, 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                ok, jpg = cv2.imencode(".jpg", get_annotated_frame(),
                                       [cv2.IMWRITE_JPEG_QUALITY, 60])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
