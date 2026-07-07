import depthai as dai
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer

pipeline = dai.Pipeline()
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
cam.setIspScale(1, 2)
cam.setFps(15)
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.isp.link(xout.input)

device = dai.Device(pipeline)
q = device.getOutputQueue("video", maxSize=4, blocking=False)
print("Camera up. Open http://<JETSON_IP>:8080 in a browser.")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                frame = q.get().getCvFrame()
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
