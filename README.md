# RobotX 2026 USV (Crusader)

Jetson Orin Nano + Pixhawk (ArduRover). ROS 2 Humble in Docker container `crusader`.
Code lives on the HOST at ~/robotx_ws, bind-mounted into the container at /root/robotx_ws.

## Hardware map
- Pixhawk: /dev/serial/by-id/usb-ArduPilot_Pixhawk1_3A001F...-if00 (owned by MAVProxy ONLY)
- LED Arduino (CH340, no serial number — do not add a second CH340): usb-1a86_USB_Serial-if00-port0
- OAK-D LR camera: MX ID 194430101110C82F00 (idle state shows as "Luxonis Bootloader" in lsusb — normal)
- RC: RC7 = arm/e-stop (RCx_OPTION=165; 994 estop / 1498 mid / 1995 arm). Channel 8 = mode (SC switch).

## Start order (all inside container: docker exec -it crusader bash)
1. ./scripts/start_mavproxy.sh <laptop-ip>
2. ros2 launch robotx_2026 core.launch.py
LED: red = disarmed/e-stop, yellow = armed manual, green = auto.

```
robotx_2026/
├── package.xml                <- ROS 2 package manifest
├── setup.py                   <- defines the nodes: led_node, pixhawk_led_node,
│                                 gate_navigator
├── setup.cfg
├── .gitignore
├── README.md
├── resource/
│   └── robotx_2026            <- empty marker file ROS 2 requires
├── launch/
│   └── core.launch.py         <- starts led_node + pixhawk_led_node together
├── firmware/
│   └── LED.ino                <- Arduino sketch on the LED controller
├── models/
│   ├── buoy_v16.blob          <- buoy detection model (MHSeals V16, YOLOv11,
│   │                             13 classes) compiled for the OAK-D camera
│   └── buoy_v16.json          <- conversion settings for the blob
├── scripts/
│   └── start_mavproxy.sh      <- MAVProxy = ONLY owner of the Pixhawk USB port
│                                 usage: ./start_mavproxy.sh <LAPTOP_IP>
├── tools/
│   ├── oak_view.py            <- plain camera check -> http://<jetson-ip>:8080
│   └── oak_buoy_view.py       <- camera + buoy detection + distances -> :8080
└── robotx_2026/               <- the Python package (all ROS nodes)
    └── api/
        ├── led/
        │   └── led_node.py                <- /led_state -> Arduino serial,
        │                                     auto-reconnects
        ├── pixhawk/
        │   └── pixhawk_led_status_node.py <- Pixhawk state -> /led_state
        │                                     (1=red 2=yellow 3=green)
        └── navigation/
            └── dp_hold.py		<- holds position in relation to target (buoy for now, hard 
                                           coded yaw and distance from target, keeping target centered 
                                           with lateral movement)
            └── gate_navigator.py          <- GUIDED-mode gate navigation:
                                              drives QGC waypoints, corrects
                                              to buoy-gate midpoints (UNTESTED)

```
