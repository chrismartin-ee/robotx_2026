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
