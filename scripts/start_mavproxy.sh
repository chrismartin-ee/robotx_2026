#!/bin/bash
# MAVProxy owns the Pixhawk USB port (only ONE process may open it).
# Rebroadcasts MAVLink on UDP: localhost for our nodes, laptop for GCS.
# Usage: ./start_mavproxy.sh [LAPTOP_IP]
LAPTOP_IP="${1:-192.168.8.137}"   # <-- put your usual laptop IP here
mavproxy.py \
  --master=/dev/serial/by-id/usb-ArduPilot_Pixhawk1_3A001F001051333531353431-if00 \
  --out=udp:127.0.0.1:14550 \
  --out=udp:${LAPTOP_IP}:14550
  --out=udp:127.0.0.1:14551 \
