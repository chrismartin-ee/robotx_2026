import sys
import time
from pymavlink import mavutil

vx, vy = float(sys.argv[1]), float(sys.argv[2])   # body frame m/s: vx fwd, vy right
m = mavutil.mavlink_connection('udpin:127.0.0.1:14551')
m.wait_heartbeat()
print(f"connected - sending vx={vx} vy={vy} for 5s")
t0 = time.time()
while time.time() - t0 < 5:
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,   # use velocity only
        0, 0, 0, vx, vy, 0, 0, 0, 0, 0, 0)
    time.sleep(0.2)
print("done")
