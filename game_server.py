"""
Assetto Corsa UDP Remote Telemetry Client module.
Your app connects TO AC on port 9996, receives telemetry on a random local port.
"""

import socket
import struct
import threading
import os
from dotenv import load_dotenv
from ac_shared_memory import start_shared_memory, stop_shared_memory, get_shared_memory_data

load_dotenv()

game_data = {}
handshake_data = {}
lap_data = {}

_lock = threading.Lock()
_running = False

# AC Protocol Constants
HANDSHAKE = 0

DISMISS = 3


def parse_handshaker_response(data: bytes) -> dict:
    # JS parser expects exactly 408 bytes
    if len(data) < 408:
        return {}

    car_name = data[0:100].decode(
        'utf-16le',
        errors='ignore'
    ).rstrip('\x00')

    driver_name = data[100:200].decode(
        'utf-16le',
        errors='ignore'
    ).rstrip('\x00')

    identifier = struct.unpack_from('<i', data, 200)[0]
    version = struct.unpack_from('<i', data, 204)[0]

    track_name = data[208:308].decode(
        'utf-16le',
        errors='ignore'
    ).rstrip('\x00')

    track_config = data[308:408].decode(
        'utf-16le',
        errors='ignore'
    ).rstrip('\x00')

    return {
        'car_name': car_name,
        'driver_name': driver_name,
        'identifier': identifier,
        'version': version,
        'track_name': track_name,
        'track_config': track_config,
    }

def parse_rtlap(data: bytes) -> dict:
    # Matches the JavaScript RTLapParser exactly
    if len(data) < 212:
        return {}

    car_identifier_number = struct.unpack_from('<i', data, 0)[0]
    lap = struct.unpack_from('<i', data, 4)[0]

    driver_name = data[8:108].decode(
        'utf-16le',
        errors='ignore'
    ).rstrip('\x00')

    car_name = data[108:208].decode(
        'utf-16le',
        errors='ignore'
    ).rstrip('\x00')

    time = struct.unpack_from('<i', data, 208)[0]

    return {
        'car_identifier_number': car_identifier_number,
        'lap': lap,
        'driver_name': driver_name,
        'car_name': car_name,
        'time': time,
    }

def parse_rtcarinfo(data: bytes) -> dict:
    # AC RTCarInfo packets use 8-byte header (identifier + size) followed by payload fields.
    if len(data) < 328:
        return {}

    offset = 8

    speed_kmh, speed_mph, speed_ms = struct.unpack_from('<fff', data, offset)
    offset += 12

    abs_enabled, abs_in_action, tc_in_action, tc_enabled, in_pit, engine_limiter = struct.unpack_from('<6b', data, offset)
    offset += 6

    # Skip 2 unknown bytes after the boolean flags
    offset += 2

    acc_vertical, acc_horizontal, acc_frontal = struct.unpack_from('<fff', data, offset)
    offset += 12

    lap_time, last_lap, best_lap, lap_count = struct.unpack_from('<4i', data, offset)
    offset += 16

    gas, brake, clutch, engine_rpm, steer = struct.unpack_from('<5f', data, offset)
    offset += 20

    gear = struct.unpack_from('<i', data, offset)[0]
    offset += 4

    cg_height = struct.unpack_from('<f', data, offset)[0]
    offset += 4

    wheel_angular_speed = struct.unpack_from('<4f', data, offset)
    offset += 16

    slip_angle = struct.unpack_from('<4f', data, offset)
    offset += 16

    slip_angle_cp = struct.unpack_from('<4f', data, offset)
    offset += 16

    slip_ratio = struct.unpack_from('<4f', data, offset)
    offset += 16

    tyre_slip = struct.unpack_from('<4f', data, offset)
    offset += 16

    nd_slip = struct.unpack_from('<4f', data, offset)
    offset += 16

    load = struct.unpack_from('<4f', data, offset)
    offset += 16

    dy = struct.unpack_from('<4f', data, offset)
    offset += 16

    mz = struct.unpack_from('<4f', data, offset)
    offset += 16

    tyre_dirty = struct.unpack_from('<4f', data, offset)
    offset += 16

    camber = struct.unpack_from('<4f', data, offset)
    offset += 16

    tyre_radius = struct.unpack_from('<4f', data, offset)
    offset += 16

    tyre_loaded_radius = struct.unpack_from('<4f', data, offset)
    offset += 16

    suspension_height = struct.unpack_from('<4f', data, offset)
    offset += 16

    car_pos_normalized, car_slope = struct.unpack_from('<2f', data, offset)
    offset += 8

    car_coords = None
    if len(data) >= offset + 12:
        car_coords = struct.unpack_from('<3f', data, offset)
        offset += 12
    # return {
    #     'speed_kmh': speed_kmh,
    #     'speed_mph': speed_mph,
    #     'speed_ms': speed_ms,
    #     'is_abs_enabled': bool(abs_enabled),
    #     'is_abs_in_action': bool(abs_in_action),
    #     'is_tc_in_action': bool(tc_in_action),
    #     'is_tc_enabled': bool(tc_enabled),
    #     'is_in_pit': bool(in_pit),
    #     'is_engine_limiter_on': bool(engine_limiter),
    #     'acc_vert': acc_vertical,
    #     'acc_horz': acc_horizontal,
    #     'acc_front': acc_frontal,
    #     'lap_time': lap_time,
    #     'last_lap': last_lap,
    #     'best_lap': best_lap,
    #     'lap_count': lap_count,
    #     'gas': gas,
    #     'brake': brake,
    #     'clutch': clutch,
    #     'engine_rpm': engine_rpm,
    #     'steer': steer,
    #     'gear': gear,
    #     'cg_height': cg_height,
    #     'wheel_angular_speed': wheel_angular_speed,
    #     'slip_angle': slip_angle,
    #     'slip_angle_cp': slip_angle_cp,
    #     'slip_ratio': slip_ratio,
    #     'tyre_slip': tyre_slip,
    #     'nd_slip': nd_slip,
    #     'load': load,
    #     'dy': dy,
    #     'mz': mz,
    #     'tyre_dirty': tyre_dirty,
    #     'camber': camber,
    #     'tyre_radius': tyre_radius,
    #     'tyre_loaded_radius': tyre_loaded_radius,
    #     'suspension_height': suspension_height,
    #     'car_pos_normalized': car_pos_normalized,
    #     'car_slope': car_slope,
    #     'car_coords': car_coords,
    # }
    return {
        'speed_kmh': speed_kmh,
        'engine_rpm': engine_rpm,
        'gear': gear,
        'gas': gas,
        'brake': brake,
        'lap_time': lap_time,
        'last_lap': last_lap,
        'best_lap': best_lap,
        'lap_count': lap_count,
        'is_abs_in_action': bool(abs_in_action),
        'is_tc_in_action': bool(tc_in_action),
        'is_engine_limiter_on': bool(engine_limiter),
    }

def create_handshaker(operation_id: int) -> bytes:
    return struct.pack('<iii', 1, 1, operation_id)


def get_game_data() -> dict:
    with _lock:
        return game_data.copy() if game_data else {}

def get_handshake_data() -> dict:
    with _lock:
        return handshake_data.copy() if handshake_data else {}

def get_lap_data() -> dict:
    with _lock:
        return lap_data.copy() if lap_data else {}


def start_game_server():
    global _running
    _running = True

    # Start shared memory reader here, before the UDP thread launches.
    # It connects on its own background thread so this returns immediately —
    # use get_shared_memory_data() later and always guard with .get().
    start_shared_memory()

    ac_host = os.getenv("AC_HOST", "127.0.0.1")
    ac_port = int(os.getenv("AC_PORT", 9996))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 0))
    sock.settimeout(5.0)
    local_port = sock.getsockname()[1]
    print(f"[AC Client] Listening on local port {local_port}, connecting to AC at {ac_host}:{ac_port}")

    def udp_client():
        global game_data
        global handshake_data
        global lap_data

        # Step 1: Send HANDSHAKE
        sock.sendto(create_handshaker(HANDSHAKE), (ac_host, ac_port))
        print("[AC Client] Sent HANDSHAKE")

        # Step 2: Wait for handshake response
        try:
            data, _ = sock.recvfrom(4096)
            print(len(data))
            if len(data) >= 408:
                handshake_data = parse_handshaker_response(data)
                print(f"[AC Client] Connected — Car: {handshake_data['car_name']} | Driver: {handshake_data['driver_name']} | Track: {handshake_data['track_name']}")
            else:
                print(f"[AC Client] Got short handshake response ({len(data)} bytes), continuing anyway")
        except socket.timeout:
            print("[AC Client] Timeout waiting for handshake response — is AC running and in-session?")
            sock.close()
            return

        # Step 4: Subscribe to lap/spot packets
        sock.sendto(create_handshaker(int(os.getenv("UPDATE_TYPE", 1))), (ac_host, ac_port))
        print("[AC Client] Subscribed to car/lap")

        sock.settimeout(2.0)

        while _running:
            try:
                data, _ = sock.recvfrom(4096)
                packet_size = len(data)

                # HandShake data
                if packet_size == 408:
                    handshake = parse_handshaker_response(data)
                    with _lock:
                        handshake_data = handshake
                    print(f"[AC Client] Handshake packet: {handshake}")

                # Car data
                elif packet_size == 328:
                    telemetry = parse_rtcarinfo(data)
                    with _lock:
                        game_data = telemetry
                        # print(f"[AC Client] Car packet: {game_data}")


                # Track data
                elif packet_size == 212:
                    lap_info = parse_rtlap(data)
                    with _lock:
                        lap_data = lap_info
                    print(f"[AC Client] Lap packet: {lap_info}")

                else:
                    print(f"[AC Client] Unknown packet size: {packet_size}")

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[AC Client] Error: {e}")

        # Clean disconnect
        sock.sendto(create_handshaker(DISMISS), (ac_host, ac_port))
        sock.close()
        print("[AC Client] Disconnected from AC")

    thread = threading.Thread(target=udp_client, daemon=True)
    thread.start()
    return thread


def stop_game_server():
    global _running
    _running = False
    stop_shared_memory()