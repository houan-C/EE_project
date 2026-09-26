#!/usr/bin/env python3
"""
Receiver RSSI Rotation & Distance Test Script

Supports iPhone default GPS coordinates DMS format (e.g. 25°0'44" 121°32'27")
as well as standard decimal coordinates (e.g. 25.0122, 121.5408 or X, Y).

Workflow:
1. Choose whether to CREATE a new CSV dataset or EDIT/APPEND to an existing CSV file.
2. Enter Receiver Coordinate (GPS DMS like 25°0'44" 121°32'27" or X, Y) and initial compass facing bearing (0-360°).
3. Enter Transmitter Coordinate (GPS DMS like 25°0'45" 121°32'28" or X, Y).
4. Script automatically calculates distance between RX and TX.
5. Record RSSI for 5 seconds to compute average RSSI and throughput.
6. Prompt user to turn receiver LEFT by 30° (using compass for alignment), then enter 'ready'.
7. Repeat for a full 360° rotation (12 steps).
8. Once 360° sweep is done, prompt for a NEW Transmitter Coordinate.
"""

import argparse
import csv
import math
import os
import re
import sys
import time
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


DEFAULT_BAUD = 921600
DEFAULT_CSV_PATH = "rssi_rotation_data.csv"
DEFAULT_DURATION = 5.0   # 5 seconds average
STEP_DEGREES = 30        # Turn 30 degrees left per step
TOTAL_STEPS = 12         # 360 degrees / 30 degrees = 12 steps


def dms_to_dd(deg, min_val, sec_val, direction=None):
    """Convert Degrees, Minutes, Seconds to Decimal Degrees."""
    dd = float(deg) + float(min_val) / 60.0 + float(sec_val) / 3600.0
    if direction and direction.upper() in ['S', 'W']:
        dd = -dd
    return dd


def parse_gps_coord(coord_str):
    """
    Parses iPhone DMS string e.g. '25°0'44" 121°32'27"' or '25°0'44"N 121°32'27"E'
    or decimal degrees '25.0122, 121.5408' or simple (x, y) coordinates.
    Returns: (lat_or_x, lon_or_y) as floats.
    """
    coord_str = coord_str.strip()
    
    # Check DMS pattern matching degrees (°), minutes ('), seconds (")
    dms_pattern = r"(\d+)\s*°\s*(\d+)\s*['’]\s*([\d.]+)\s*[\"”]?\s*([NSEWnsew])?"
    dms_matches = re.findall(dms_pattern, coord_str)
    
    if len(dms_matches) >= 2:
        lat_d, lat_m, lat_s, lat_dir = dms_matches[0]
        lon_d, lon_m, lon_s, lon_dir = dms_matches[1]
        lat = dms_to_dd(lat_d, lat_m, lat_s, lat_dir)
        lon = dms_to_dd(lon_d, lon_m, lon_s, lon_dir)
        return lat, lon

    # Fallback to comma/space separated numbers (e.g. "25.0122, 121.5408" or "1.5, 2.0")
    parts = re.findall(r"[-+]?\d*\.\d+|\d+", coord_str)
    if len(parts) >= 2:
        return float(parts[0]), float(parts[1])

    raise ValueError(f"Could not parse coordinate: '{coord_str}'")


def haversine_distance_m(lat1, lon1, lat2, lon2):
    """Calculate distance in meters between two GPS coordinates using Haversine formula."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def find_serial_port():
    """Auto-detect RX board serial port."""
    if not SERIAL_AVAILABLE:
        return None
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        if "XDS110" in p.description:
            return p.device
    for p in ports:
        if "USB" in p.description or "UART" in p.description:
            return p.device
    return ports[0].device


def record_sample(ser=None, duration=DEFAULT_DURATION, simulate=False, rx_bearing=0.0):
    """
    Record RSSI frames and throughput over `duration` seconds.
    Returns: (avg_rssi, max_rssi, min_rssi, avg_kbps, sample_count)
    """
    frame_buffer = bytearray()
    received_bytes = 0
    rssi_list = []
    
    start_time = time.monotonic()
    
    if simulate or ser is None:
        # Simulation mode: generate synthetic RSSI & throughput data
        angle_rad = math.radians(rx_bearing)
        base_rssi = -45.0 - 15.0 * (math.sin(angle_rad / 2) ** 2)
        
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            if elapsed >= duration:
                break
            remaining = duration - elapsed
            
            sample_rssi = int(base_rssi + (math.sin(now * 3) * 2.0))
            rssi_list.append(sample_rssi)
            received_bytes += 200  # Simulate 200-byte frame
            
            current_kbps = (received_bytes * 8) / max(elapsed, 0.001) / 1000
            print(f"\r[Sampling...] Remaining: {remaining:4.1f}s | Bearing: {rx_bearing:5.1f}° | Live RSSI: {sample_rssi:4d} dBm | Rate: {current_kbps:6.1f} kb/s", end="", flush=True)
            time.sleep(0.05)
    else:
        # Hardware serial reading mode
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        while True:
            now = time.monotonic()
            elapsed = now - start_time
            if elapsed >= duration:
                break
            remaining = duration - elapsed

            try:
                data = ser.read(ser.in_waiting or 1)
                if data:
                    frame_buffer.extend(data)
            except Exception as e:
                print(f"\nSerial read warning: {e}")

            while len(frame_buffer) >= 3:
                payload_len = frame_buffer[0]
                frame_len = payload_len + 3

                if payload_len == 0:
                    del frame_buffer[0]
                    continue
                if len(frame_buffer) < frame_len:
                    break
                if frame_buffer[payload_len + 2] != 0x00:
                    del frame_buffer[0]
                    continue

                rssi = frame_buffer[payload_len + 1] - 256
                rssi_list.append(rssi)
                received_bytes += payload_len
                del frame_buffer[:frame_len]

            current_kbps = (received_bytes * 8) / max(elapsed, 0.001) / 1000
            rssi_str = f"{rssi_list[-1]:4d} dBm" if rssi_list else "waiting"
            print(f"\r[Sampling...] Remaining: {remaining:4.1f}s | Bearing: {rx_bearing:5.1f}° | Live RSSI: {rssi_str} | Rate: {current_kbps:6.1f} kb/s", end="", flush=True)
            time.sleep(0.01)

    print() # Newline after sampling completion

    if not rssi_list:
        print("Warning: No RSSI frames received during sampling!")
        return None, None, None, 0.0, 0

    total_elapsed = max(time.monotonic() - start_time, 0.001)
    avg_rssi = float(sum(rssi_list) / len(rssi_list))
    max_rssi = float(max(rssi_list))
    min_rssi = float(min(rssi_list))
    avg_kbps = float((received_bytes * 8) / total_elapsed / 1000)
    sample_count = len(rssi_list)

    return avg_rssi, max_rssi, min_rssi, avg_kbps, sample_count


class CSVLogger:
    """Handles logging measurement records to CSV, with support for new creation or editing/appending."""
    CSV_HEADERS = [
        "timestamp",
        "rx_coord_raw",
        "tx_coord_raw",
        "rx_lat",
        "rx_lon",
        "tx_lat",
        "tx_lon",
        "calculated_distance_m",
        "rx_bearing_deg",
        "turn_left_accum_deg",
        "avg_rssi_dbm",
        "max_rssi_dbm",
        "min_rssi_dbm",
        "avg_kbps",
        "sample_count"
    ]

    def __init__(self, filepath=DEFAULT_CSV_PATH, overwrite=False):
        self.filepath = filepath
        self.overwrite = overwrite
        self._initialize_file()

    def _initialize_file(self):
        if self.overwrite or not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def count_records(self):
        if not os.path.exists(self.filepath):
            return 0
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                return max(len(rows) - 1, 0)
        except Exception:
            return 0

    def log(self, rx_raw, tx_raw, rx_lat, rx_lon, tx_lat, tx_lon, distance_m, rx_bearing, turn_accum, avg_rssi, max_rssi, min_rssi, avg_kbps, sample_count):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = [
            timestamp,
            str(rx_raw),
            str(tx_raw),
            f"{rx_lat:.6f}" if rx_lat is not None else "",
            f"{rx_lon:.6f}" if rx_lon is not None else "",
            f"{tx_lat:.6f}" if tx_lat is not None else "",
            f"{tx_lon:.6f}" if tx_lon is not None else "",
            f"{distance_m:.2f}" if distance_m is not None else "",
            f"{rx_bearing:.1f}",
            f"{turn_accum:.1f}",
            f"{avg_rssi:.2f}",
            f"{max_rssi:.2f}",
            f"{min_rssi:.2f}",
            f"{avg_kbps:.2f}",
            sample_count
        ]
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(record)
        print(f"--> Saved record to CSV ({self.filepath})")


def prompt_input(text):
    """Prompt user input with exit check."""
    val = input(text).strip()
    if val.lower() in ["esc", "exit", "quit", "q"]:
        return None
    return val


def select_csv_mode(default_filepath):
    """
    Prompt user to select dataset operation mode:
      [1] Create a NEW CSV file (overwrite/reset)
      [2] Edit / Append to existing CSV file
    Returns: (filepath, overwrite_bool)
    """
    print("\nPlease select CSV Dataset Mode:")
    print(f"  [1] Create a NEW dataset (reset CSV file)")
    print(f"  [2] Edit / Append to existing CSV file ('{default_filepath}')")
    print("  [ESC/q] Exit")

    choice = prompt_input("\nEnter choice [1/2/esc]: ")
    if choice is None:
        print("Exiting.")
        sys.exit(0)

    if choice == "1":
        file_input = prompt_input(f"Enter file path for NEW CSV dataset [Press Enter for '{default_filepath}']: ")
        if file_input is None:
            print("Exiting.")
            sys.exit(0)
        target_path = file_input if file_input else default_filepath
        print(f"--> Creating NEW CSV dataset: {os.path.abspath(target_path)}")
        return target_path, True

    elif choice == "2":
        file_input = prompt_input(f"Enter file path of existing CSV dataset [Press Enter for '{default_filepath}']: ")
        if file_input is None:
            print("Exiting.")
            sys.exit(0)
        target_path = file_input if file_input else default_filepath
        
        if os.path.exists(target_path):
            existing_count = CSVLogger(target_path, overwrite=False).count_records()
            print(f"--> Appending to existing CSV dataset: {os.path.abspath(target_path)} ({existing_count} records currently)")
        else:
            print(f"--> Target CSV file '{target_path}' does not exist yet. A new file will be created.")
        return target_path, False

    else:
        print("Invalid choice. Defaulting to Append mode.")
        return default_filepath, False


def main():
    parser = argparse.ArgumentParser(description="Receiver 360° RSSI Rotation Test Script (iPhone GPS Supported)")
    parser.add_argument("--port", default=None, help="RX serial port (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate (default: 921600)")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="CSV file path (default: rssi_rotation_data.csv)")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Sampling duration per step in seconds (default: 5.0)")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode without serial hardware")
    args = parser.parse_args()

    print("=" * 72)
    print("       RECEIVER 360° RSSI ROTATION & DISTANCE TEST SYSTEM       ")
    print("       (Supports iPhone GPS format e.g. 25°0'44\" 121°32'27\")      ")
    print("=" * 72)

    # Prompt user for CSV mode (Create New vs Edit/Append)
    csv_path, overwrite_flag = select_csv_mode(args.csv)
    logger = CSVLogger(csv_path, overwrite=overwrite_flag)

    # Connect serial port or use simulation
    ser = None
    simulate_mode = args.simulate

    if not simulate_mode:
        port = args.port if args.port else find_serial_port()
        if not port:
            print("\nNotice: No serial port found. Enable SIMULATION mode? (y/n)")
            ans = input("> ").strip().lower()
            if ans in ["y", "yes"]:
                simulate_mode = True
                print("--> Operating in SIMULATION mode.")
            else:
                print("Error: Serial port required. Exiting.")
                sys.exit(1)
        else:
            try:
                ser = serial.Serial(port, args.baud, timeout=0.1)
                print(f"Connected to RX serial port: {port} @ {args.baud} baud\n")
            except Exception as err:
                print(f"Error opening serial port {port}: {err}")
                print("Switching to simulation mode...")
                simulate_mode = True

    # Step 1: Input Receiver Coordinate & Initial Facing
    print("\n--- [RECEIVER SETUP] ---")
    while True:
        rx_raw = prompt_input("Enter Receiver Coordinate (iPhone GPS e.g. 25°0'44\" 121°32'27\" or X,Y): ")
        if rx_raw is None:
            print("Exiting.")
            sys.exit(0)
        try:
            rx_lat, rx_lon = parse_gps_coord(rx_raw)
            print(f"  [Parsed RX GPS] Lat: {rx_lat:.6f}°, Lon: {rx_lon:.6f}°")
            break
        except Exception as err:
            print(f"  Warning: {err}. Please re-enter coordinate.")

    while True:
        facing_str = prompt_input("Enter Receiver Initial Compass Facing Bearing (0-360°): ")
        if facing_str is None:
            print("Exiting.")
            sys.exit(0)
        try:
            initial_facing = float(facing_str) % 360.0
            break
        except ValueError:
            print("Invalid angle. Please enter a valid number (e.g. 0, 90, 180).")

    print(f"\n[OK] Receiver Position set to: '{rx_raw}' | Initial Facing: {initial_facing:.1f}° Compass Bearing\n")

    # Step 2: Outer loop for Transmitter placement
    tx_count = 0
    try:
        while True:
            tx_count += 1
            print("=" * 72)
            print(f"--- [TRANSMITTER POSITION #{tx_count}] ---")
            tx_raw = prompt_input("Enter Transmitter Coordinate (iPhone GPS e.g. 25°0'45\" 121°32'29\" or 'esc' to exit): ")
            if tx_raw is None:
                print("Finishing all tests. Exiting.")
                break

            tx_lat, tx_lon = None, None
            calculated_dist_m = None
            try:
                tx_lat, tx_lon = parse_gps_coord(tx_raw)
                calculated_dist_m = haversine_distance_m(rx_lat, rx_lon, tx_lat, tx_lon)
                print(f"  [Parsed TX GPS] Lat: {tx_lat:.6f}°, Lon: {tx_lon:.6f}°")
                print(f"  [Calculated Distance] {calculated_dist_m:.2f} meters between RX and TX")
            except Exception as err:
                print(f"  Notice: Coordinate treated as raw position ({err})")

            print(f"\n>> Transmitter placed at: '{tx_raw}'")
            print(f">> Starting 360° Rotation Test ({TOTAL_STEPS} steps x {STEP_DEGREES}° x {args.duration}s)...")
            print("-" * 72)

            # Step 3: Inner 360° rotation loop (12 steps of 30°)
            current_facing = initial_facing

            for step in range(1, TOTAL_STEPS + 1):
                turn_accum = (step - 1) * STEP_DEGREES
                
                print(f"\n>>> Step {step}/{TOTAL_STEPS} | Turned Accum: {turn_accum}° | Compass Bearing: {current_facing:.1f}°")
                print(f"Starting {args.duration:g}-second RSSI recording for TX at {tx_raw}...")

                avg_rssi, max_rssi, min_rssi, avg_kbps, samples = record_sample(
                    ser=ser,
                    duration=args.duration,
                    simulate=simulate_mode,
                    rx_bearing=current_facing
                )

                if samples > 0 and avg_rssi is not None:
                    print(f"[{args.duration:g}s OK] Avg RSSI: {avg_rssi:.2f} dBm (Max: {max_rssi:.1f}, Min: {min_rssi:.1f}) | Throughput: {avg_kbps:.2f} kb/s | Samples: {samples}")
                    logger.log(
                        rx_raw=rx_raw,
                        tx_raw=tx_raw,
                        rx_lat=rx_lat,
                        rx_lon=rx_lon,
                        tx_lat=tx_lat,
                        tx_lon=tx_lon,
                        distance_m=calculated_dist_m,
                        rx_bearing=current_facing,
                        turn_accum=turn_accum,
                        avg_rssi=avg_rssi,
                        max_rssi=max_rssi,
                        min_rssi=min_rssi,
                        avg_kbps=avg_kbps,
                        sample_count=samples
                    )
                else:
                    print("[!] Warning: Sampling failed or 0 frames received. Point skipped.")

                # If not the last step, prompt user to turn left STEP_DEGREES
                if step < TOTAL_STEPS:
                    next_facing = (current_facing - STEP_DEGREES) % 360.0
                    print("\n" + "-" * 50)
                    print(f"  [ACTION REQUIRED] Turn receiver LEFT by {STEP_DEGREES}°")
                    print(f"  Current Bearing: {current_facing:.1f}°  -->  Target Bearing: {next_facing:.1f}°")
                    print("  Please use compass to align receiver to target bearing.")
                    print("-" * 50)

                    ready = prompt_input("Type 'ready' (or press Enter) when aligned [or 'esc' to stop TX test]: ")
                    if ready is None:
                        print("Stopping 360° sweep for current TX position.")
                        break

                    current_facing = next_facing

            print(f"\n[OK] 360° Rotation Sweep Complete for TX Coordinate: {tx_raw}!\n")

    except KeyboardInterrupt:
        print("\nTest interrupted by user (Ctrl+C). Exiting.")
    finally:
        if ser:
            ser.close()
        print(f"Data successfully saved to: {os.path.abspath(csv_path)}")


if __name__ == "__main__":
    main()