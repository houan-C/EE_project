#!/usr/bin/env python3
"""
Receiver RSSI Heatmap Testing Script (rx_heatmap_test.py)

Features:
- Parse iPhone GPS coordinates in DMS (e.g., 25°7'40" 121°29'56" or 25°7'40"N 121°29'56"E) or Decimal formats.
- Connect to serial radio receiver (CC1352) or run in simulation mode.
- 30-second RSSI average measurement with progress reporting.
- Calculate distance, bearing, and directional antenna relative angle.
- 10m resolution grid graph and smoothed RSSI heatmap output (saved to CSV and PNG).
- Session management: Create new session or append/edit existing session.
"""

import sys
import os
import re
import math
import time
import argparse
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# ============================================================
# === Coordinate Parsing (iPhone GPS DMS & Decimal Degrees) ===
# ============================================================

def parse_dms_component(dms_str):
    """
    Parses a single coordinate component (Lat or Lon) in DMS or Decimal format.
    Supports formats like:
      - 25°7'40" or 25°7'40.5"
      - 25° 7' 40" N
      - 121°29'56" or 121° 29' 56" E
      - 25 7 40
      - 25.12777
    """
    s = dms_str.strip()
    
    # Check direction (N, S, E, W)
    sign = 1.0
    if re.search(r'[SWsw]', s):
        sign = -1.0
    
    # Clean string: replace common symbols with spaces
    # Standardize degree °, minute ', second ", unicode quotes
    cleaned = re.sub(r'[°\'"″’`\t,NnSsEeWw]', ' ', s)
    tokens = [t for t in cleaned.split() if t]
    
    if not tokens:
        raise ValueError(f"Invalid coordinate format: {dms_str}")
        
    try:
        if len(tokens) == 1:
            # Decimal degree string
            val = float(tokens[0])
        elif len(tokens) == 2:
            # Degrees and minutes
            deg = float(tokens[0])
            minute = float(tokens[1])
            val = deg + (minute / 60.0)
        elif len(tokens) >= 3:
            # Degrees, minutes, seconds
            deg = float(tokens[0])
            minute = float(tokens[1])
            sec = float(tokens[2])
            val = deg + (minute / 60.0) + (sec / 3600.0)
        else:
            raise ValueError()
        
        return sign * val
    except Exception as e:
        raise ValueError(f"Could not parse coordinate '{dms_str}': {e}")


def parse_coordinates(input_str):
    """
    Parses a string containing both Latitude and Longitude.
    Example iPhone GPS formats:
      - 25°7'40" 121°29'56"
      - 25°7'40"N 121°29'56"E
      - 25° 7' 40", 121° 29' 56"
      - 25.12777, 121.49888
    Returns (lat, lon) in decimal degrees.
    """
    input_str = input_str.strip()
    
    # Try splitting by comma first if present
    if ',' in input_str:
        parts = input_str.split(',', 1)
        lat = parse_dms_component(parts[0])
        lon = parse_dms_component(parts[1])
        return lat, lon

    # Regex matching two DMS patterns with ° ' "
    # iPhone format: 25°7'40" 121°29'56"
    dms_pattern = r'(\d+[\s°\d\'".″’`]+[NnSs]?)\s+(\d+[\s°\d\'".″’`]+[EeWw]?)'
    match = re.search(dms_pattern, input_str)
    if match:
        try:
            lat = parse_dms_component(match.group(1))
            lon = parse_dms_component(match.group(2))
            return lat, lon
        except ValueError:
            pass

    # Fallback space split
    tokens = input_str.split()
    if len(tokens) == 2:
        return parse_dms_component(tokens[0]), parse_dms_component(tokens[1])
    elif len(tokens) == 6:
        # e.g., 25 7 40 121 29 56
        lat_str = f"{tokens[0]}°{tokens[1]}'{tokens[2]}\""
        lon_str = f"{tokens[3]}°{tokens[4]}'{tokens[5]}\""
        return parse_dms_component(lat_str), parse_dms_component(lon_str)
    
    raise ValueError(f"Unrecognized coordinate string format: '{input_str}'\n"
                     f"Expected format example: 25°7'40\" 121°29'56\" or 25.12777, 121.49888")


# ============================================================
# === Geospatial & Antenna Mathematics ===
# ============================================================

def geo_distance_and_bearing(lat1, lon1, lat2, lon2):
    """
    Computes distance (meters), initial bearing (degrees from North),
    and relative Cartesian offsets (dx, dy) in meters from (lat1, lon1) to (lat2, lon2).
    """
    # Earth radius in meters
    R = 6371000.0
    
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    # Flat-earth approximation for local area (accurate for < 50km)
    mean_lat = (lat1_r + lat2_r) / 2.0
    dx = dlon * math.cos(mean_lat) * R  # East offset in meters
    dy = dlat * R                       # North offset in meters
    
    distance = math.sqrt(dx * dx + dy * dy)
    
    # True bearing relative to North (0 deg = North, 90 deg = East)
    bearing_rad = math.atan2(dx, dy)
    bearing_deg = (math.degrees(bearing_rad) + 360.0) % 360.0
    
    return distance, bearing_deg, dx, dy


def calculate_relative_angle(bearing_deg, rx_facing_deg):
    """
    Calculates TX direction relative to RX antenna facing direction.
      0 deg: Directly in front of RX directional antenna main lobe.
     +90 deg: Directly to the right of RX antenna.
    -90 deg: Directly to the left of RX antenna.
     180 deg: Directly behind RX antenna.
    """
    rel = (bearing_deg - rx_facing_deg + 180.0) % 360.0 - 180.0
    return rel


# ============================================================
# === Serial Communication & RSSI Reader ===
# ============================================================

def find_serial_port():
    if not HAS_SERIAL:
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


def record_rssi_30s(ser=None, simulate=False, rx_lat=0, rx_lon=0, tx_lat=0, tx_lon=0, rx_facing=0):
    """
    Records RSSI values for 30 seconds.
    Returns: avg_rssi, min_rssi, max_rssi, std_rssi, samples_count
    """
    duration = 30.0
    start_time = time.time()
    rssi_samples = []
    
    print(f"\n[Recording] Starting 30-second RSSI sampling session...")
    
    if simulate or ser is None:
        # Generate realistic simulated RSSI based on log-distance path loss + antenna pattern
        dist, bearing, _, _ = geo_distance_and_bearing(rx_lat, rx_lon, tx_lat, tx_lon)
        dist = max(dist, 1.0)
        rel_angle = calculate_relative_angle(bearing, rx_facing)
        
        # Antenna directional gain pattern (3dB beamwidth approx 60 deg)
        antenna_gain_db = 12.0 * max(0, math.cos(math.radians(rel_angle / 2.0)))**4 - 10.0
        # Log distance path loss model: P(d) = P(1m) - 10 * n * log10(d)
        expected_rssi = -35.0 - 20.0 * math.log10(dist) + antenna_gain_db
        expected_rssi = max(-110.0, min(-20.0, expected_rssi))
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            # Add Gaussian noise
            sample = expected_rssi + np.random.normal(0.0, 1.8)
            rssi_samples.append(sample)
            
            print(f"\r  [SIMULATION] Time remaining: {remaining:4.1f}s | Current RSSI: {sample:6.1f} dBm | Samples: {len(rssi_samples)}", end="")
            sys.stdout.flush()
            time.sleep(0.2)
    else:
        # Hardware serial reader
        chunk_buffer = bytearray()
        ser.reset_input_buffer()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            try:
                data = ser.read(ser.in_waiting or 1)
                if data:
                    chunk_buffer.extend(data)
                    # DSSS MAC framing parser as used in repo: chunk_buffer[0] = payloadLen, payloadLen+1 = rssi byte
                    while len(chunk_buffer) >= 3:
                        payload_len = chunk_buffer[0]
                        if len(chunk_buffer) < payload_len + 3:
                            break
                        # Convert unsigned byte to signed RSSI in dBm
                        rssi_val = chunk_buffer[payload_len + 1] - 256
                        if -120 <= rssi_val <= 0:
                            rssi_samples.append(float(rssi_val))
                        chunk_buffer = chunk_buffer[payload_len + 3:]
            except Exception as e:
                print(f"\n[Serial Error] {e}")
                time.sleep(0.1)
                
            curr = rssi_samples[-1] if rssi_samples else 0.0
            print(f"\r  [HARDWARE] Time remaining: {remaining:4.1f}s | Current RSSI: {curr:6.1f} dBm | Samples: {len(rssi_samples)}", end="")
            sys.stdout.flush()
            time.sleep(0.05)

    print("\n[Recording] 30-second recording completed.")
    
    if not rssi_samples:
        print("[Warning] No valid RSSI samples collected during 30s period! Using fallback -95 dBm.")
        return -95.0, -95.0, -95.0, 0.0, 0
        
    rssi_arr = np.array(rssi_samples)
    avg_rssi = float(np.mean(rssi_arr))
    min_rssi = float(np.min(rssi_arr))
    max_rssi = float(np.max(rssi_arr))
    std_rssi = float(np.std(rssi_arr))
    count = len(rssi_samples)
    
    print(f"  --> Average RSSI: {avg_rssi:.2f} dBm (Min: {min_rssi:.1f}, Max: {max_rssi:.1f}, Std: {std_rssi:.2f}, Count: {count})")
    return avg_rssi, min_rssi, max_rssi, std_rssi, count


# ============================================================
# === Heatmap Generation & Grid Graph Plotting ===
# ============================================================

def generate_heatmap_plot(records, rx_lat, rx_lon, rx_facing, output_image_path="rssi_heatmap.png", show_window=False):
    """
    Renders 10m grid heatmap plot with smoothed RSSI contour, RX directional orientation,
    and measured TX points.
    """
    if not records:
        print("[Plot] No records available to generate heatmap.")
        return

    # Extract coordinates and RSSI
    # Calculate dx, dy relative to RX (0,0) in meters
    pts_x = []
    pts_y = []
    rssi_vals = []
    tx_labels = []
    
    for idx, r in enumerate(records, start=1):
        d_m = float(r['Distance_m'])
        b_deg = float(r['Bearing_deg'])
        b_rad = math.radians(b_deg)
        dx = d_m * math.sin(b_rad)  # East
        dy = d_m * math.cos(b_rad)  # North
        pts_x.append(dx)
        pts_y.append(dy)
        rssi_vals.append(float(r['Avg_RSSI_dBm']))
        tx_labels.append(f"P{idx}: {float(r['Avg_RSSI_dBm']):.1f}dBm")
        
    pts_x = np.array(pts_x)
    pts_y = np.array(pts_y)
    rssi_vals = np.array(rssi_vals)

    # 10m Grid bounds determination with padding
    grid_res = 10.0  # 10 meters resolution
    margin = 30.0    # 30 meters margin around data
    
    min_x = min(0.0, np.min(pts_x)) - margin
    max_x = max(0.0, np.max(pts_x)) + margin
    min_y = min(0.0, np.min(pts_y)) - margin
    max_y = max(0.0, np.max(pts_y)) + margin

    # Align min/max bounds to 10m grid boundaries
    min_x = math.floor(min_x / grid_res) * grid_res
    max_x = math.ceil(max_x / grid_res) * grid_res
    min_y = math.floor(min_y / grid_res) * grid_res
    max_y = math.ceil(max_y / grid_res) * grid_res

    # Create 2D mesh grid
    grid_x, grid_y = np.mgrid[min_x:max_x:0.5, min_y:max_y:0.5]

    # Interpolation: combine RX (0,0) and TX points for spatial heatmap
    rx_x, rx_y = 0.0, 0.0

    fit_x = np.append(pts_x, rx_x)
    fit_y = np.append(pts_y, rx_y)
    # Estimate RX near-field RSSI as max(measured) + 15dB
    rx_est_rssi = max(np.max(rssi_vals) + 15.0, -25.0)
    fit_rssi = np.append(rssi_vals, rx_est_rssi)

    if len(fit_x) >= 3:
        try:
            grid_z = griddata((fit_x, fit_y), fit_rssi, (grid_x, grid_y), method='cubic')
            nan_mask = np.isnan(grid_z)
            if np.any(nan_mask):
                grid_z_near = griddata((fit_x, fit_y), fit_rssi, (grid_x, grid_y), method='nearest')
                grid_z[nan_mask] = grid_z_near[nan_mask]
        except Exception:
            grid_z = griddata((fit_x, fit_y), fit_rssi, (grid_x, grid_y), method='nearest')
    else:
        grid_z = griddata((fit_x, fit_y), fit_rssi, (grid_x, grid_y), method='nearest')

    # Apply Gaussian smoothing over 10m grid
    grid_z_smoothed = gaussian_filter(grid_z, sigma=2.0)

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    
    # Heatmap color mesh
    vmin = min(-105.0, np.min(rssi_vals) - 5)
    vmax = max(-30.0, np.max(rssi_vals) + 5)
    
    contour = ax.contourf(grid_x, grid_y, grid_z_smoothed, levels=30, cmap='YlOrRd_r', vmin=vmin, vmax=vmax, alpha=0.85)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Average RSSI (dBm)', fontsize=12, fontweight='bold')

    # Add 10m grid lines
    x_ticks = np.arange(min_x, max_x + 1, grid_res)
    y_ticks = np.arange(min_y, max_y + 1, grid_res)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.6, alpha=0.6)

    # Plot Receiver (RX) at (0,0)
    ax.scatter([0], [0], color='blue', s=180, zorder=5, marker='o', edgecolors='black', label=f'RX (Facing: {rx_facing:.1f}°)')
    
    # Plot RX Antenna Facing Direction Arrow (Main Lobe)
    facing_rad = math.radians(rx_facing)
    arrow_dx = 15.0 * math.sin(facing_rad)
    arrow_dy = 15.0 * math.cos(facing_rad)
    ax.arrow(0, 0, arrow_dx, arrow_dy, head_width=4, head_length=5, fc='blue', ec='darkblue', zorder=6, length_includes_head=True)
    
    # Draw RX directional beam cone (approx +-30 deg)
    cone_left_rad = math.radians(rx_facing - 30)
    cone_right_rad = math.radians(rx_facing + 30)
    ax.plot([0, 25*math.sin(cone_left_rad)], [0, 25*math.cos(cone_left_rad)], color='blue', linestyle=':', linewidth=1.5)
    ax.plot([0, 25*math.sin(cone_right_rad)], [0, 25*math.cos(cone_right_rad)], color='blue', linestyle=':', linewidth=1.5)

    # Plot Measured Transmitter (TX) points
    ax.scatter(pts_x, pts_y, c=rssi_vals, cmap='YlOrRd_r', vmin=vmin, vmax=vmax, s=120, edgecolors='black', linewidth=1.5, zorder=7, label='TX Points')
    
    for x, y, lbl in zip(pts_x, pts_y, tx_labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(8, 8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.85),
                    fontsize=9, fontweight='bold', zorder=8)

    ax.set_xlabel('East Displacement (meters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('North Displacement (meters)', fontsize=11, fontweight='bold')
    ax.set_title(f'10m Resolution RSSI Heatmap\nRX Coord: ({rx_lat:.6f}, {rx_lon:.6f}) | RX Facing: {rx_facing}°', fontsize=13, fontweight='bold')
    ax.set_aspect('equal', 'box')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=200)
    print(f"[Heatmap] Saved updated heatmap image to '{output_image_path}'.")
    
    if show_window:
        plt.show(block=False)
        plt.pause(0.5)
    plt.close(fig)


# ============================================================
# === Main Interactive Test CLI ===
# ============================================================

def load_session_csv(csv_path):
    records = []
    rx_info = {}
    if not os.path.exists(csv_path):
        return records, rx_info
        
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    if records:
        rx_info['rx_lat'] = float(records[0]['RX_Lat'])
        rx_info['rx_lon'] = float(records[0]['RX_Lon'])
        rx_info['rx_facing'] = float(records[0]['RX_Facing'])
        
    return records, rx_info


def save_record_csv(csv_path, record):
    fieldnames = [
        'Session_ID', 'Timestamp', 'RX_Lat', 'RX_Lon', 'RX_Facing',
        'TX_Lat', 'TX_Lon', 'Distance_m', 'Bearing_deg', 'Relative_Angle_deg',
        'Avg_RSSI_dBm', 'Min_RSSI', 'Max_RSSI', 'Std_RSSI', 'Samples_Count'
    ]
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def main():
    parser = argparse.ArgumentParser(description="Receiver RSSI Heatmap Test Script")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode without hardware")
    parser.add_argument("--port", type=str, default=None, help="Serial COM port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    args = parser.parse_args()

    print("==========================================================")
    print("      CC1352 Directional Receiver RSSI Heatmap Tester     ")
    print("==========================================================")

    # Hardware / Serial initialization
    ser = None
    simulate = args.simulate
    if not simulate:
        port = args.port or find_serial_port()
        if port and HAS_SERIAL:
            try:
                ser = serial.Serial(port, args.baud, timeout=1.0)
                print(f"[Serial] Connected successfully to port {port} @ {args.baud} baud.")
            except Exception as e:
                print(f"[Serial] Failed to open port {port}: {e}")
                print("[Notice] Falling back to Simulation Mode.")
                simulate = True
        else:
            print("[Serial] No hardware COM port found.")
            print("[Notice] Falling back to Simulation Mode.")
            simulate = True

    # 1. Choose Session Mode
    print("\nSelect Session Mode:")
    print("  [1] Start a NEW recording session")
    print("  [2] Edit / Append to an EXISTING CSV session file")
    
    choice = input("Enter choice (1 or 2) [Default=1]: ").strip()
    
    records = []
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"rssi_heatmap_{session_id}.csv"
    
    rx_lat, rx_lon, rx_facing = None, None, None

    if choice == "2":
        csv_input = input("Enter existing CSV filepath to append: ").strip()
        if os.path.exists(csv_input):
            csv_file = csv_input
            records, rx_info = load_session_csv(csv_file)
            print(f"[Session] Loaded {len(records)} existing test points from '{csv_file}'.")
            if rx_info:
                rx_lat = rx_info['rx_lat']
                rx_lon = rx_info['rx_lon']
                rx_facing = rx_info['rx_facing']
                print(f"[Session] Previous RX Location: ({rx_lat:.6f}, {rx_lon:.6f}), Facing: {rx_facing}°")
        else:
            print(f"[Warning] File '{csv_input}' not found. Starting new session.")

    # 2. Enter Receiver (RX) Coordinate & Facing Direction if not loaded
    if rx_lat is None:
        print("\n--- Enter Receiver (RX) Setup ---")
        while True:
            coord_str = input("Enter Receiver Coordinate (e.g., iPhone format 25°7'40\" 121°29'56\" or 25.1277, 121.4988): ").strip()
            try:
                rx_lat, rx_lon = parse_coordinates(coord_str)
                print(f"  --> Parsed RX Latitude:  {rx_lat:.7f}°")
                print(f"  --> Parsed RX Longitude: {rx_lon:.7f}°")
                break
            except Exception as err:
                print(f"[Error] {err}. Please try again.")

        while True:
            facing_str = input("Enter Receiver Antenna Facing Direction in degrees (0=North, 90=East, 180=South, 270=West): ").strip()
            try:
                rx_facing = float(facing_str) % 360.0
                print(f"  --> RX Directional Antenna Facing: {rx_facing:.1f}°")
                break
            except ValueError:
                print("[Error] Invalid angle. Enter a numeric degree value.")

    # Initial plot rendering if existing records present
    if records:
        generate_heatmap_plot(records, rx_lat, rx_lon, rx_facing, output_image_path="rssi_heatmap.png")

    # 3. Main TX Point Sampling Loop
    pt_count = len(records)
    
    while True:
        pt_count += 1
        print(f"\n==========================================================")
        print(f"            Measurement Point #{pt_count}                 ")
        print(f"==========================================================")
        
        # Step A: Wait for Transmitter Ready
        ready_in = input("\nWhen Transmitter (TX) is at position and ready, type 'ready' (or press Enter) [or 'q' to finish]: ").strip().lower()
        if ready_in == 'q':
            print("\nSession ended by user.")
            break

        # Step B: 30-Second RSSI Recording
        print("\nGet ready! Starting 30s RSSI averaging...")
        avg_rssi, min_rssi, max_rssi, std_rssi, samples_cnt = record_rssi_30s(
            ser=ser, simulate=simulate,
            rx_lat=rx_lat, rx_lon=rx_lon, tx_lat=rx_lat, tx_lon=rx_lon, rx_facing=rx_facing
        )

        # Step C: Enter TX Coordinate
        while True:
            tx_coord_str = input(f"Enter TX #{pt_count} Coordinate (e.g., iPhone format 25°7'45\" 121°29'50\"): ").strip()
            try:
                tx_lat, tx_lon = parse_coordinates(tx_coord_str)
                print(f"  --> Parsed TX Latitude:  {tx_lat:.7f}°")
                print(f"  --> Parsed TX Longitude: {tx_lon:.7f}°")
                break
            except Exception as err:
                print(f"[Error] {err}. Please try again.")

        # Step D: Compute Distance, Bearing, and Relative Angle
        distance_m, bearing_deg, dx, dy = geo_distance_and_bearing(rx_lat, rx_lon, tx_lat, tx_lon)
        rel_angle_deg = calculate_relative_angle(bearing_deg, rx_facing)

        print("\n--- Point Measurement Summary ---")
        print(f"  TX Coordinate:       ({tx_lat:.7f}°, {tx_lon:.7f}°)")
        print(f"  Distance from RX:    {distance_m:.2f} meters")
        print(f"  True Bearing:        {bearing_deg:.1f}°")
        print(f"  Angle Rel to Antenna:{rel_angle_deg:+.1f}°")
        print(f"  Average RSSI (30s):  {avg_rssi:.2f} dBm")

        # Step E: Record to CSV
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            'Session_ID': session_id,
            'Timestamp': timestamp_str,
            'RX_Lat': f"{rx_lat:.7f}",
            'RX_Lon': f"{rx_lon:.7f}",
            'RX_Facing': f"{rx_facing:.1f}",
            'TX_Lat': f"{tx_lat:.7f}",
            'TX_Lon': f"{tx_lon:.7f}",
            'Distance_m': f"{distance_m:.2f}",
            'Bearing_deg': f"{bearing_deg:.1f}",
            'Relative_Angle_deg': f"{rel_angle_deg:.1f}",
            'Avg_RSSI_dBm': f"{avg_rssi:.2f}",
            'Min_RSSI': f"{min_rssi:.1f}",
            'Max_RSSI': f"{max_rssi:.1f}",
            'Std_RSSI': f"{std_rssi:.2f}",
            'Samples_Count': str(samples_cnt)
        }
        
        save_record_csv(csv_file, record)
        records.append(record)
        print(f"[CSV] Record appended to '{csv_file}'.")

        # Step F: Update 10m Grid Heatmap Plot
        generate_heatmap_plot(records, rx_lat, rx_lon, rx_facing, output_image_path="rssi_heatmap.png")

        # Prompt next point
        cont = input("\nAdd another measurement point? (y/n) [Default=y]: ").strip().lower()
        if cont == 'n':
            break

    print("\n==========================================================")
    print(f"Session complete! Recorded {len(records)} total points.")
    print(f"  - CSV Data File:     {os.path.abspath(csv_file)}")
    print(f"  - Smoothed Heatmap:  {os.path.abspath('rssi_heatmap.png')}")
    print("==========================================================")


if __name__ == "__main__":
    main()
