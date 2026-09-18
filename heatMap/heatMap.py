#!/usr/bin/env python3
"""
Antenna Directivity & Distance 2D RSSI HeatMap Generator

Measures RSSI from receiver board over serial, logs measurements to CSV,
and generates 2D spatial heat maps & polar radiation patterns.

Supports iPhone default GPS coordinate format (e.g. 25°0'44" 121°32'27")
as well as standard decimal coordinates and distance inputs in meters.
"""

import argparse
import csv
import math
import os
import re
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# Default Configuration
DEFAULT_CSV_PATH = "antenna_heatmap_data_2d.csv"
DEFAULT_BAUD = 921600
MEASUREMENT_DURATION_SEC = 5.0


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
    # Matches formats like: 25°0'44", 25° 0' 44" N, 25°0'44.2"N
    dms_pattern = r"(\d+)\s*°\s*(\d+)\s*['’]\s*([\d.]+)\s*[\"”]?\s*([NSEWnsew])?"
    dms_matches = re.findall(dms_pattern, coord_str)
    
    if len(dms_matches) >= 2:
        lat_d, lat_m, lat_s, lat_dir = dms_matches[0]
        lon_d, lon_m, lon_s, lon_dir = dms_matches[1]
        lat = dms_to_dd(lat_d, lat_m, lat_s, lat_dir)
        lon = dms_to_dd(lon_d, lon_m, lon_s, lon_dir)
        return lat, lon

    # Fallback to comma/space separated numbers
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


def collect_rssi_samples(ser=None, duration=MEASUREMENT_DURATION_SEC, simulate=False, distance=1.0, azimuth_cw=0.0):
    """
    Collect RSSI samples for `duration` seconds.
    Returns: (avg_rssi, max_rssi, min_rssi, sample_count)
    """
    rssi_list = []
    start_time = time.monotonic()
    
    if simulate or ser is None:
        # Simulation mode: generate realistic synthetic RSSI based on directional pattern & path loss
        r = max(distance, 0.1)
        path_loss = 20 * math.log10(r) + 30.0  # Approx path loss
        az_rad = math.radians(azimuth_cw)
        directivity_loss = 15 * (math.sin(az_rad / 2) ** 2)
        base_rssi = -28.0 - path_loss - directivity_loss
        
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= duration:
                break
            remaining = duration - elapsed
            sample_rssi = int(base_rssi + np.random.normal(0, 1.2))
            rssi_list.append(sample_rssi)
            print(f"\r[Sampling...] Time remaining: {remaining:4.1f}s | Received: {len(rssi_list):4d} frames | Live RSSI: {sample_rssi:4d} dBm", end="", flush=True)
            time.sleep(0.05)
    else:
        # Real serial reading mode
        frame_buffer = bytearray()
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        while True:
            elapsed = time.monotonic() - start_time
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
                del frame_buffer[:frame_len]

            latest_rssi_str = f"{rssi_list[-1]:4d} dBm" if rssi_list else "waiting"
            print(f"\r[Sampling...] Time remaining: {remaining:4.1f}s | Received: {len(rssi_list):4d} frames | Live RSSI: {latest_rssi_str}", end="", flush=True)
            time.sleep(0.01)

    print() # New line after sampling loop finishes
    
    if not rssi_list:
        print("Warning: No RSSI frames received during sampling duration!")
        return None, None, None, 0

    avg_rssi = float(np.mean(rssi_list))
    max_rssi = float(np.max(rssi_list))
    min_rssi = float(np.min(rssi_list))
    sample_count = len(rssi_list)

    return avg_rssi, max_rssi, min_rssi, sample_count


class CSVManager:
    """Handles reading and writing 2D RSSI dataset CSV files."""
    CSV_HEADERS = [
        "timestamp",
        "distance_m",
        "azimuth_input_deg",
        "azimuth_ccw_deg",
        "avg_rssi_dbm",
        "max_rssi_dbm",
        "min_rssi_dbm",
        "sample_count"
    ]

    def __init__(self, filepath=DEFAULT_CSV_PATH):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def append_record(self, distance_m, azimuth_input_deg, azimuth_ccw_deg, avg_rssi, max_rssi, min_rssi, sample_count):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = [
            timestamp,
            f"{distance_m:.3f}",
            f"{azimuth_input_deg:.2f}",
            f"{azimuth_ccw_deg:.2f}",
            f"{avg_rssi:.2f}",
            f"{max_rssi:.2f}",
            f"{min_rssi:.2f}",
            sample_count
        ]
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(record)
        print(f"--> Record saved to CSV ({self.filepath})")

    def read_records(self):
        records = []
        if not os.path.exists(self.filepath):
            return records
        with open(self.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Support reading both heatMap CSV and rssi_rx_test CSV formats!
                    dist = float(row.get("distance_m", row.get("calculated_distance_m", 1.0)))
                    
                    if "azimuth_input_deg" in row:
                        az_input = float(row["azimuth_input_deg"])
                        az_ccw = float(row.get("azimuth_ccw_deg", (-az_input) % 360.0))
                    elif "turn_left_accum_deg" in row:
                        az_input = float(row["turn_left_accum_deg"])
                        az_ccw = (-az_input) % 360.0
                    else:
                        az_input = float(row.get("rx_bearing_deg", 0.0))
                        az_ccw = (-az_input) % 360.0

                    records.append({
                        "timestamp": row.get("timestamp", ""),
                        "distance_m": dist,
                        "azimuth_input_deg": az_input,
                        "azimuth_ccw_deg": az_ccw,
                        "avg_rssi_dbm": float(row["avg_rssi_dbm"]),
                        "max_rssi_dbm": float(row.get("max_rssi_dbm", row["avg_rssi_dbm"])),
                        "min_rssi_dbm": float(row.get("min_rssi_dbm", row["avg_rssi_dbm"])),
                        "sample_count": int(row.get("sample_count", 1))
                    })
                except (ValueError, KeyError):
                    continue
        return records


def input_with_esc_check(prompt_text):
    """Prompt user for input, checking if 'esc' or exit key was requested."""
    val = input(prompt_text).strip()
    if val.lower() in ["esc", "exit", "quit", "q"]:
        return None
    return val


def generate_2d_heatmap(records, output_image_path="heatmap_2d.png", show_plot=True):
    """Generates 2D Spatial HeatMap and Polar Radiation Pattern from measurement records."""
    if not records:
        print("No valid records to display for 2D HeatMap.")
        return

    # Extract coordinates & RSSI values
    distances = np.array([r["distance_m"] for r in records])
    az_ccw_deg = np.array([r["azimuth_ccw_deg"] for r in records])
    az_input_deg = np.array([r["azimuth_input_deg"] for r in records])
    rssi = np.array([r["avg_rssi_dbm"] for r in records])

    # Convert 2D polar to Cartesian coordinates
    phi = np.radians(az_ccw_deg)
    r = distances

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Figure configuration with dark theme
    fig = plt.figure(figsize=(16, 7), facecolor='#121212')
    
    # -------------------------------------------------------------------------
    # Subplot 1: 2D Spatial Cartesian HeatMap (XY Plane)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(121, facecolor='#121212')
    ax1.set_aspect('equal')
    ax1.tick_params(colors='white', labelsize=10)
    ax1.set_xlabel('X (East / 0° CCW) [m]', color='#00E5FF', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Y (North / 90° CCW) [m]', color='#00E5FF', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_title('2D Spatial Antenna RSSI HeatMap', color='white', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, color='#333333', linestyle=':', alpha=0.6)
    for spine in ax1.spines.values():
        spine.set_color('#444444')

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=np.min(rssi) - 2, vmax=np.max(rssi) + 2)

    # Max distance for axis scaling
    max_dist = np.max(r) * 1.2 if len(r) > 0 else 1.5

    # Draw reference concentric distance circles
    dist_rings = np.linspace(max_dist * 0.25, max_dist * 0.9, 4)
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    for ring_r in dist_rings:
        rx = ring_r * np.cos(circle_angles)
        ry = ring_r * np.sin(circle_angles)
        ax1.plot(rx, ry, color='#555555', linestyle='--', linewidth=0.9, alpha=0.7)
        ax1.text(ring_r * 0.707, ring_r * 0.707, f" {ring_r:.1f}m", color='#888888', fontsize=8, alpha=0.8)

    # 2D Interpolated HeatMap Surface (Contour mesh)
    if len(records) >= 4:
        try:
            grid_x, grid_y = np.mgrid[-max_dist:max_dist:200j, -max_dist:max_dist:200j]
            grid_rssi = griddata((x, y), rssi, (grid_x, grid_y), method='cubic')
            
            # Fill outer NaN region with nearest neighbor interpolation
            nan_mask = np.isnan(grid_rssi)
            if np.any(nan_mask):
                grid_rssi_nearest = griddata((x, y), rssi, (grid_x, grid_y), method='nearest')
                grid_rssi[nan_mask] = grid_rssi_nearest[nan_mask]

            # Mask values outside maximum distance boundary
            grid_r = np.sqrt(grid_x**2 + grid_y**2)
            grid_rssi[grid_r > max_dist] = np.nan

            contour = ax1.contourf(
                grid_x, grid_y, grid_rssi,
                levels=50,
                cmap=cmap,
                norm=norm,
                alpha=0.85
            )
        except Exception as err:
            print(f"Notice: 2D Contour interpolation skipped ({err})")

    # Plot measurement scatter points
    scatter = ax1.scatter(
        x, y,
        c=rssi,
        cmap=cmap,
        norm=norm,
        s=120,
        edgecolors='white',
        linewidth=1.2,
        zorder=5,
        label='Measurement Points'
    )

    # Highlight peak RSSI point
    max_idx = np.argmax(rssi)
    ax1.scatter(
        [x[max_idx]], [y[max_idx]],
        color='#00FF66',
        s=250,
        marker='*',
        edgecolors='black',
        linewidth=1.5,
        zorder=6,
        label=f'Peak RSSI: {rssi[max_idx]:.1f} dBm'
    )

    ax1.annotate(
        f" Peak: {rssi[max_idx]:.1f} dBm\n (r={r[max_idx]:.1f}m, CW={az_input_deg[max_idx]}°)",
        xy=(x[max_idx], y[max_idx]),
        xytext=(x[max_idx] + max_dist * 0.1, y[max_idx] + max_dist * 0.1),
        arrowprops=dict(facecolor='#00FF66', edgecolor='black', shrink=0.08, width=1.5, headwidth=6),
        color='#00FF66',
        fontsize=9,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#222222', edgecolor='#00FF66', alpha=0.85)
    )

    ax1.set_xlim(-max_dist, max_dist)
    ax1.set_ylim(-max_dist, max_dist)
    ax1.legend(loc='lower left', facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax1, shrink=0.8, pad=0.04)
    cbar.set_label('RSSI Signal Strength (dBm)', color='white', fontsize=10, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # -------------------------------------------------------------------------
    # Subplot 2: 2D Polar Directivity Pattern
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(122, projection='polar', facecolor='#121212')
    ax2.set_theta_zero_location('N')  # 0 deg at top
    ax2.set_theta_direction(-1)       # Clockwise positive to match user antenna input!
    ax2.tick_params(colors='white', labelsize=9)
    ax2.set_title('Antenna Directivity Pattern (Polar View)', color='white', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, color='#444444', linestyle=':', alpha=0.7)

    # Sort by input CW angle for clean polar line plotting
    sort_indices = np.argsort(az_input_deg)
    polar_angles_rad = np.radians(az_input_deg[sort_indices])
    polar_rssi = rssi[sort_indices]

    # Map RSSI (e.g. -80 to -30 dBm) to positive radius for polar display
    min_rssi_val = np.min(rssi) - 5
    polar_radius = polar_rssi - min_rssi_val

    # Connect pattern line
    angles_closed = np.append(polar_angles_rad, polar_angles_rad[0])
    radius_closed = np.append(polar_radius, polar_radius[0])

    ax2.plot(angles_closed, radius_closed, color='#00E5FF', linewidth=2, linestyle='-', marker='o', markersize=6, label='Directivity Pattern')
    ax2.fill(angles_closed, radius_closed, color='#00E5FF', alpha=0.2)

    # Customize radial ticks to show actual dBm values
    radial_ticks = np.linspace(np.min(polar_radius), np.max(polar_radius), 4)
    radial_labels = [f"{v + min_rssi_val:.0f} dBm" for v in radial_ticks]
    ax2.set_rticks(radial_ticks)
    ax2.set_yticklabels(radial_labels, color='#AAAAAA', fontsize=8)

    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), facecolor='#222222', edgecolor='#444444', labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"\n[+] 2D HeatMap saved as: {os.path.abspath(output_image_path)}")
    if show_plot and sys.stdin.isatty():
        print("[+] Displaying 2D HeatMap window... (Close window to exit)")
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Antenna Directivity & Distance 2D RSSI HeatMap Generator (iPhone GPS Supported)")
    parser.add_argument("--port", default=None, help="RX serial port (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate (default: 921600)")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="CSV file path (default: antenna_heatmap_data_2d.csv)")
    parser.add_argument("--simulate", action="store_true", help="Run in simulation mode without hardware serial port")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive plot window")
    args = parser.parse_args()

    print("=" * 72)
    print("      ANTENNA 2D RSSI HEATMAP & DIRECTIVITY TEST SYSTEM      ")
    print("      (Supports iPhone GPS format e.g. 25°0'44\" 121°32'27\")      ")
    print("=" * 72)

    csv_mgr = CSVManager(args.csv)
    existing_records = csv_mgr.read_records()
    print(f"Target CSV file: {os.path.abspath(args.csv)}")
    print(f"Currently existing records in CSV: {len(existing_records)}")

    # Prompt user mode selection
    print("\nPlease select operation mode:")
    print("  [1] Start new measurement session / add data to CSV")
    print("  [2] Directly generate 2D HeatMap from existing CSV data")
    print("  [ESC/q] Exit")

    choice = input_with_esc_check("\nEnter choice [1/2/esc]: ")
    if choice is None:
        print("Exiting.")
        sys.exit(0)

    if choice == "2":
        if not existing_records:
            print("No records found in CSV file to generate HeatMap.")
            sys.exit(1)
        generate_2d_heatmap(existing_records, show_plot=not args.no_show)
        sys.exit(0)

    # Mode 1: Measurement loop
    ser = None
    simulate_mode = args.simulate

    if not simulate_mode:
        port = args.port if args.port else find_serial_port()
        if not port:
            print("\nNotice: No serial port found. Would you like to enable SIMULATION mode? (y/n)")
            sim_choice = input("> ").strip().lower()
            if sim_choice in ["y", "yes"]:
                simulate_mode = True
                print("--> Operating in SIMULATION mode.")
            else:
                print("Error: Serial port required. Exiting.")
                sys.exit(1)
        else:
            try:
                ser = serial.Serial(port, args.baud, timeout=0.1)
                print(f"Connected to RX serial port: {port} @ {args.baud} baud")
            except Exception as err:
                print(f"Error opening serial port {port}: {err}")
                print("Switching to simulation mode for testing...")
                simulate_mode = True

    print("\n" + "=" * 72)
    print("INSTRUCTIONS FOR MEASUREMENT:")
    print("  - Distance (r): Distance in meters (OR enter two iPhone GPS DMS coordinates)")
    print("  - Horizontal Angle (Azimuth): Positive angle for Clockwise (CW) rotation")
    print("  - Type 'esc' at any input prompt to stop recording & generate 2D HeatMap immediately.")
    print("=" * 72 + "\n")

    point_count = 0

    try:
        while True:
            point_count += 1
            print(f"\n--- Measurement Point #{point_count} ---")
            
            # Input distance or GPS coordinates
            dist_str = input_with_esc_check("Enter Distance (meters) OR iPhone GPS coord (e.g. 25°0'44\" 121°32'27\") [or 'esc' to finish]: ")
            if dist_str is None:
                print("\n[+] Finish requested. Saving data & building 2D HeatMap...")
                break

            distance_m = 1.0
            try:
                # Try direct float distance first
                distance_m = float(dist_str)
                if distance_m < 0:
                    distance_m = 0.1
            except ValueError:
                # Try parsing as iPhone GPS coordinate relative to a base RX coordinate
                try:
                    rx_lat, rx_lon = parse_gps_coord(dist_str)
                    print(f"  [Parsed GPS] Lat: {rx_lat:.6f}°, Lon: {rx_lon:.6f}°")
                    tx_str = input_with_esc_check("Enter Transmitter iPhone GPS coord (e.g. 25°0'45\" 121°32'28\"): ")
                    if tx_str is None:
                        break
                    tx_lat, tx_lon = parse_gps_coord(tx_str)
                    distance_m = haversine_distance_m(rx_lat, rx_lon, tx_lat, tx_lon)
                    print(f"  [Calculated Distance] {distance_m:.2f} meters")
                except Exception as err:
                    print(f"Invalid distance or GPS input ({err}). Setting default distance to 1.0m.")
                    distance_m = 1.0

            # Input Horizontal Angle (CW positive)
            az_str = input_with_esc_check("Enter Horizontal Angle in degrees (CW positive) [or 'esc' to finish]: ")
            if az_str is None:
                print("\n[+] Finish requested. Saving data & building 2D HeatMap...")
                break
            try:
                azimuth_cw_deg = float(az_str)
            except ValueError:
                print("Invalid input for horizontal angle. Please enter a number.")
                point_count -= 1
                continue

            # Angle conversion: CW positive -> CCW polar coordinate
            azimuth_ccw_deg = (-azimuth_cw_deg) % 360.0

            print(f"\n[>>] Starting 5-second RSSI sampling for (r={distance_m:.2f}m, CW={azimuth_cw_deg}°, CCW={azimuth_ccw_deg:.1f}°)...")
            
            avg_rssi, max_rssi, min_rssi, samples = collect_rssi_samples(
                ser=ser,
                duration=MEASUREMENT_DURATION_SEC,
                simulate=simulate_mode,
                distance=distance_m,
                azimuth_cw=azimuth_cw_deg
            )

            if samples > 0 and avg_rssi is not None:
                print(f"[OK] Completed: Avg RSSI = {avg_rssi:.2f} dBm (Max: {max_rssi:.1f}, Min: {min_rssi:.1f}, Samples: {samples})")
                csv_mgr.append_record(
                    distance_m=distance_m,
                    azimuth_input_deg=azimuth_cw_deg,
                    azimuth_ccw_deg=azimuth_ccw_deg,
                    avg_rssi=avg_rssi,
                    max_rssi=max_rssi,
                    min_rssi=min_rssi,
                    sample_count=samples
                )
            else:
                print("[!] Warning: Sampling failed or 0 samples received. Point skipped.")

    except KeyboardInterrupt:
        print("\n\nMeasurement interrupted by user (Ctrl+C). Generating 2D HeatMap with collected data...")
    finally:
        if ser:
            ser.close()

    # Load all records including newly appended ones and generate 2D HeatMap
    all_records = csv_mgr.read_records()
    if all_records:
        print(f"\nTotal records available: {len(all_records)}. Generating 2D HeatMap...")
        generate_2d_heatmap(all_records, show_plot=not args.no_show)
    else:
        print("\nNo records saved. Exiting without generating HeatMap.")


if __name__ == "__main__":
    main()
