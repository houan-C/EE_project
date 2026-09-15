#!/usr/bin/env python3
"""
Transmitter Continuous Beacon Script (tx_beacon.py)

Runs at the Transmitter (TX) end with omnidirectional antenna.
Continuously broadcasts radio beacon frames over serial/CC1352 hardware.
"""

import sys
import time
import struct
import argparse

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


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


def main():
    parser = argparse.ArgumentParser(description="Transmitter Continuous Beacon Script")
    parser.add_argument("--port", type=str, default=None, help="Serial COM port (e.g., COM4 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--interval", type=float, default=0.05, help="Transmit packet interval in seconds (default: 0.05s)")
    args = parser.parse_args()

    print("==========================================================")
    print("      CC1352 Omnidirectional Transmitter Beacon           ")
    print("==========================================================")

    if not HAS_SERIAL:
        print("[Error] 'pyserial' package is not installed.")
        sys.exit(1)

    port = args.port or find_serial_port()
    if not port:
        print("[Error] No COM port found for CC1352 transmitter.")
        sys.exit(1)

    try:
        ser = serial.Serial(port, args.baud, timeout=1.0)
        print(f"[TX] Connected to serial port {port} @ {args.baud} baud.")
    except Exception as e:
        print(f"[Error] Failed to open serial port {port}: {e}")
        sys.exit(1)

    # Beacon packet format matching CC1352 project
    MAGIC_HEADER = b'FRIEREN'
    # Header format: Magic(7s) + Level(B) + PayloadLen(I)
    HEADER_FORMAT = '<7sBI'
    
    payload = b'HEATMAP_BEACON_BEACON_BEACON_BEACON'
    header = struct.pack(HEADER_FORMAT, MAGIC_HEADER, 1, len(payload))
    packet = header + payload

    print(f"[TX] Continuous beacon transmission started (Interval: {args.interval*1000:.0f} ms)...")
    print("[TX] Press Ctrl+C to stop.\n")

    pkt_count = 0
    start_time = time.time()

    try:
        while True:
            ser.write(packet)
            pkt_count += 1
            elapsed = time.time() - start_time
            rate = pkt_count / elapsed if elapsed > 0 else 0
            
            print(f"\r[TX] Sent {pkt_count} beacon packets | Elapsed: {elapsed:.1f}s ({rate:.1f} pkts/sec)", end="")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n[TX] Transmission stopped by user. Total sent: {pkt_count} packets.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
