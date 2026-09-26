import argparse
import sys
import time

import serial
import serial.tools.list_ports


PAYLOAD_SIZE = 200


def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for port in ports:
        if "XDS110" in port.description:
            return port.device
    for port in ports:
        if "USB" in port.description or "UART" in port.description:
            return port.device
    return ports[0].device


def main():
    parser = argparse.ArgumentParser(description="Simple TX throughput test")
    parser.add_argument("--port", default=None, help="TX board serial port (default: auto-detect)")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--interval", type=float, default=0.02)
    args = parser.parse_args()

    com_port = args.port if args.port else find_serial_port()
    if not com_port:
        print("Error: No serial port found.")
        sys.exit(1)

    try:
        ser = serial.Serial(com_port, args.baud, timeout=1)
    except serial.SerialException as error:
        print(f"Error opening serial port: {error}")
        sys.exit(1)

    payload = bytes(i % 256 for i in range(PAYLOAD_SIZE))
    sent_bytes = 0
    start_time = time.monotonic()
    next_send = start_time

    try:
        print(f"Sending {PAYLOAD_SIZE}-byte payloads on {com_port}")
        print("Press Ctrl+C to stop.\n")

        while True:
            now = time.monotonic()
            if now < next_send:
                time.sleep(min(next_send - now, 0.005))
                continue

            ser.write(payload)
            ser.flush()
            sent_bytes += len(payload)
            next_send += args.interval

            elapsed = max(time.monotonic() - start_time, 0.001)
            current_kbps = sent_bytes * 8 / elapsed / 1000
            print(f"\rTX: {current_kbps:8.2f} kb/s | Bytes: {sent_bytes:10d}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nTX test stopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()