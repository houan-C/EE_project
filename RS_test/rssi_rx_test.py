import argparse
import sys
import time

import serial


def main():
    parser = argparse.ArgumentParser(description="Simple RX throughput and RSSI test")
    parser.add_argument("--port", required=True, help="RX board serial port")
    parser.add_argument("--baud", type=int, default=921600)
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as error:
        print(f"Error opening serial port: {error}")
        sys.exit(1)

    frame_buffer = bytearray()
    received_bytes = 0
    window_bytes = 0
    window_start = time.monotonic()
    latest_rssi = None

    try:
        ser.reset_input_buffer()
        print(f"Listening for RX frames on {args.port}")
        print("Press Ctrl+C to stop.\n")

        while True:
            data = ser.read(ser.in_waiting or 1)
            if data:
                frame_buffer.extend(data)

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

                latest_rssi = frame_buffer[payload_len + 1] - 256
                received_bytes += payload_len
                window_bytes += payload_len
                del frame_buffer[:frame_len]

            now = time.monotonic()
            elapsed = max(now - window_start, 0.001)
            current_kbps = window_bytes * 8 / elapsed / 1000
            rssi_text = f"{latest_rssi:4d} dBm" if latest_rssi is not None else "waiting"
            print(f"\rRX: {current_kbps:8.2f} kb/s | RSSI: {rssi_text} | Bytes: {received_bytes:10d}", end="", flush=True)

            if elapsed >= 1.0:
                window_start = now
                window_bytes = 0
    except KeyboardInterrupt:
        print("\nRX test stopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()