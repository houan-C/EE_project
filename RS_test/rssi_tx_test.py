import argparse
import sys
import time

import serial


PAYLOAD_SIZE = 200


def main():
    parser = argparse.ArgumentParser(description="Simple TX throughput test")
    parser.add_argument("--port", required=True, help="TX board serial port")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--interval", type=float, default=0.02)
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as error:
        print(f"Error opening serial port: {error}")
        sys.exit(1)

    payload = bytes(i % 256 for i in range(PAYLOAD_SIZE))
    sent_bytes = 0
    start_time = time.monotonic()
    next_send = start_time

    try:
        print(f"Sending {PAYLOAD_SIZE}-byte payloads on {args.port}")
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