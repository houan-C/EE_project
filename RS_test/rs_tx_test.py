import time
import serial
import serial.tools.list_ports
import struct
import sys
import argparse
from reedsolo import RSCodec

def find_serial_port():
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
    parser = argparse.ArgumentParser(description="Reed-Solomon Hardware Test Transmitter")
    parser.add_argument("--port", type=str, default=None, help="COM port (default: auto-discover)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--interval", type=float, default=0.02, help="Transmit interval in seconds (default: 0.02)")
    parser.add_argument("--count", type=int, default=2000, help="Total packets to send, 0 for infinite (default: 2000)")
    args = parser.parse_args()

    com_port = args.port if args.port else find_serial_port()
    if not com_port:
        print("Error: No serial port found.")
        sys.exit(1)

    print(f"Opening serial port: {com_port} at {args.baud} baud...")
    try:
        ser = serial.Serial(com_port, args.baud, timeout=1)
    except Exception as e:
        print(f"Error: Could not open serial port {com_port}: {e}")
        sys.exit(1)

    # Perform hardware reset on CC1310 LaunchPad
    print("Performing CC1310 hardware reset...")
    ser.dtr = True
    ser.rts = True
    time.sleep(0.2)
    ser.dtr = False
    ser.rts = False

    print("Waiting 5 seconds for CC1310 TX board to boot and initialize RF...")
    time.sleep(5.0)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("Initializing Reed-Solomon Encoder (nsym=32)...")
    rs = RSCodec(32)

    # For 200-byte total block size with nsym=32, data payload must be 168 bytes
    seq_no = 0
    total_packets = args.count

    print(f"\n--- Starting RS Transmission ---")
    print(f"Interval: {args.interval}s | Target Count: {'Infinite' if total_packets == 0 else total_packets}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            if total_packets > 0 and seq_no >= total_packets:
                print(f"Finished sending {total_packets} packets.")
                break

            # Message structure: [SeqNo (4 bytes) | Payload (164 bytes)] = 168 bytes
            payload = bytes(i % 256 for i in range(164))
            message = struct.pack(">I", seq_no) + payload

            # RS Encode to exactly 200 bytes (168 bytes data + 32 bytes parity)
            rs_block = rs.encode(message)

            # Transmit exactly 200 bytes to match CC1310 PAYLOAD_LENGTH
            ser.write(rs_block)
            ser.flush()

            if seq_no % 100 == 0:
                print(f"Sent packet #{seq_no}...")

            seq_no += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nTransmission stopped by user. Sent {seq_no} packets.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()
