import os
import sys
import time
import subprocess
import argparse

# Add noCamSim to path to load auto_discover
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "noCamSim"))

try:
    from auto_discover import get_role_port
except ImportError:
    get_role_port = lambda role: None

def main():
    parser = argparse.ArgumentParser(description="Multi-Process Reed-Solomon Test Runner")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (default: 921600)")
    parser.add_argument("--interval", type=float, default=0.02, help="Transmit interval in seconds (default: 0.02)")
    parser.add_argument("--count", type=int, default=2000, help="Total packets to send (default: 2000)")
    parser.add_argument("--raw", action="store_true", help="Use raw UART mode (disable CC1310 DSSS MAC parsing)")
    args = parser.parse_args()

    tx_port = get_role_port("TX")
    rx_port = get_role_port("RX")

    if not tx_port or not rx_port:
        print("Error: Could not dynamically discover both TX and RX ports.")
        print("Please check board connections or configure noCamSim/board_config.json.")
        sys.exit(1)

    print("=======================================")
    print("Starting Multi-Process RS Test Suite")
    print("=======================================")
    print(f"TX Port: {tx_port}")
    print(f"RX Port: {rx_port}")
    print(f"Baud:    {args.baud}")
    print("=======================================\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    rx_script = os.path.join(current_dir, "rs_rx_test.py")
    tx_script = os.path.join(current_dir, "rs_tx_test.py")

    # Start Receiver
    rx_cmd = [sys.executable, "-u", rx_script, "--port", rx_port, "--baud", str(args.baud)]
    if args.raw:
        rx_cmd.append("--raw")
    
    print("[1/2] Launching RS Analyzer Dashboard...")
    rx_proc = subprocess.Popen(rx_cmd)

    # Wait 3 seconds for RX board to fully boot and sync RF
    time.sleep(3.0)

    # Start Transmitter
    tx_cmd = [
        sys.executable, "-u", tx_script, 
        "--port", tx_port, 
        "--baud", str(args.baud), 
        "--interval", str(args.interval),
        "--count", str(args.count)
    ]
    print("\n[2/2] Launching RS Packet Generator...")
    tx_proc = subprocess.Popen(tx_cmd)

    print("\nTest running. Press Ctrl+C in this terminal to terminate both scripts early.")

    try:
        # Wait for processes
        tx_proc.wait()
        # Give RX a second to gather any final trailing bytes
        time.sleep(1.0)
        rx_proc.terminate()
        rx_proc.wait(timeout=2)
    except KeyboardInterrupt:
        print("\n\nTerminating test processes...")
        try:
            tx_proc.terminate()
            rx_proc.terminate()
            tx_proc.wait(timeout=2)
            rx_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            tx_proc.kill()
            rx_proc.kill()
        print("Test suite shutdown complete.")

if __name__ == "__main__":
    main()
