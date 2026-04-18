import os
import sys
import time
import subprocess

def main():
    # Make sure we use the correct paths regardless of where start.py is executed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tx_script = os.path.join(current_dir, "sim_tx.py")
    rx_script = os.path.join(current_dir, "sim_rx.py")
    
    print("=======================================")
    print("Starting Drone Video Transmission Sim")
    print("=======================================\n")
    
    print("[1/2] Launching Receiver (sim_rx.py)...")
    rx_process = subprocess.Popen([sys.executable, rx_script])
    
    # Give the receiver a second to initialize its queues and TensorRT models 
    # before we start blasting data from the transmitter
    time.sleep(3.0)
    
    print("\n[2/2] Launching Transmitter (sim_tx.py)...")
    tx_process = subprocess.Popen([sys.executable, tx_script])
    
    print("\nProcesses running. Press Ctrl+C in this terminal to shut down both.")
    
    try:
        # Wait indefinitely for both processes
        rx_process.wait()
        tx_process.wait()
    except KeyboardInterrupt:
        print("\n\nCaught KeyboardInterrupt! Shutting down Simulator scripts...")
        try:
            tx_process.terminate()
            rx_process.terminate()
            tx_process.wait(timeout=2)
            rx_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            tx_process.kill()
            rx_process.kill()
        print("Simulator shutdown complete.")

if __name__ == "__main__":
    main()
