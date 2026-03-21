## Continuous webcam  - working TX with dynamic settings
## Send compressed video to CC1310 with robust frame markers in chunks

import cv2
import serial
import serial.tools.list_ports
import time
import numpy as np
from PIL import Image
import io
import sys

# Try to import pillow_avif to register the AVIF plugin. 
# User needs to install: pip install pillow pillow-avif-plugin
try:
    import pillow_avif
except ImportError:
    print("Error: pillow_avif module not found. Please install it using: pip install pillow-avif-plugin")

def find_serial_port():
    """Automatically find the available COM port."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    
    # Priority 1: Look for XDS110 (common for CC1310 LaunchPad)
    for p in ports:
        if "XDS110" in p.description:
            return p.device
            
    # Priority 2: Look for any USB Serial port
    for p in ports:
        if "USB" in p.description or "UART" in p.description:
            return p.device
            
    # Fallback: Return the first available port
    return ports[0].device

# Set up the serial connection
com_port = find_serial_port()

if com_port is None:
    print("Error: No available COM port found. Please check your connection.")
    sys.exit(1)
else:
    print(f"Found COM port: {com_port}")

baud_rate = 921600  # Ensure this matches the sending baud rate 
timeout = 1  # Set a timeout for reading

# Set up the serial connection
try:
    ser = serial.Serial(com_port, baud_rate, timeout=timeout)
except Exception as e:
    print(f"Error opening serial port {com_port}: {e}")
    sys.exit(1)

cap = cv2.VideoCapture(0)

# Get original camera resolution
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original Camera Resolution: {orig_w}x{orig_h}")

# Settings mapping: level -> [resolution_ratio, quality]
# 4: [0.64, 29], 3: [0.59, 25], 2: [0.48, 22], 1: [0.38, 23], 0: [0.34, 22]
settings = {
    4: [0.64, 29],
    3: [0.59, 25],
    2: [0.48, 22],
    1: [0.38, 23],
    0: [0.34, 22]
}

current_level = 4
chunk_size = 812  # Define the size of each chunk

print("\n--- Controls ---")
print("Press '+' to increase quality/resolution (max 4)")
print("Press '-' to decrease quality/resolution (min 0)")
print("Press 'q' to quit")
print("----------------\n")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Get current settings based on level
    ratio, quality = settings[current_level]
    
    # Calculate new resolution based on ratio
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    # Resize the frame (Dynamic Resolution)
    frame_resized = cv2.resize(frame, (new_w, new_h))
    
    # Encode with Pillow (AVIF)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    buffer = io.BytesIO()
    # Save with Dynamic Quality
    pil_img.save(buffer, format="AVIF", quality=quality, speed=10) 
    frame_bytes = buffer.getvalue()

    # Calculate and print compression information
    raw_frame_size = frame_resized.nbytes  # unit bytes
    compressed_frame_size = len(frame_bytes)  # unit bytes
    compression_ratio = compressed_frame_size / raw_frame_size * 100  # unit %

    # Transmit data in chunks
    for i in range(0, len(frame_bytes), chunk_size):
        chunk = frame_bytes[i:i + chunk_size]
        ser.write(chunk)
        time.sleep(0.03)  # 120K DSSS
        
    end_time = time.time()
    duration = end_time - start_time
    datasize = len(frame_bytes)
    datarate = datasize / duration / 1000
    
    # Status prints
    print(f"Level: {current_level} | Res: {new_w}x{new_h} ({ratio}) | Quality: {quality}")
    print(f"Compressed size: {compressed_frame_size} bytes | Ratio: {compression_ratio:.2f}%")
    print(f"Duration: {duration:.3f}s | Datarate: {datarate:.2f} kB/sec\n")

    # Display the frame with overlaid info (UI for the user)
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Level: {current_level}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Res: {new_w}x{new_h} (x{ratio})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Quality: {quality}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(display_frame, "Keys: + / - to adjust, Q to quit", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.imshow('TX_AVIF_Dynamic_Control', display_frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: # 'q' or ESC
        break
    elif key == ord('+') or key == ord('='):
        if current_level < 4:
            current_level += 1
            print(f">>> Switched to Level {current_level}")
    elif key == ord('-') or key == ord('_'):
        if current_level > 0:
            current_level -= 1
            print(f">>> Switched to Level {current_level}")

# Clean up
cap.release()
ser.close()
cv2.destroyAllWindows()
