## Continuous webcam  - working TX
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

# start_marker = b'\xAA\xBB\xCC' * 3  # Three consecutive start markers
# stop_marker = b'\xDD\xEE\xFF' * 3  # Three consecutive stop markers

chunk_size = 812  # Define the size of each chunk  , was 812

while True:
    start = time.time()
    ret, frame = cap.read()
    #print(f"The frame size is {len(frame)}")
    # frame = cv2.resize(frame, (160, 120))
    if not ret:
        print("Failed to capture frame")
        break

    # Compress the frame
    frame = cv2.resize(frame, (480, 360))
    
    # Encode with Pillow (AVIF) because OpenCV default build lacks AVIF encoder
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    buffer = io.BytesIO()
    # quality=10: matches previous setting
    # speed=10: fastest encoding speed (critical for webcam). Valid range 0-10.
    pil_img.save(buffer, format="AVIF", quality=20, speed=10) 
    frame_bytes = buffer.getvalue()

    # with open('frame.avif', 'wb') as f:            # 看avif
    #     f.write(frame_bytes)

    # start = time.time()
    # if not ret:
    #    print("Failed to encode frame")
    #    break

    # Calculate and print compression information
    raw_frame_size = frame.nbytes  # unit bytes
    compressed_frame_size = len(frame_bytes)  # unit bytes
    compression_ratio = compressed_frame_size / raw_frame_size * 100  # unit %

    print(f" Captured frame resolution: {frame.shape[1]}x{frame.shape[0]}, channels: {frame.shape[2]}")
    print(f" Raw frame size: {raw_frame_size} bytes")
    print(f" Compressed frame size: {compressed_frame_size} bytes")
    print(f" Compression ratio: {compression_ratio:.2f}%\n")

    # all_bytes = start_marker + frame_bytes + stop_marker

    # Transmit data in chunks
    for i in range(0, len(frame_bytes), chunk_size):
        chunk = frame_bytes[i:i + chunk_size]
        ser.write(chunk)
        # print(chunk)
        time.sleep(0.03)  # 120K DSSS
        #time.sleep(0.02) #750K
    end = time.time()
    duration = end - start
    datasize = len(frame_bytes)
    datarate = datasize / duration / 1000
    print(f'Done! The duration for sending one frame: {(duration)} seconds')

    print(f"Sent frame of size {len(frame_bytes)} bytes, with datarate of {(datarate)} kB/sec.\n")

    # Display the compressed frame for debugging
    # cv2.imshow('Camera Feed', frame)

    # Press 'q' to exit the display loop
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# Clean up
cap.release()
ser.close()
cv2.destroyAllWindows()
