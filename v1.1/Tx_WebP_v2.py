## Continuous webcam  - working TX
## Send compressed video to CC1310 with robust frame markers in chunks

import cv2
import serial
import serial.tools.list_ports
import time
import numpy as np

def find_com_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No COM ports found!")
        return None
    
    # Try to find a port that looks like a CC1310 / XDS110
    for p in ports:
        if "XDS110" in p.description or "Texas Instruments" in p.description:
            print(f"Automatically selected port: {p.device} ({p.description})")
            return p.device
    
    # Fallback to the first available port if no XDS110 is found
    print(f"No specific device found, using first available port: {ports[0].device} ({ports[0].description})")
    return ports[0].device

# Set up the serial connection
com_port = find_com_port()

if com_port is None:
    print("Error: Could not find any COM port. Please check your connection.")
    exit()

baud_rate = 921600  # Ensure this matches the sending baud rate 
timeout = 1  # Set a timeout for reading

# Set up the serial connection
try:
    ser = serial.Serial(com_port, baud_rate, timeout=timeout)
    print(f"Successfully opened {com_port}")
except Exception as e:
    print(f"Error opening serial port {com_port}: {e}")
    exit()

cap = cv2.VideoCapture(0)

# start_marker = b'\xAA\xBB\xCC' * 3  # Three consecutive start markers
# stop_marker = b'\xDD\xEE\xFF' * 3  # Three consecutive stop markers

chunk_size = 812  # Define the size of each chunk  , was 812

try:
    while True:
        start = time.time()
        ret, frame = cap.read()
        #print(f"The frame size is {len(frame)}")
        # frame = cv2.resize(frame, (160, 120))
        if not ret:
            print("Failed to capture frame")
            break

        # Compress the frame
        frame = cv2.resize(frame, (160, 120))
        ret, compressed_frame = cv2.imencode('.webp', frame,
                                             [int(cv2.IMWRITE_WEBP_QUALITY), 20])  # default 50 (higher = high quality )
        frame_bytes = compressed_frame.tobytes()

        # with open('frame.webp', 'wb') as f:            # 看webp
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
except KeyboardInterrupt:
    print("Manually stopped by user")

finally:
    # Clean up
    cap.release()
    ser.close()
    cv2.destroyAllWindows()
    print("Cleaned up resources.")
