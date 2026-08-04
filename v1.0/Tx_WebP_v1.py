## Continuous webcam  - working TX
## Send compressed video to CC1310 with robust frame markers in chunks

import cv2
import serial
import time
import numpy as np

# Set up the serial connection
#com_port = "/dev/ttyACM0"  # Raspberry Pi 上常見的 USB-to-Serial port，必要時改為 /dev/ttyACM1
com_port = "COM3"
baud_rate = 921600  # Ensure this matches the sending baud rate 
timeout = 1  # Set a timeout for reading

# Set up the serial connection
ser = serial.Serial(com_port, baud_rate, timeout=timeout)

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

# Clean up
cap.release()
ser.close()
cv2.destroyAllWindows()
