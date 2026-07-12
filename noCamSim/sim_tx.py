import cv2
import time
import socket
import numpy as np
from PIL import Image
import io
import sys
import os
import csv

try:
    import pillow_avif
except ImportError:
    print("Error: pillow_avif module not found. Please install it using: pip install pillow-avif-plugin")

import serial
from auto_discover import get_role_port

try:
    from ultralytics import YOLO
    print("Loading YOLOv10 for pedestrian detection...")
    yolo_model = YOLO("yolov10n.pt")  # Requires latest ultralytics
except ImportError:
    yolo_model = None
    print("Warning: ultralytics package not found. YOLOv10 detection will be entirely disabled.")

DRAW_BBOXES = False # Let user decide if human bounding boxes should be drawn on TX side

com_port = "COM6"
# com_port = get_role_port("TX")

if com_port is None:
    print("Error: No available COM port found. Please check your connection.")
    sys.exit(1)
else:
    print(f"Found COM port: {com_port}")

baud_rate = 921600
timeout = 1

try:
    ser = serial.Serial(com_port, baud_rate, timeout=timeout)
except Exception as e:
    print(f"Error opening serial port {com_port}: {e}")
    sys.exit(1)

fight_index = 5

# Open the drone video
video_path = r"g:/code/EE_project/filghtRecord/" + str(fight_index) + ".mp4"
flighRecord_path = r"g:/code/EE_project/filghtRecord/" + str(fight_index) + ".csv"
if not os.path.exists(video_path):
    print(f"Video not found: {video_path}")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Assume the base webcam resolution normally used in Tx_AVIF_v3 
base_w, base_h = 640, 480
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_duration = total_frames / fps if fps > 0 else 0

print(f"Video Resolution: {orig_w}x{orig_h} @ {fps}fps")
print(f"Simulating base camera resolution of: {base_w}x{base_h}")

flight_data = []
if os.path.exists(flighRecord_path):
    print("Loading flight record data...")
    with open(flighRecord_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if "RC.aileron" in line:
                start_idx = i
                break
        
        reader = csv.DictReader(lines[start_idx:])
        for row in reader:
            try:
                t_val = float(row["OSD.flyTime [s]"])
                a = int(row["RC.aileron"])
                e = int(row["RC.elevator"])
                th = int(row["RC.throttle"])
                r = int(row["RC.rudder"])
                flight_data.append({"t": t_val, "a": a, "e": e, "th": th, "r": r})
            except Exception:
                pass
    
    if flight_data:
        start_t = flight_data[0]["t"]
        for d in flight_data:
            d["t"] -= start_t
        print(f"Loaded {len(flight_data)} RC records synced to video.")

chunk_size = 812

print("\n--- Controls ---")
print("Press 'q' to quit")
print("----------------\n")

start_real_time = time.time()

while True:
    loop_start = time.time()
    
    # Calculate what time in the video we should be at based on real world time
    elapsed_real = loop_start - start_real_time
    
    # Loop video if it ends
    if total_duration > 0 and elapsed_real > total_duration:
        start_real_time = time.time()
        elapsed_real = 0

    # Seek to this exact time
    cap.set(cv2.CAP_PROP_POS_MSEC, int(elapsed_real * 1000))
    ret, frame = cap.read()
    
    if not ret:
        start_real_time = time.time()
        cap.set(cv2.CAP_PROP_POS_MSEC, 0)
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from video.")
            break

    # Pedestrian YOLO Detection
    has_target = False
    if yolo_model is not None:
        results = yolo_model(frame, classes=[0], verbose=False) # class 0 is 'person'
        for r in results:
            if len(r.boxes) > 0:
                has_target = True
                if DRAW_BBOXES:
                    frame = r.plot() # Draw detection boxes directly onto the frame

    # Settings based on stick movement to toggle mode
    isMoving = True
    if flight_data:
        closest = flight_data[-1]
        for d in flight_data:
            if d["t"] >= elapsed_real:
                closest = d
                break
        # Deadzone checking
        if 1023 <= closest["a"] <= 1025 and 1023 <= closest["e"] <= 1025 and \
           1023 <= closest["th"] <= 1025 and 1023 <= closest["r"] <= 1025:
            isMoving = False

    if isMoving:
        ratio, quality = 0.64, 29
        flag_marker = b'MODE_LQP' if has_target else b'MODE_LQN'
    else:
        # Highest possible quality mode when still!
        ratio, quality = 1.0, 80
        flag_marker = b'MODE_HQP' if has_target else b'MODE_HQN'
    # Calculate new resolution based on ratio on base dimensions
    new_w = int(base_w * ratio)
    new_h = int(base_h * ratio)

    # Resize the frame (Dynamic Resolution)
    frame_resized = cv2.resize(frame, (new_w, new_h))
    
    # Encode with Pillow (AVIF)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format="AVIF", quality=quality, speed=10) 
    frame_bytes = flag_marker + buffer.getvalue()

    raw_frame_size = frame_resized.nbytes
    compressed_frame_size = len(frame_bytes)
    compression_ratio = compressed_frame_size / raw_frame_size * 100

    # Transmit data in chunks through serial connection
    try:
        for i in range(0, len(frame_bytes), chunk_size):
            chunk = frame_bytes[i:i + chunk_size]
            ser.write(chunk)
            time.sleep(0.03)  # 120K DSSS UART wait from Tx_AVIF_v3
    except Exception as e:
        print(f"Serial disconnected: {e}")
        break

    end_time = time.time()
    duration = end_time - loop_start
    datasize = len(frame_bytes)
    datarate = datasize / duration / 1000
    
    print(f"Res: {new_w}x{new_h} ({ratio}) | Quality: {quality}")
    print(f"Compressed size: {compressed_frame_size} bytes | Ratio: {compression_ratio:.2f}%")
    print(f"Duration: {duration:.3f}s | Datarate: {datarate:.2f} kB/sec\n")

    # Decode the compressed frame for display (skip 8 bytes of mode marker)
    dec_img = Image.open(io.BytesIO(frame_bytes[8:]))
    dec_bgr = cv2.cvtColor(np.array(dec_img), cv2.COLOR_RGB2BGR)

    # Resize both for side-by-side preview to avoid exceeding screen width
    preview_w, preview_h = 640, 360
    
    orig_display = cv2.resize(frame, (preview_w, preview_h))
    dec_display = cv2.resize(dec_bgr, (preview_w, preview_h))

    cv2.putText(orig_display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(orig_display, "Keys: Q to quit", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.putText(dec_display, f"Compressed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(dec_display, f"Res: {new_w}x{new_h} (x{ratio})", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(dec_display, f"Quality: {quality}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    combined_frame = np.hstack((orig_display, dec_display))
    cv2.imshow('Sim_TX_Stream', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
ser.close()
cv2.destroyAllWindows()
