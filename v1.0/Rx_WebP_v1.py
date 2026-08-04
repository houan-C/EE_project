# receiver_upgrade.py (with Partial Frame Buffering + Timeout Frame Closing)
import cv2
import serial
import numpy as np
import time

# UART settings
com_port = "COM6"
baud_rate = 921600
chunk_size = 512
ser = serial.Serial(com_port, baud_rate, timeout=0.01)
right_count = 0 #計算
wrong_count = 0 #計算

# Frame markers
start_marker = b'\xAA\xBB\xCC' * 3
stop_marker = b'\xDD\xEE\xFF' * 3

# Buffers
buffer = bytearray()
chunk = bytearray()

# RSSI
RSSI_max = -100
RSSI_min = 100
RSSI_count = 0

# FPS
prev_time = time.time()
fps = 0

# Timeout control
frame_start_time = None
frame_timeout_sec = 5.0  # Timeout if no stop_marker after 1.0 sec

print(f" Listening on {com_port}...")

try:
    while True:
        read_data = ser.read(chunk_size)
        if not read_data:
            continue  # No data read

        chunk.extend(read_data)

        # Try to extract packets from chunk
        while len(chunk) >= 3:
            payloadLen = chunk[0]
            if payloadLen > 255:
                print(f"⚠️ Warning: payloadLen too big ({payloadLen}), drop 1 byte")
                chunk = chunk[1:]
                continue

            if len(chunk) < payloadLen + 3:
                break  # Not enough for a complete packet

            payload = chunk[1 : payloadLen + 1]
            rssi = chunk[payloadLen + 1] - 256
            status = bin(chunk[payloadLen + 2])

            RSSI_count += 1
            RSSI_max = max(RSSI_max, rssi)
            RSSI_min = min(RSSI_min, rssi)

            # print(f" payloadLen={payloadLen}, rssi={rssi}, status={status}")

            buffer.extend(payload)
            chunk = chunk[payloadLen + 3:]

            # Start timing when we see the first payload (for timeout)
            if frame_start_time is None:
                frame_start_time = time.time()

        # Try to find a complete frame
        while start_marker in buffer and stop_marker in buffer:
            start_idx = buffer.index(start_marker) + len(start_marker)
            stop_idx = buffer.index(stop_marker)

            if stop_idx <= start_idx:
                print(f"❌ Marker order wrong (stop_idx={stop_idx} <= start_idx={start_idx}), drop")
                buffer = buffer[stop_idx + len(stop_marker):]
                frame_start_time = None
                wrong_count = wrong_count + 1 #計算
                continue

            frame_data = buffer[start_idx:stop_idx]

            if len(frame_data) == 0:
                print("❌ Frame_data is empty, skip")
                buffer = buffer[stop_idx + len(stop_marker):]
                frame_start_time = None
                wrong_count = wrong_count + 1 #計算
                continue

            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"❌ Frame decode failed, frame_data size={len(frame_data)}")
                buffer = buffer[stop_idx + len(stop_marker):]
                frame_start_time = None
                wrong_count = wrong_count + 1 #計算
                continue

            # FPS calculation
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            text_rssi = f"RSSI: {rssi} dBm"
            text_fps = f"FPS: {fps:.2f}"

            cv2.putText(frame, text_rssi, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text_fps, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow("Received Video Feed", frame)

            print(f" payloadLen={payloadLen}, rssi={rssi}, status={status}")
            print(f"Decode Frame Successfully")
            right_count = right_count + 1 #計算

            buffer = buffer[stop_idx + len(stop_marker):]  # Remove processed frame
            frame_start_time = None  # Reset timeout after successful frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt

        # Check for timeout
        if frame_start_time is not None and (time.time() - frame_start_time) > frame_timeout_sec:
            print(f"⏰ Frame assembly timeout (> {frame_timeout_sec} sec), discard buffer")
            buffer.clear()
            frame_start_time = None

except KeyboardInterrupt:
    print(" End Receive")
    print(f"right count:{right_count}")
    print(f"wrong count:{wrong_count}")
    print(f"正確率:{right_count/(right_count + wrong_count)}")
finally:
    ser.close()
    cv2.destroyAllWindows()
