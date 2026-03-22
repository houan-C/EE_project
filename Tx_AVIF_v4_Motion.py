import cv2
import serial
import serial.tools.list_ports
import time
import numpy as np
from PIL import Image
import io
import sys
import struct
import threading

# Try to import pillow_avif to register the AVIF plugin. 
try:
    import pillow_avif
except ImportError:
    print("Error: pillow_avif module not found. Please install it using: pip install pillow-avif-plugin")

def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        if "XDS110" in p.description: return p.device
    for p in ports:
        if "USB" in p.description or "UART" in p.description: return p.device
    return ports[0].device

com_port = find_serial_port()

if com_port is None:
    print("Error: No available COM port found.")
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

cap = cv2.VideoCapture(0)

# LOCK AUTO EXPOSURE AND WHITE BALANCE
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Original Camera Resolution: {orig_w}x{orig_h}")

settings = {
    4: [0.64, 29],
    3: [0.59, 25],
    2: [0.48, 22],
    1: [0.38, 23],
    0: [0.34, 22]
}

current_level = 4
chunk_size = 812

print("\n--- Controls ---")
print("Press '+' to increase quality/resolution (max 4)")
print("Press '-' to decrease quality/resolution (min 0)")
print("Press 'f' to force send a full frame")
print("Press 'q' to quit")
print("----------------\n")

force_full_frame = True 
frames_since_full = 0
prev_gray = None

accumulated_dx = 0.0
accumulated_dy = 0.0

GRID_ROWS = 6
GRID_COLS = 8

# ====== Async Zero-Latency Serial Writer Thread ======
# 由於 queue 會排隊導致超過 0.5s 以上的舊圖被送出產生嚴重延遲，我們改用單一共用語句直接覆蓋最新畫面！
latest_payload_lock = threading.Lock()
latest_payload = None
running = True

def serial_writer_thread(serial_port, c_size):
    global latest_payload
    while running:
        payload = None
        # 直接抓走目前卡位的最新完整畫面 (0ms 延遲出發！)
        with latest_payload_lock:
            if latest_payload is not None:
                payload = latest_payload
                latest_payload = None
        
        if payload is not None:
            # 開始漫長的無線電發送，這段期間相機依舊可以無阻礙的捕捉 30 FPS 畫面並覆蓋
            for i in range(0, len(payload), c_size):
                chunk = payload[i:i + c_size]
                serial_port.write(chunk)
                time.sleep(0.03)  # 120K DSSS waiting
        else:
            time.sleep(0.005)

writer_thread = threading.Thread(target=serial_writer_thread, args=(ser, chunk_size), daemon=True)
writer_thread.start()
# ===================================================

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    ratio, quality = settings[current_level]
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    frame_resized = cv2.resize(frame, (new_w, new_h))
    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        prev_gray = curr_gray.copy()
        force_full_frame = True

    # ---- 1. Global Motion Compensation (GMC) ----
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    
    dx_local = 0.0
    dy_local = 0.0
    
    if prev_pts is not None and len(prev_pts) > 4:
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is not None:
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            if len(good_prev) > 4:
                matrix, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
                if matrix is not None:
                    dx, dy = matrix[0, 2], matrix[1, 2]
                    # 濾除微小感光元件雜訊
                    if abs(dx) >= 1.5 or abs(dy) >= 1.5:
                        dx_local = float(round(dx))
                        dy_local = float(round(dy))

    local_matrix = np.array([[1.0, 0.0, dx_local], [0.0, 1.0, dy_local]], dtype=np.float32)
    aligned_prev = cv2.warpAffine(prev_gray, local_matrix, (new_w, new_h))
    
    accumulated_dx += dx_local
    accumulated_dy += dy_local

    diff = cv2.absdiff(curr_gray, aligned_prev)
    _, fg_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)
    
    # 找尋運動主體的外框 - 終結 RIFE 拉扯殘影的殺手鐧！
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    force_block_active = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    
    block_w = new_w // GRID_COLS
    block_h = new_h // GRID_ROWS

    if contours:
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                
                # 將同一個主體移動軌跡內的所有背景/前景框起來，要求網格要麼整塊人同時代謝更新，要麼都別更新
                # 這樣 RIFE 補幀軟體才不會發現半張臉卡住半張臉在動，導致時空錯亂產生拉扯感。
                px1 = max(0, cx - 20)
                py1 = max(0, cy - 20)
                px2 = min(new_w, cx + cw + 20)
                py2 = min(new_h, cy + ch + 20)

                start_c = min(GRID_COLS - 1, max(0, px1 // block_w))
                end_c = min(GRID_COLS - 1, max(0, px2 // block_w))
                start_r = min(GRID_ROWS - 1, max(0, py1 // block_h))
                end_r = min(GRID_ROWS - 1, max(0, py2 // block_h))
                
                for r in range(start_r, end_r + 1):
                    for c in range(start_c, end_c + 1):
                        force_block_active[r, c] = True

    prev_gray = curr_gray.copy()

    # ---- 2. Macroblock Grid 判斷與打包 ----
    send_full_frame = force_full_frame
    frames_since_full += 1
    if frames_since_full >= 40:
        send_full_frame = True
    
    bitmask = 0
    active_blocks = []
    
    if not send_full_frame:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                # 嚴格固定 block_w 與 block_h 避免縮放尺寸擠壓破壞畫質，也不再需要 cv2.resize 修補！
                y1 = r * block_h
                y2 = y1 + block_h
                x1 = c * block_w
                x2 = x1 + block_w
                
                cell_mask = fg_mask[y1:y2, x1:x2]
                active_pixels = cv2.countNonZero(cell_mask)
                total_pixels = cell_mask.size
                
                # 如果該網格有超過面積變動，或者它是被「統一強迫更新(force_block_active)」的同車夥伴，全部傳送！
                if active_pixels > total_pixels * 0.02 or force_block_active[r, c]:
                    bitmask |= (1 << (r * GRID_COLS + c))
                    block_img = frame_resized[y1:y2, x1:x2]
                    active_blocks.append(block_img)
                    
        if bitmask == 0:
            time.sleep(0.01)
            display_frame = frame_resized.copy()
            cv2.putText(display_frame, "NO MOTION - SKIPPED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('TX_AVIF_Motion_Control', display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('f'):
                force_full_frame = True
            continue
            
        if len(active_blocks) > (GRID_ROWS * GRID_COLS) * 0.6:
            send_full_frame = True

    if send_full_frame:
        target_crop = frame_resized
        bitmask = 0
    else:
        target_crop = np.vstack(active_blocks)

    # ---- 3. 編碼為 AVIF ----
    frame_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="AVIF", quality=quality, speed=10) 
    frame_bytes = buffer.getvalue()
    compressed_frame_size = len(frame_bytes)

    # ---- 4. 新版通訊協定 Header ----
    pkt_type = 0 if send_full_frame else 1
    matrix_flat = [1.0, 0.0, accumulated_dx, 0.0, 1.0, accumulated_dy]
    header_format = '<4sBQHHIffffff'
    header = struct.pack(header_format, b'AVIF', pkt_type, bitmask, new_w, new_h, compressed_frame_size, *matrix_flat)
    full_payload = header + frame_bytes

    # ---- 5. 終極零延遲 LIFO 狀態同步寫入 ----
    tx_busy = False
    with latest_payload_lock:
        if latest_payload is None:
            # 傳送器剛好發完。此處是最安全、且確定必發出去的時刻，把帳本歸零！
            latest_payload = full_payload
            accumulated_dx = 0.0
            accumulated_dy = 0.0
            if send_full_frame:
                force_full_frame = False
                frames_since_full = 0
        else:
            # 傳送器還在忙！將最新的 30FPS 高效能運算結果『覆蓋』舊封包，徹底消除 Queue 過渡延遲！
            # 注意：此時不清空 accumulated_dx/dy，讓位移帳單繼續滾到這包含有累積偏移量的最新 Payload 給 RX 當下一包！
            latest_payload = full_payload
            tx_busy = True

    end_time = time.time()
    duration = end_time - start_time
    datarate = compressed_frame_size / max(duration, 0.001) / 1000

    # ---- 6. 開發者 UI & Debug Info ----
    print(f"Level: {current_level} | Base Res: {new_w}x{new_h} | Type: {'FULL' if pkt_type == 0 else 'GRID'}")
    if pkt_type == 1:
        print(f"Active Blocks: {len(active_blocks)}/48 | Bitmask: {hex(bitmask)}")
        print(f"Packed Grid Size: {target_crop.shape[1]}x{target_crop.shape[0]}")
    
    if tx_busy:
        print(f"Total Sent Payload: {len(full_payload)} bytes | TX STILL BUSY: Overwriting Frame To Zero Latency!")
    else:
        print(f"Total Sent Payload: {len(full_payload)} bytes | TX SENT!")
        
    print(f"Capture+Encode Duration: {duration:.3f}s | Equivalent Datarate: {datarate:.2f} kB/sec\n")

    display_frame = frame_resized.copy()
    if pkt_type == 1:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if (bitmask & (1 << (r * GRID_COLS + c))) != 0:
                    y1 = r * block_h
                    x1 = c * block_w
                    y2 = y1 + block_h
                    x2 = x1 + block_w
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(display_frame, f"GRID BLOCKS: {len(active_blocks)}/48", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "FULL FRAME", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    cv2.putText(display_frame, f"Quality: {current_level}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('TX_AVIF_Motion_Control', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('+') or key == ord('='):
        if current_level < 4:
            current_level += 1
            force_full_frame = True
    elif key == ord('-') or key == ord('_'):
        if current_level > 0:
            current_level -= 1
            force_full_frame = True
    elif key == ord('f'):
        force_full_frame = True

running = False
writer_thread.join(timeout=1.0)
cap.release()
ser.close()
cv2.destroyAllWindows()
