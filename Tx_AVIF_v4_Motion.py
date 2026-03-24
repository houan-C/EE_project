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
import queue

# Try to import pillow_avif to register the AVIF plugin. 
try:
    import pillow_avif
except ImportError:
    print("Error: pillow_avif module not found. Please install it using: pip install pillow-avif-plugin")

def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports: return None
    for p in ports:
        if "XDS110" in p.description: return p.device
    for p in ports:
        if "USB" in p.description or "UART" in p.description: return p.device
    return ports[0].device

com_port = find_serial_port()
if com_port is None:
    print("Error: No available COM port found.")
    sys.exit(1)
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

# ====== Thread 1: Async Zero-Latency Camera Capture ======
latest_frame_lock = threading.Lock()
latest_frame = None
running = True

def camera_capture_thread(cap_obj):
    global latest_frame
    while running:
        ret, frm = cap_obj.read()
        if ret:
            with latest_frame_lock:
                latest_frame = frm
        else:
            time.sleep(0.01)

cap_thread = threading.Thread(target=camera_capture_thread, args=(cap,), daemon=True)
cap_thread.start()

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
print("Press '+' to increase quality (max 4)")
print("Press '-' to decrease quality (min 0)")
print("Press 'f' to force Full Frame")
print("Press 'q' to quit")
print("----------------\n")

# HEADER FORMAT: 17 bytes -> Magic(4) + PktType(1) + dx(2) + dy(2) + CropX(2) + CropY(2) + PayloadLen(4)
HEADER_FORMAT = '<4sBhhHHI'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

force_full_frame = True
frames_since_full = 0
prev_gray = None
rx_mock_bg = None  # 模擬 RX 端的背景，解決慢速移動與殘影

accumulated_dx = 0.0
accumulated_dy = 0.0

class PayloadState:
    def __init__(self, payload, dx, dy, crop_img, crop_x, crop_y, send_full):
        self.payload = payload
        self.dx = dx
        self.dy = dy
        self.crop_img = crop_img
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.send_full = send_full

latest_payload_lock = threading.Lock()
latest_payload_state = None
sent_queue = queue.Queue()

# ====== Thread 3: Zero-Latency Serial Writer ======
def serial_writer_thread(serial_port, c_size):
    global latest_payload_state
    while running:
        state = None
        with latest_payload_lock:
            if latest_payload_state is not None:
                state = latest_payload_state
                latest_payload_state = None

        if state is not None:
            # 確認要發送了，將其回傳給 Main Loop 同步 mock 畫布
            sent_queue.put(state)
            for i in range(0, len(state.payload), c_size):
                chunk = state.payload[i:i + c_size]
                serial_port.write(chunk)
                time.sleep(0.03)
        else:
            time.sleep(0.005)

writer_thread = threading.Thread(target=serial_writer_thread, args=(ser, chunk_size), daemon=True)
writer_thread.start()

# ====== Thread 2: Main Processing Loop ======
while True:
    start_time = time.time()
    
    with latest_frame_lock:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        frame = latest_frame.copy()

    ratio, quality = settings[current_level]
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    frame_resized = cv2.resize(frame, (new_w, new_h))
    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # ==== 處理由發送端完成的封包以同步模擬背景 ====
    while not sent_queue.empty():
        st = sent_queue.get()
        # 扣除已經順利送給 RX 的物理偏移量 (保留小數點殘差追蹤)
        accumulated_dx -= st.dx
        accumulated_dy -= st.dy
        
        if st.send_full:
            rx_mock_bg = st.crop_img.copy()
        else:
            # 模擬 RX 的移動背景操作
            M_sent = np.array([[1.0, 0.0, float(st.dx)], [0.0, 1.0, float(st.dy)]], dtype=np.float32)
            rx_mock_bg = cv2.warpAffine(rx_mock_bg, M_sent, (new_w, new_h))
            # 貼上 RX 即將貼上的補丁（包含羽化與防閃爍設計，確保 TX 追蹤數學精確吻合）
            cy, cx = st.crop_y, st.crop_x
            ch, cw = st.crop_img.shape[:2]

            target_roi = rx_mock_bg[cy:cy+ch, cx:cx+cw].astype(np.float32)
            patch_f = st.crop_img.astype(np.float32)
            
            # 灰階均值色彩匹配
            bg_mean = cv2.mean(target_roi)[0]
            patch_mean = cv2.mean(patch_f)[0]
            diff_c = np.clip(bg_mean - patch_mean, -8.0, 8.0)
            patch_f = np.clip(patch_f + diff_c, 0, 255)
            
            feather_px = 8
            alpha = np.ones((ch, cw), dtype=np.float32)
            for i in range(feather_px):
                val = i / feather_px
                if cy > 0: alpha[i, :] = np.minimum(alpha[i, :], val)
                if cy+ch < new_h: alpha[-(i+1), :] = np.minimum(alpha[-(i+1), :], val)
                if cx > 0: alpha[:, i] = np.minimum(alpha[:, i], val)
                if cx+cw < new_w: alpha[:, -(i+1)] = np.minimum(alpha[:, -(i+1)], val)
                
            blended = patch_f * alpha + target_roi * (1.0 - alpha)
            rx_mock_bg[cy:cy+ch, cx:cx+cw] = blended.astype(np.uint8)

    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        prev_gray = curr_gray.copy()
        rx_mock_bg = curr_gray.copy()
        force_full_frame = True

    # ---- 1. Global Motion Compensation (GMC) ----
    dx_local, dy_local = 0.0, 0.0
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    if prev_pts is not None and len(prev_pts) > 4:
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is not None:
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            if len(good_prev) > 4:
                matrix, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
                if matrix is not None:
                    dx, dy = matrix[0, 2], matrix[1, 2]
                    # 死區 (Deadzone) 加大到 1.2 像素，徹底過濾感光元件雜訊帶來的無效背景扭曲
                    if abs(dx) > 1.2 or abs(dy) > 1.2:
                        dx_local = dx
                        dy_local = dy

    accumulated_dx += dx_local
    accumulated_dy += dy_local
    prev_gray = curr_gray.copy()

    # 以 RX 端實際會被執行的「整數化偏移量」來做預測
    int_acc_dx = int(round(accumulated_dx))
    int_acc_dy = int(round(accumulated_dy))
    
    M_shift = np.array([[1.0, 0.0, float(int_acc_dx)], [0.0, 1.0, float(int_acc_dy)]], dtype=np.float32)
    aligned_rx_bg = cv2.warpAffine(rx_mock_bg, M_shift, (new_w, new_h))
    
    # 完美解決慢速移動！直接和「模擬 RX 背景」做差異比對
    diff = cv2.absdiff(curr_gray, aligned_rx_bg)
    _, fg_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)

    # ---- 2. 決定傳送模式 ----
    send_full = force_full_frame
    frames_since_full += 1
    if frames_since_full >= 40:
        send_full = True

    crop_x, crop_y = 0, 0
    target_crop = frame_resized

    if not send_full:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_contours = [c for c in contours if cv2.contourArea(c) > 100]

        if not motion_contours:
            # 畫面與 RX 端已經完美一致，不浪費頻寬傳送任何資料！
            disp = frame_resized.copy()
            cv2.putText(disp, "NO MOTION - PERFECT SYNC", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow('TX_AVIF_Motion_Control', disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: break
            elif key == ord('f'): force_full_frame = True
            continue

        # 合併為單一 Bounding Box
        x_min, y_min = new_w, new_h
        x_max, y_max = 0, 0
        for cnt in motion_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        bbox_area = (x_max - x_min) * (y_max - y_min)
        if bbox_area > new_w * new_h * 0.6:
            send_full = True
        else:
            pad = 20
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(new_w, x_max + pad)
            y_max = min(new_h, y_max + pad)
            crop_x, crop_y = x_min, y_min
            target_crop = frame_resized[y_min:y_max, x_min:x_max]

    # ---- 3. 編碼 (統一使用單一 BBox + AVIF) ----
    frame_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    # 全部用 AVIF，單一區塊不會有跨區塊邊緣污染！
    pil_img.save(buffer, format="AVIF", quality=quality, speed=10)
    frame_bytes = buffer.getvalue()
    pkt_type = 0 if send_full else 1

    tx_dx = max(-32768, min(32767, int_acc_dx))
    tx_dy = max(-32768, min(32767, int_acc_dy))
    
    header = struct.pack(HEADER_FORMAT, b'AVIF', pkt_type, tx_dx, tx_dy, crop_x, crop_y, len(frame_bytes))
    full_payload = header + frame_bytes

    # ---- 4. 零延遲原子推送 ----
    state = PayloadState(
        payload=full_payload,
        dx=tx_dx, 
        dy=tx_dy,
        crop_img=cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY),
        crop_x=crop_x,
        crop_y=crop_y,
        send_full=send_full
    )
    
    tx_busy = False
    with latest_payload_lock:
        if latest_payload_state is None:
            latest_payload_state = state
            if send_full:
                force_full_frame = False
                frames_since_full = 0
        else:
            # 覆蓋未發送的上一幀！因為這幀並未發出，rx_mock_bg 完全不會收到！
            # 舊幀就好像不存在一樣，無損同步確保畫面完美連貫！
            latest_payload_state = state
            tx_busy = True

    # ---- UI & Debug Logs ----
    end_time = time.time()
    
    ch, cw = target_crop.shape[:2]
    area_saving = (1.0 - (cw*ch)/(new_w*new_h))*100 if not send_full else 0
    print(f"[TX] {'FULL' if send_full else 'BBOX'} | "
          f"Crop: {cw}x{ch} | "
          f"Size: {len(full_payload):,} B | "
          f"Saved: {area_saving:.0f}% | "
          f"Busy: {tx_busy} | "
          f"Encode: {end_time - start_time:.3f}s")
          
    disp = frame_resized.copy()
    if pkt_type == 1:
        cv2.rectangle(disp, (crop_x, crop_y), (crop_x + cw, crop_y + ch), (0, 0, 255), 2)
        cv2.putText(disp, f"BBOX AVIF ({len(frame_bytes):,}B)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(disp, f"FULL AVIF ({len(frame_bytes):,}B)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    cv2.putText(disp, f"Level: {current_level}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
    cv2.imshow('TX_AVIF_Motion_Control', disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: break
    elif key in (ord('+'), ord('=')):
        if current_level < 4: current_level += 1; force_full_frame = True
    elif key in (ord('-'), ord('_')):
        if current_level > 0: current_level -= 1; force_full_frame = True
    elif key == ord('f'):
        force_full_frame = True

running = False
cap_thread.join(timeout=1.0)
writer_thread.join(timeout=1.0)
cap.release()
ser.close()
cv2.destroyAllWindows()
