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

if not cap.isOpened():
    print("Error: Could not open web camera")
    sys.exit(1)

# LOCK AUTO EXPOSURE AND WHITE BALANCE
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 限制最高解析度，若相機原始解析度過高(如1080p)，會導致傳輸量太大
if orig_w > 640:
    scale = 640.0 / orig_w
    orig_w = 640
    orig_h = int(orig_h * scale)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0
print(f"Original Camera Resolution: {orig_w}x{orig_h} @ {fps} FPS")

# ====== Thread 1: Async Zero-Latency Video Capture ======
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
rx_mock_bg = None
prev_bbox = None
BBOX_SMOOTH = 0.4

# GMC 狀態：使用絕對總量追蹤，避免累積減法的浮點誤差
total_camera_dx = 0.0
total_camera_dy = 0.0
last_sent_dx = 0.0
last_sent_dy = 0.0
smooth_flow_dx = 0.0
smooth_flow_dy = 0.0
FLOW_EMA = 0.4  # 光流平滑係數

# ★ 防抖：用滑動窗口平滑 dx/dy，消除幀間 ±1px 的跳動
import collections as _collections
_dx_history = _collections.deque(maxlen=5)
_dy_history = _collections.deque(maxlen=5)

# ★ Sky Mask 相關設定
sky_mask_cache = None        # 快取的天空遮罩
sky_mask_update_interval = 30  # 每 N 幀更新一次 sky mask
sky_mask_frame_count = 0
SKY_REFRESH_INTERVAL = 180   # 天空區域每 180 幀才強制刷新一次

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

def detect_sky_mask(frame_bgr, new_h, new_w):
    """
    用色彩特徵偵測天空區域：
    - 天空通常是高亮度、低飽和度（灰白色/淡藍色）
    - 在 HSV 空間中，天空的 S 很低（<80）且 V 很高（>150）
    - 另外結合位置先驗：畫面上半部更可能是天空
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 條件 1：低飽和度 + 高亮度 = 天空（包含灰白陰天和藍天）
    sky_color = ((s < 80) & (v > 140)).astype(np.uint8) * 255
    
    # 條件 2：藍色天空（H 在 90-130 範圍）
    blue_sky = ((h > 90) & (h < 130) & (s > 30) & (v > 100)).astype(np.uint8) * 255
    
    # 合併兩種天空偵測
    sky_raw = cv2.bitwise_or(sky_color, blue_sky)
    
    # 位置先驗權重：畫面越上方越可能是天空
    # 上半部權重 100%，下半部逐漸降到 0%
    position_weight = np.zeros((new_h, new_w), dtype=np.uint8)
    for row in range(new_h):
        # 上方 40% 完全保留，40%~70% 逐漸降低，70% 以下不算天空
        if row < new_h * 0.4:
            position_weight[row, :] = 255
        elif row < new_h * 0.7:
            ratio = 1.0 - (row - new_h * 0.4) / (new_h * 0.3)
            position_weight[row, :] = int(255 * ratio)
        # else: 0
    
    # 結合色彩偵測與位置先驗
    sky_mask = cv2.bitwise_and(sky_raw, position_weight)
    
    # 形態學清理：去除小噪點，填補小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    
    # 用高斯模糊讓邊界平滑，避免天空/地面交界處的突然切換
    sky_mask = cv2.GaussianBlur(sky_mask, (21, 21), 0)
    _, sky_mask = cv2.threshold(sky_mask, 127, 255, cv2.THRESH_BINARY)
    
    return sky_mask

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
            sent_queue.put(state)
            for i in range(0, len(state.payload), c_size):
                chunk = state.payload[i:i + c_size]
                serial_port.write(chunk)
                time.sleep(0.03)
        else:
            time.sleep(0.005)

writer_thread = threading.Thread(target=serial_writer_thread, args=(ser, chunk_size), daemon=True)
writer_thread.start()

# ====== Main Processing Loop ======
while True:
    start_time = time.time()
    
    with latest_frame_lock:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        frame = latest_frame.copy()

    # 將影片調整為基礎解析度
    frame = cv2.resize(frame, (orig_w, orig_h))

    ratio, quality = settings[current_level]
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    frame_resized = cv2.resize(frame, (new_w, new_h))
    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # ★ 定期更新 Sky Mask
    sky_mask_frame_count += 1
    if sky_mask_cache is None or sky_mask_frame_count >= sky_mask_update_interval:
        sky_mask_cache = detect_sky_mask(frame_resized, new_h, new_w)
        sky_mask_frame_count = 0
    # 確保 sky_mask 尺寸正確
    if sky_mask_cache.shape != (new_h, new_w):
        sky_mask_cache = detect_sky_mask(frame_resized, new_h, new_w)
    
    ground_mask = cv2.bitwise_not(sky_mask_cache)  # 地面遮罩 = 非天空區域

    while not sent_queue.empty():
        st = sent_queue.get()
        last_sent_dx += st.dx
        last_sent_dy += st.dy
        
        if st.send_full:
            rx_mock_bg = st.crop_img.copy()
        else:
            M_sent = np.array([[1.0, 0.0, float(st.dx)], [0.0, 1.0, float(st.dy)]], dtype=np.float32)
            rx_mock_bg = cv2.warpAffine(rx_mock_bg, M_sent, (new_w, new_h))
            cy, cx = st.crop_y, st.crop_x
            ch, cw = st.crop_img.shape[:2]
            rx_mock_bg[cy:cy+ch, cx:cx+cw] = st.crop_img

    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        prev_gray = curr_gray.copy()
        rx_mock_bg = curr_gray.copy()
        force_full_frame = True

    # ---- 1. Global Motion Compensation (GMC) ----
    raw_dx, raw_dy = 0.0, 0.0
    # ★ 只在地面區域抓特徵點，避免天空無紋理干擾 GMC
    ground_gray_for_features = cv2.bitwise_and(prev_gray, ground_mask)
    prev_pts = cv2.goodFeaturesToTrack(ground_gray_for_features, maxCorners=150, qualityLevel=0.2, minDistance=7)
    if prev_pts is not None and len(prev_pts) > 4:
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is not None:
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            if len(good_prev) > 4:
                matrix, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
                if matrix is not None:
                    raw_dx, raw_dy = matrix[0, 2], matrix[1, 2]

    # ★ EMA 平滑光流
    smooth_flow_dx = smooth_flow_dx * (1 - FLOW_EMA) + raw_dx * FLOW_EMA
    smooth_flow_dy = smooth_flow_dy * (1 - FLOW_EMA) + raw_dy * FLOW_EMA
    
    # ★ 死區 0.3px：過濾掉子像素雜訊，但仍允許低速飛行的真實位移通過
    if abs(smooth_flow_dx) > 0.3 or abs(smooth_flow_dy) > 0.3:
        total_camera_dx += smooth_flow_dx
        total_camera_dy += smooth_flow_dy
    
    prev_gray = curr_gray.copy()

    raw_acc_dx = total_camera_dx - last_sent_dx
    raw_acc_dy = total_camera_dy - last_sent_dy
    
    # ★ 防抖：將 dx/dy 放入滑動窗口做中位數濾波，消除 ±1px 的跳動
    _dx_history.append(raw_acc_dx)
    _dy_history.append(raw_acc_dy)
    int_acc_dx = int(round(sorted(_dx_history)[len(_dx_history)//2]))
    int_acc_dy = int(round(sorted(_dy_history)[len(_dy_history)//2]))
    
    M_shift = np.array([[1.0, 0.0, float(int_acc_dx)], [0.0, 1.0, float(int_acc_dy)]], dtype=np.float32)
    aligned_rx_bg = cv2.warpAffine(rx_mock_bg, M_shift, (new_w, new_h))
    
    diff = cv2.absdiff(curr_gray, aligned_rx_bg)
    
    # ★ 分離天空與地面的差異處理
    # 地面：低閾值，捕捉每一個細微變化（行人、動物等）
    ground_diff = cv2.bitwise_and(diff, ground_mask)
    ground_noise_std = np.std(ground_diff[ground_mask > 0]) if np.any(ground_mask > 0) else 10
    ground_thresh = max(8, min(25, int(ground_noise_std * 2.5 + 5)))
    _, ground_fg = cv2.threshold(ground_diff, ground_thresh, 255, cv2.THRESH_BINARY)
    ground_fg = cv2.bitwise_and(ground_fg, ground_mask)
    
    # 天空：高閾值，忽略壓縮殘差和光流補償誤差
    sky_diff = cv2.bitwise_and(diff, sky_mask_cache)
    _, sky_fg = cv2.threshold(sky_diff, 50, 255, cv2.THRESH_BINARY)
    sky_fg = cv2.bitwise_and(sky_fg, sky_mask_cache)
    
    # 合併：地面敏感偵測 + 天空寬鬆偵測
    fg_mask = cv2.bitwise_or(ground_fg, sky_fg)

    # ★ 更小的 OPEN kernel：不要把小目標（行人）洗掉
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    # CLOSE 填孔
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # ★ 更小的膨脹核心，避免小目標膨脹後與天空連在一起
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)

    # ---- 2. 決定傳送模式 ----
    send_full = force_full_frame
    frames_since_full += 1
    # ★ 延長 full frame 間隔到 SKY_REFRESH_INTERVAL，因為天空遮罩已經處理了天空問題
    if frames_since_full >= SKY_REFRESH_INTERVAL:
        send_full = True

    crop_x, crop_y = 0, 0
    target_crop = frame_resized

    if not send_full:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # ★ 降低面積門檻到 30，讓遠處小行人也能被捕捉
        motion_contours = [c for c in contours if cv2.contourArea(c) > 30]

        if not motion_contours:
            if prev_bbox is not None:
                x_min, y_min, x_max, y_max = prev_bbox
                crop_x, crop_y = x_min, y_min
                target_crop = frame_resized[y_min:y_max, x_min:x_max]
                prev_bbox = None
            else:
                # ★ 即使偵測為 no motion，仍然每 5 幀發送一次 ground-only 差異幀
                # 防止低速飛行時 RX 完全沒收到更新
                if frames_since_full % 5 == 0:
                    # 發送整個地面區域的小幅更新
                    ground_rows = np.where(np.any(ground_mask > 0, axis=1))[0]
                    if len(ground_rows) > 0:
                        y_min_g = int(ground_rows[0])
                        y_max_g = int(ground_rows[-1]) + 1
                        crop_x, crop_y = 0, y_min_g
                        target_crop = frame_resized[y_min_g:y_max_g, 0:new_w]
                    # 如果地面也沒有就跳過
                    else:
                        disp = frame_resized.copy()
                        cv2.putText(disp, "NO MOTION", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.imshow('TX_AVIF_Motion_Control', disp)
                        cv2.imshow('TX_Original_Source', frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27: break
                        elif key == ord('f'): force_full_frame = True
                        continue
                else:
                    disp = frame_resized.copy()
                    cv2.putText(disp, "NO MOTION - SKIP", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow('TX_AVIF_Motion_Control', disp)
                    cv2.imshow('TX_Original_Source', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27: break
                    elif key == ord('f'): force_full_frame = True
                    continue
        else:
            # 合併為單一 Bounding Box
            x_min, y_min = new_w, new_h
            x_max, y_max = 0, 0
            for cnt in motion_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            # EMA 時間平滑
            if prev_bbox is not None:
                px_min, py_min, px_max, py_max = prev_bbox
                s = BBOX_SMOOTH
                x_min = int(px_min * (1-s) + x_min * s)
                y_min = int(py_min * (1-s) + y_min * s)
                x_max = int(px_max * (1-s) + x_max * s)
                y_max = int(py_max * (1-s) + y_max * s)
                for cnt in motion_contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    x_min = min(x_min, bx)
                    y_min = min(y_min, by)
                    x_max = max(x_max, bx + bw)
                    y_max = max(y_max, by + bh)

            bbox_area = (x_max - x_min) * (y_max - y_min)
            # ★ 提高 full frame 觸發比例到 0.75（原本 0.6），讓 BBox 模式更有機會被使用
            if bbox_area > new_w * new_h * 0.75:
                send_full = True
            else:
                pad = 20  # ★ 縮小 padding，減少不必要的天空被包進來
                x_min = max(0, x_min - pad)
                y_min = max(0, y_min - pad)
                x_max = min(new_w, x_max + pad)
                y_max = min(new_h, y_max + pad)
                crop_x, crop_y = x_min, y_min
                target_crop = frame_resized[y_min:y_max, x_min:x_max]
                prev_bbox = (x_min, y_min, x_max, y_max)

    # ---- 3. 編碼 (統一使用單一 BBox + AVIF) ----
    frame_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
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
                prev_bbox = None
                total_camera_dx = 0.0
                total_camera_dy = 0.0
                last_sent_dx = 0.0
                last_sent_dy = 0.0
                smooth_flow_dx = 0.0
                smooth_flow_dy = 0.0
        else:
            latest_payload_state = state
            tx_busy = True

    # ---- UI & Debug Logs ----
    end_time = time.time()
    
    ch, cw = target_crop.shape[:2]
    area_saving = (1.0 - (cw*ch)/(new_w*new_h))*100 if not send_full else 0
    sky_pct = np.count_nonzero(sky_mask_cache) / (new_w * new_h) * 100
    print(f"[TX] {'FULL' if send_full else 'BBOX'} | "
          f"Crop: {cw}x{ch} | "
          f"Size: {len(full_payload):,} B | "
          f"Saved: {area_saving:.0f}% | "
          f"Sky: {sky_pct:.0f}% | "
          f"GThresh: {ground_thresh} | "
          f"Busy: {tx_busy} | "
          f"Encode: {end_time - start_time:.3f}s")
          
    disp = frame_resized.copy()
    # ★ 顯示 sky mask 邊界（青色線）
    sky_contours, _ = cv2.findContours(sky_mask_cache, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(disp, sky_contours, -1, (255, 255, 0), 1)
    
    if pkt_type == 1:
        cv2.rectangle(disp, (crop_x, crop_y), (crop_x + cw, crop_y + ch), (0, 0, 255), 2)
        cv2.putText(disp, f"BBOX AVIF ({len(frame_bytes):,}B)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(disp, f"FULL AVIF ({len(frame_bytes):,}B)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    cv2.putText(disp, f"Level: {current_level} | Sky: {sky_pct:.0f}%", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
    cv2.imshow('TX_AVIF_Motion_Control', disp)
    
    # 顯示原始影像
    cv2.imshow('TX_Original_Source', frame)
    
    # 顯示最後傳輸出去的 AVIF 影像
    try:
        transmitted_pil = Image.open(io.BytesIO(frame_bytes))
        transmitted_cv = cv2.cvtColor(np.array(transmitted_pil), cv2.COLOR_RGB2BGR)
        final_disp = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        if send_full:
            final_disp = transmitted_cv
        else:
            final_disp[crop_y:crop_y+ch, crop_x:crop_x+cw] = transmitted_cv
        cv2.imshow('TX_Transmitted_Quality_AVIF', final_disp)
    except Exception as e:
        pass

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
