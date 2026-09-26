"""
Motion_experiment.py
====================
實驗目的：比較「全幀 AVIF 壓縮 (v3)」與「動態差分後 AVIF 壓縮 (v4)」的資料量差異。

操作變因（唯一改變的）：
  - 全幀壓縮 (v3)   → 每幀傳送整張畫面
  - 動態差分 (v4)   → 每幀只傳送有動態變化的 BBox 區塊 (+ 17-byte header)

控制變因（與原始程式完全相同）：
  - 輸入影片：同一支 mp4
  - 解析度限制：orig_w <= 640（超過就等比縮放）
  - settings 對照表：{4:[0.64,29], 3:[0.59,25], 2:[0.48,22], 1:[0.38,23], 0:[0.34,22]}
  - 預設 Level = 4
  - AVIF 編碼參數：quality=quality, speed=10
  - v4 動態偵測所有參數：BBOX_SMOOTH=0.4, FLOW_EMA=0.4, 死區=0.3px
    SKY_REFRESH_INTERVAL=180, sky_mask_update_interval=30, pad=20, BBox面積觸發full=0.75
    ground_thresh = max(15, min(35, int(noise_std*3.5+8))), sky_thresh=50  ← 與 noSkyMask 版統一
    OPEN kernel (3,3), CLOSE kernel ellipse(11,11), dilate ellipse(15,15)
    maxCorners=150, qualityLevel=0.2, minDistance=7 (GMC)
    deque maxlen=5 (防抖)

執行方式：
  python Motion_experiment.py --video fly_1.mp4 --level 4

按鍵說明（執行期間）：
  q / ESC  → 提前結束並顯示統計圖
  +        → 提高畫質 Level（0~4）
  -        → 降低畫質 Level
  f        → 強制 v4 發送一次全幀
"""

import cv2
import numpy as np
from PIL import Image
import io
import sys
import struct
import collections
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import pillow_avif
except ImportError:
    print("Error: pillow_avif module not found. Please install: pip install pillow-avif-plugin")
    sys.exit(1)

# ============================================================
# 命令列參數
# ============================================================
parser = argparse.ArgumentParser(description="AVIF Full-Frame vs Motion-Differential Benchmark")
parser.add_argument("--video", type=str, default="fly_1.mp4", help="輸入影片路徑")
parser.add_argument("--level", type=int, default=4, choices=[0,1,2,3,4], help="初始畫質 Level (0~4)")
args = parser.parse_args()

# ============================================================
# 共用設定（控制變因）
# ============================================================
settings = {
    4: [0.64, 29],
    3: [0.59, 25],
    2: [0.48, 22],
    1: [0.38, 23],
    0: [0.34, 22]
}
current_level = args.level

# ============================================================
# 開啟影片（兩個版本共用同一個 cap，逐幀同步讀取）
# ============================================================
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Error: Could not open video file '{args.video}'")
    sys.exit(1)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 與原始程式相同：解析度限制 640
if orig_w > 640:
    scale = 640.0 / orig_w
    orig_w = 640
    orig_h = int(orig_h * scale)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0

print(f"Video: {args.video}  |  Resolution: {orig_w}x{orig_h}  |  FPS: {fps:.1f}")
print(f"Initial Level: {current_level}  |  Settings: ratio={settings[current_level][0]}, quality={settings[current_level][1]}")
print("\n--- Controls ---")
print("Press '+'/'-' to change quality level")
print("Press 'f' to force v4 full-frame")
print("Press 'q' / ESC to stop and show chart")
print("----------------\n")

# ============================================================
# v4 動態差分的所有狀態變數（控制變因：與 Tx_AVIF_v4_Motion_Video_Once.py 完全相同）
# ============================================================
HEADER_FORMAT     = '<4sBhhHHI'
HEADER_SIZE       = struct.calcsize(HEADER_FORMAT)
BBOX_SMOOTH       = 0.4
FLOW_EMA          = 0.4
SKY_REFRESH_INTERVAL     = 180
sky_mask_update_interval = 30

force_full_frame  = True
frames_since_full = 0
prev_gray         = None
rx_mock_bg        = None
prev_bbox         = None

total_camera_dx   = 0.0
total_camera_dy   = 0.0
last_sent_dx      = 0.0
last_sent_dy      = 0.0
smooth_flow_dx    = 0.0
smooth_flow_dy    = 0.0

_dx_history = collections.deque(maxlen=5)
_dy_history = collections.deque(maxlen=5)

sky_mask_cache        = None
sky_mask_frame_count  = 0

# ============================================================
# Sky Mask 偵測（與 v4 完全相同）
# ============================================================
def detect_sky_mask(frame_bgr, new_h, new_w):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    sky_color = ((s < 80) & (v > 140)).astype(np.uint8) * 255
    blue_sky  = ((h > 90) & (h < 130) & (s > 30) & (v > 100)).astype(np.uint8) * 255
    sky_raw   = cv2.bitwise_or(sky_color, blue_sky)
    position_weight = np.zeros((new_h, new_w), dtype=np.uint8)
    for row in range(new_h):
        if row < new_h * 0.4:
            position_weight[row, :] = 255
        elif row < new_h * 0.7:
            ratio_val = 1.0 - (row - new_h * 0.4) / (new_h * 0.3)
            position_weight[row, :] = int(255 * ratio_val)
    sky_mask = cv2.bitwise_and(sky_raw, position_weight)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    sky_mask = cv2.GaussianBlur(sky_mask, (21, 21), 0)
    _, sky_mask = cv2.threshold(sky_mask, 127, 255, cv2.THRESH_BINARY)
    return sky_mask

# ============================================================
# 統計記錄
# ============================================================
# v3：記錄整幀 AVIF bytes 大小（無 header）
v3_frame_sizes = []
# v4：記錄實際 payload 大小（AVIF bytes + 17-byte header），跳幀記 0
v4_frame_sizes = []

total_frames = 0

# ============================================================
# 主迴圈：每幀同時跑 v3 編碼 和 v4 動態差分編碼
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # ---------- 共用前處理（控制變因）----------
    frame = cv2.resize(frame, (orig_w, orig_h))
    ratio, quality = settings[current_level]
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    frame_resized = cv2.resize(frame, (new_w, new_h))

    # ============================================================
    # [方法 A] V3：全幀 AVIF 壓縮
    # ============================================================
    frame_rgb_v3 = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    pil_v3       = Image.fromarray(frame_rgb_v3)
    buf_v3       = io.BytesIO()
    pil_v3.save(buf_v3, format="AVIF", quality=quality, speed=10)
    v3_bytes     = buf_v3.getvalue()
    v3_size      = len(v3_bytes)   # 無 header，只記 AVIF 壓縮大小
    v3_frame_sizes.append(v3_size)

    # ============================================================
    # [方法 B] V4：動態差分 AVIF 壓縮（與 Tx_AVIF_v4_Motion_Video_Once.py 完全相同）
    # ============================================================
    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Sky Mask 更新
    sky_mask_frame_count += 1
    if sky_mask_cache is None or sky_mask_frame_count >= sky_mask_update_interval:
        sky_mask_cache = detect_sky_mask(frame_resized, new_h, new_w)
        sky_mask_frame_count = 0
    if sky_mask_cache.shape != (new_h, new_w):
        sky_mask_cache = detect_sky_mask(frame_resized, new_h, new_w)
    ground_mask = cv2.bitwise_not(sky_mask_cache)

    # 同步 mock background（v4 內部 sent_queue 的邏輯：實驗版直接同步更新）
    # 注意：實驗中不需要非同步佇列，直接在主迴圈同步模擬 RX mock
    # （這樣才能讓 rx_mock_bg 與 v4 原版的行為完全一致）

    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        prev_gray  = curr_gray.copy()
        rx_mock_bg = curr_gray.copy()
        force_full_frame = True

    # GMC
    raw_dx, raw_dy = 0.0, 0.0
    ground_gray_for_features = cv2.bitwise_and(prev_gray, ground_mask)
    prev_pts = cv2.goodFeaturesToTrack(ground_gray_for_features, maxCorners=150, qualityLevel=0.2, minDistance=7)
    if prev_pts is not None and len(prev_pts) > 4:
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is not None:
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            if len(good_prev) > 4:
                matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_curr)
                if matrix is not None:
                    raw_dx, raw_dy = matrix[0, 2], matrix[1, 2]

    smooth_flow_dx = smooth_flow_dx * (1 - FLOW_EMA) + raw_dx * FLOW_EMA
    smooth_flow_dy = smooth_flow_dy * (1 - FLOW_EMA) + raw_dy * FLOW_EMA

    if abs(smooth_flow_dx) > 0.3 or abs(smooth_flow_dy) > 0.3:
        total_camera_dx += smooth_flow_dx
        total_camera_dy += smooth_flow_dy

    prev_gray = curr_gray.copy()

    raw_acc_dx = total_camera_dx - last_sent_dx
    raw_acc_dy = total_camera_dy - last_sent_dy
    _dx_history.append(raw_acc_dx)
    _dy_history.append(raw_acc_dy)
    int_acc_dx = int(round(sorted(_dx_history)[len(_dx_history)//2]))
    int_acc_dy = int(round(sorted(_dy_history)[len(_dy_history)//2]))

    M_shift      = np.array([[1.0, 0.0, float(int_acc_dx)], [0.0, 1.0, float(int_acc_dy)]], dtype=np.float32)
    aligned_rx_bg = cv2.warpAffine(rx_mock_bg, M_shift, (new_w, new_h))
    diff          = cv2.absdiff(curr_gray, aligned_rx_bg)

    ground_diff      = cv2.bitwise_and(diff, ground_mask)
    ground_noise_std = np.std(ground_diff[ground_mask > 0]) if np.any(ground_mask > 0) else 10
    ground_thresh    = max(15, min(35, int(ground_noise_std * 3.5 + 8)))
    _, ground_fg     = cv2.threshold(ground_diff, ground_thresh, 255, cv2.THRESH_BINARY)
    ground_fg        = cv2.bitwise_and(ground_fg, ground_mask)

    sky_diff  = cv2.bitwise_and(diff, sky_mask_cache)
    _, sky_fg = cv2.threshold(sky_diff, 50, 255, cv2.THRESH_BINARY)
    sky_fg    = cv2.bitwise_and(sky_fg, sky_mask_cache)

    fg_mask = cv2.bitwise_or(ground_fg, sky_fg)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_RECT,    (3,  3 )))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    fg_mask = cv2.dilate(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    # 決定傳送模式
    send_full = force_full_frame
    frames_since_full += 1
    if frames_since_full >= SKY_REFRESH_INTERVAL:
        send_full = True

    crop_x, crop_y = 0, 0
    target_crop    = frame_resized

    if not send_full:
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_contours = [c for c in contours if cv2.contourArea(c) > 30]

        if not motion_contours:
            if prev_bbox is not None:
                x_min, y_min, x_max, y_max = prev_bbox
                crop_x, crop_y = x_min, y_min
                target_crop    = frame_resized[y_min:y_max, x_min:x_max]
                prev_bbox      = None
            else:
                # 每 5 幀發一次地面更新；其餘幀跳過，記 0
                if frames_since_full % 5 == 0:
                    ground_rows = np.where(np.any(ground_mask > 0, axis=1))[0]
                    if len(ground_rows) > 0:
                        y_min_g   = int(ground_rows[0])
                        y_max_g   = int(ground_rows[-1]) + 1
                        crop_x, crop_y = 0, y_min_g
                        target_crop    = frame_resized[y_min_g:y_max_g, 0:new_w]
                    else:
                        # 跳過此幀
                        v4_frame_sizes.append(0)
                        total_frames += 1
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27: break
                        elif key == ord('f'): force_full_frame = True
                        continue
                else:
                    # 跳過此幀
                    v4_frame_sizes.append(0)
                    total_frames += 1
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27: break
                    elif key == ord('f'): force_full_frame = True
                    continue
        else:
            x_min, y_min = new_w, new_h
            x_max, y_max = 0, 0
            for cnt in motion_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

            if prev_bbox is not None:
                px_min, py_min, px_max, py_max = prev_bbox
                s     = BBOX_SMOOTH
                x_min = int(px_min*(1-s) + x_min*s)
                y_min = int(py_min*(1-s) + y_min*s)
                x_max = int(px_max*(1-s) + x_max*s)
                y_max = int(py_max*(1-s) + y_max*s)
                for cnt in motion_contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    x_min = min(x_min, bx)
                    y_min = min(y_min, by)
                    x_max = max(x_max, bx + bw)
                    y_max = max(y_max, by + bh)

            bbox_area = (x_max - x_min) * (y_max - y_min)
            if bbox_area > new_w * new_h * 0.75:
                send_full = True
            else:
                pad    = 20
                x_min  = max(0,    x_min - pad)
                y_min  = max(0,    y_min - pad)
                x_max  = min(new_w, x_max + pad)
                y_max  = min(new_h, y_max + pad)
                crop_x, crop_y = x_min, y_min
                target_crop    = frame_resized[y_min:y_max, x_min:x_max]
                prev_bbox      = (x_min, y_min, x_max, y_max)

    # AVIF 編碼
    frame_rgb_v4 = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
    pil_v4       = Image.fromarray(frame_rgb_v4)
    buf_v4       = io.BytesIO()
    pil_v4.save(buf_v4, format="AVIF", quality=quality, speed=10)
    frame_bytes_v4 = buf_v4.getvalue()
    pkt_type       = 0 if send_full else 1

    tx_dx = max(-32768, min(32767, int_acc_dx))
    tx_dy = max(-32768, min(32767, int_acc_dy))
    header     = struct.pack(HEADER_FORMAT, b'AVIF', pkt_type, tx_dx, tx_dy, crop_x, crop_y, len(frame_bytes_v4))
    full_payload = header + frame_bytes_v4
    v4_size      = len(full_payload)   # AVIF bytes + 17-byte header
    v4_frame_sizes.append(v4_size)

    # 同步更新 rx_mock_bg（模擬 sent_queue 的行為）
    last_sent_dx += tx_dx
    last_sent_dy += tx_dy
    crop_img = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
    ch_crop, cw_crop = crop_img.shape[:2]
    if send_full:
        rx_mock_bg = crop_img.copy()
        force_full_frame  = False
        frames_since_full = 0
        prev_bbox         = None
        total_camera_dx   = 0.0
        total_camera_dy   = 0.0
        last_sent_dx      = 0.0
        last_sent_dy      = 0.0
        smooth_flow_dx    = 0.0
        smooth_flow_dy    = 0.0
    else:
        M_sent     = np.array([[1.0, 0.0, float(tx_dx)], [0.0, 1.0, float(tx_dy)]], dtype=np.float32)
        rx_mock_bg = cv2.warpAffine(rx_mock_bg, M_sent, (new_w, new_h))
        rx_mock_bg[crop_y:crop_y+ch_crop, crop_x:crop_x+cw_crop] = crop_img

    total_frames += 1

    # ---- 終端機統計 ----
    sky_pct     = np.count_nonzero(sky_mask_cache) / (new_w * new_h) * 100
    area_saving = (1.0 - (cw_crop*ch_crop)/(new_w*new_h)) * 100 if not send_full else 0
    saving_vs_v3 = (1.0 - v4_size / v3_size) * 100 if v3_size > 0 else 0
    print(f"Frame {total_frames:4d} | Level {current_level} | "
          f"V3(Full): {v3_size:6,} B | "
          f"V4({'FULL' if send_full else 'BBOX'}): {v4_size:6,} B | "
          f"Saving vs V3: {saving_vs_v3:+.1f}% | "
          f"Sky: {sky_pct:.0f}%")

    # ---- UI 顯示 ----
    disp_v3 = frame_resized.copy()
    cv2.putText(disp_v3, f"V3 FULL AVIF | Level {current_level}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(disp_v3, f"Size: {v3_size:,} B", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(disp_v3, f"Ratio: {ratio} | Quality: {quality}", (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)

    disp_v4 = frame_resized.copy()
    sky_contours, _ = cv2.findContours(sky_mask_cache, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(disp_v4, sky_contours, -1, (255,255,0), 1)
    if pkt_type == 1:
        cv2.rectangle(disp_v4, (crop_x, crop_y), (crop_x+cw_crop, crop_y+ch_crop), (0,0,255), 2)
        cv2.putText(disp_v4, f"V4 BBOX AVIF | Saved {area_saving:.0f}%", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    else:
        cv2.putText(disp_v4, f"V4 FULL AVIF | Level {current_level}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(disp_v4, f"Size: {v4_size:,} B", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,255), 2)
    cv2.putText(disp_v4, f"Sky: {sky_pct:.0f}% | GThresh: {ground_thresh}", (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,220), 1)

    combined = np.hstack([disp_v3, disp_v4])
    cv2.putText(combined, "V3: Full Frame (left)   |   V4: Motion Diff (right)", (5, combined.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.imshow("Motion_experiment | Left:V3  Right:V4  |  q=quit  f=force-full  +/-=level", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key in (ord('+'), ord('=')):
        if current_level < 4:
            current_level += 1
            force_full_frame = True
            print(f">>> Switched to Level {current_level}")
    elif key in (ord('-'), ord('_')):
        if current_level > 0:
            current_level -= 1
            force_full_frame = True
            print(f">>> Switched to Level {current_level}")
    elif key == ord('f'):
        force_full_frame = True

cap.release()
cv2.destroyAllWindows()

# ============================================================
# 最終統計與折線圖
# ============================================================
# 補齊長度（防止 v4 因 continue 造成的長度差異）
max_len = max(len(v3_frame_sizes), len(v4_frame_sizes))
v3_frame_sizes += [0] * (max_len - len(v3_frame_sizes))
v4_frame_sizes += [0] * (max_len - len(v4_frame_sizes))

v3_total = sum(v3_frame_sizes)
v4_total = sum(v4_frame_sizes)
total_saving = (1 - v4_total / v3_total) * 100 if v3_total > 0 else 0

print("\n" + "="*60)
print(f"{'Experiment Results':^60}")
print("="*60)
print(f"  Video  : {args.video}")
print(f"  Level  : {current_level}  |  Ratio: {settings[current_level][0]}  |  Quality: {settings[current_level][1]}")
print(f"  Frames : {total_frames} frames")
print(f"  V3 Full Frame  -> Total: {v3_total:,} bytes  ({v3_total/1024:.1f} KB)")
print(f"  V4 Motion Diff -> Total: {v4_total:,} bytes  ({v4_total/1024:.1f} KB)")
print(f"  Saved          : {v3_total - v4_total:,} bytes ({total_saving:.1f}%)")
print("="*60 + "\n")

# ---- 折線圖 ----
frames_axis = list(range(1, max_len + 1))

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"AVIF Encoding Benchmark  |  Video: {args.video}  |  Level {current_level}  "
    f"(ratio={settings[current_level][0]}, quality={settings[current_level][1]})",
    fontsize=13, fontweight='bold'
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# 子圖 1：V3 每幀大小折線圖
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(frames_axis, v3_frame_sizes, color='tomato', linewidth=0.8, label='V3 Full Frame')
ax1.set_title('V3: Full Frame AVIF - Size per Frame', fontsize=11)
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('Bytes')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 子圖 2：V4 每幀大小折線圖
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(frames_axis, v4_frame_sizes, color='steelblue', linewidth=0.8, label='V4 Motion Diff')
ax2.set_title('V4: Motion Diff AVIF - Size per Frame', fontsize=11)
ax2.set_xlabel('Frame Number')
ax2.set_ylabel('Bytes')
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# 子圖 3：V3 vs V4 疊加比較折線圖
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(frames_axis, v3_frame_sizes, color='tomato',   linewidth=0.8, alpha=0.8, label='V3 Full Frame')
ax3.plot(frames_axis, v4_frame_sizes, color='steelblue',linewidth=0.8, alpha=0.8, label='V4 Motion Diff')
ax3.set_title('V3 vs V4 - Size per Frame Comparison', fontsize=11)
ax3.set_xlabel('Frame Number')
ax3.set_ylabel('Bytes')
ax3.set_xlim(left=0)
ax3.set_ylim(bottom=0)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend()

# 子圖 4：總量長條圖與節省量文字
ax4 = fig.add_subplot(gs[1, 1])
labels = ['V3\nFull Frame', 'V4\nMotion Diff']
values = [v3_total, v4_total]
colors = ['tomato', 'steelblue']
bars   = ax4.bar(labels, [v / 1024 for v in values], color=colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val/1024:.1f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_title('Total Data Size Comparison', fontsize=11)
ax4.set_ylabel('Total Size (KB)')
ax4.set_ylim(bottom=0)
ax4.grid(True, axis='y', linestyle='--', alpha=0.6)
# Annotate savings
ax4.text(0.5, 0.92,
         f"V4 saves {total_saving:.1f}%  ({(v3_total-v4_total)/1024:.1f} KB)",
         transform=ax4.transAxes, ha='center', fontsize=11,
         color='green' if total_saving > 0 else 'red', fontweight='bold')

plt.savefig("motion_experiment_result.png", dpi=150, bbox_inches='tight')
print("Chart saved as motion_experiment_result.png")
plt.show()
