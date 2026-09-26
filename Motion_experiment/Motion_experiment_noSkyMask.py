"""
Motion_experiment_noSkyMask.py
==============================
實驗目的：比較「全幀 AVIF 壓縮 (v3)」與「動態差分後 AVIF 壓縮 (v4，無 Sky Mask)」的資料量差異。

操作變因（唯一改變的）：
  - 全幀壓縮 (v3)              → 每幀傳送整張畫面
  - 動態差分，無 Sky Mask (v4) → 對整張差分圖套用自適應閾值，找出動態 BBox 後只傳該區塊

控制變因（與 Tx_AVIF_v4_Motion_Video_Once.py 完全相同，僅移除 Sky Mask 功能）：
  - settings: {4:[0.64,29], 3:[0.59,25], 2:[0.48,22], 1:[0.38,23], 0:[0.34,22]}
  - 預設 Level = 4
  - AVIF 編碼參數: quality=quality, speed=10
  - BBOX_SMOOTH = 0.4
  - FLOW_EMA = 0.4, 死區 = 0.3px
  - GMC: maxCorners=150, qualityLevel=0.2, minDistance=7（作用於整張 prev_gray）
  - 防抖 deque maxlen=5
  - SKY_REFRESH_INTERVAL = 180（仍保留強制全幀間隔機制）
  - 差分形態學: OPEN rect(3,3), CLOSE ellipse(11,11), dilate ellipse(15,15)
  - BBox 觸發全幀面積閾值 = 0.75
  - BBox padding = 20
  - 無動態跳幀邏輯: 每 5 幀送一次地面行更新（此版本改為全畫面更新）

與原版的差異（已移除的功能）：
  - detect_sky_mask()          → 移除
  - 天空/地面分離閾值處理        → 改為整張差分圖自適應閾值
  - Sky Mask 更新計數器          → 移除
  - GMC 使用 ground_mask 遮罩   → 改為直接用整張 prev_gray
  - UI 顯示 sky mask 邊界        → 移除

執行方式：
  python Motion_experiment_noSkyMask.py --video fly_1.mp4 --level 2

按鍵說明：
  q / ESC  → 停止並顯示統計圖
  +        → 提高 Level
  -        → 降低 Level
  f        → 強制 v4 全幀
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
    print("Error: pillow_avif not found. Run: pip install pillow-avif-plugin")
    sys.exit(1)

# ============================================================
# 命令列參數
# ============================================================
parser = argparse.ArgumentParser(description="AVIF Full-Frame vs Motion-Diff (No Sky Mask) Benchmark")
parser.add_argument("--video", type=str, default="fly_1.mp4", help="Input video path")
parser.add_argument("--level", type=int, default=4, choices=[0,1,2,3,4], help="Initial quality level (0~4)")
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
# 開啟影片
# ============================================================
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Error: Could not open video file '{args.video}'")
    sys.exit(1)

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if orig_w > 640:
    scale  = 640.0 / orig_w
    orig_w = 640
    orig_h = int(orig_h * scale)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0

print(f"Video: {args.video}  |  Resolution: {orig_w}x{orig_h}  |  FPS: {fps:.1f}")
print(f"Initial Level: {current_level}  |  ratio={settings[current_level][0]}, quality={settings[current_level][1]}")
print("\n--- Controls ---")
print("Press '+'/'-' to change level  |  'f' to force full-frame  |  'q'/ESC to quit")
print("----------------\n")

# ============================================================
# V4（無 Sky Mask）狀態變數
# ============================================================
HEADER_FORMAT        = '<4sBhhHHI'
HEADER_SIZE          = struct.calcsize(HEADER_FORMAT)
BBOX_SMOOTH          = 0.4
FLOW_EMA             = 0.4
SKY_REFRESH_INTERVAL = 180      # 每 180 幀強制送一次全幀（保留機制，與原版一致）

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

# ============================================================
# 統計記錄
# ============================================================
v3_frame_sizes = []   # V3：整幀 AVIF bytes（無 header）
v4_frame_sizes = []   # V4：AVIF bytes + 17-byte header（跳幀記 0）
total_frames   = 0

# ============================================================
# 主迴圈
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # ---------- 共用前處理 ----------
    frame         = cv2.resize(frame, (orig_w, orig_h))
    ratio, quality = settings[current_level]
    new_w         = int(orig_w * ratio)
    new_h         = int(orig_h * ratio)
    frame_resized = cv2.resize(frame, (new_w, new_h))

    # ============================================================
    # [方法 A] V3：全幀 AVIF 壓縮
    # ============================================================
    buf_v3 = io.BytesIO()
    Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)).save(
        buf_v3, format="AVIF", quality=quality, speed=10)
    v3_size = len(buf_v3.getvalue())
    v3_frame_sizes.append(v3_size)

    # ============================================================
    # [方法 B] V4：動態差分（無 Sky Mask）
    # ============================================================
    curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # 初始化
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        prev_gray        = curr_gray.copy()
        rx_mock_bg       = curr_gray.copy()
        force_full_frame = True

    # ---- GMC（全畫面特徵點，不用 ground_mask 遮罩） ----
    raw_dx, raw_dy = 0.0, 0.0
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=150, qualityLevel=0.2, minDistance=7)
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

    M_shift       = np.array([[1.0, 0.0, float(int_acc_dx)],
                               [0.0, 1.0, float(int_acc_dy)]], dtype=np.float32)
    aligned_rx_bg = cv2.warpAffine(rx_mock_bg, M_shift, (new_w, new_h))
    diff          = cv2.absdiff(curr_gray, aligned_rx_bg)

    # ---- 統一自適應閾值（整張差分圖，無 Sky Mask 區分） ----
    noise_std      = np.std(diff)
    adaptive_thresh = max(15, min(35, int(noise_std * 3.5 + 8)))
    _, fg_mask     = cv2.threshold(diff, adaptive_thresh, 255, cv2.THRESH_BINARY)

    # ---- 形態學（與原版 v4_Motion_Video_Once 參數相同） ----
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT,    (3,  3 )))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    fg_mask = cv2.dilate(fg_mask,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

    # ---- 決定傳送模式 ----
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
                if frames_since_full % 5 == 0:
                    # 每 5 幀送一次全畫面行更新（無 sky/ground 分離，直接全列）
                    crop_x, crop_y = 0, 0
                    target_crop    = frame_resized
                else:
                    # 跳幀
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
                x_min = min(x_min, x);  y_min = min(y_min, y)
                x_max = max(x_max, x+w); y_max = max(y_max, y+h)

            if prev_bbox is not None:
                px_min, py_min, px_max, py_max = prev_bbox
                s     = BBOX_SMOOTH
                x_min = int(px_min*(1-s) + x_min*s)
                y_min = int(py_min*(1-s) + y_min*s)
                x_max = int(px_max*(1-s) + x_max*s)
                y_max = int(py_max*(1-s) + y_max*s)
                for cnt in motion_contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    x_min = min(x_min, bx);    y_min = min(y_min, by)
                    x_max = max(x_max, bx+bw); y_max = max(y_max, by+bh)

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

    # ---- AVIF 編碼 ----
    buf_v4 = io.BytesIO()
    Image.fromarray(cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)).save(
        buf_v4, format="AVIF", quality=quality, speed=10)
    frame_bytes_v4 = buf_v4.getvalue()
    pkt_type       = 0 if send_full else 1

    tx_dx = max(-32768, min(32767, int_acc_dx))
    tx_dy = max(-32768, min(32767, int_acc_dy))
    header       = struct.pack(HEADER_FORMAT, b'AVIF', pkt_type,
                               tx_dx, tx_dy, crop_x, crop_y, len(frame_bytes_v4))
    full_payload = header + frame_bytes_v4
    v4_size      = len(full_payload)
    v4_frame_sizes.append(v4_size)

    # ---- 同步更新 rx_mock_bg ----
    last_sent_dx += tx_dx
    last_sent_dy += tx_dy
    crop_img     = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
    ch_c, cw_c   = crop_img.shape[:2]
    if send_full:
        rx_mock_bg       = crop_img.copy()
        force_full_frame  = False
        frames_since_full = 0
        prev_bbox         = None
        total_camera_dx   = 0.0; total_camera_dy  = 0.0
        last_sent_dx      = 0.0; last_sent_dy      = 0.0
        smooth_flow_dx    = 0.0; smooth_flow_dy    = 0.0
    else:
        M_sent     = np.array([[1.0, 0.0, float(tx_dx)],
                                [0.0, 1.0, float(tx_dy)]], dtype=np.float32)
        rx_mock_bg = cv2.warpAffine(rx_mock_bg, M_sent, (new_w, new_h))
        rx_mock_bg[crop_y:crop_y+ch_c, crop_x:crop_x+cw_c] = crop_img

    total_frames += 1

    # ---- 終端機輸出 ----
    saving_pct = (1.0 - v4_size / v3_size) * 100 if v3_size > 0 else 0
    area_saving = (1.0 - (cw_c*ch_c)/(new_w*new_h))*100 if not send_full else 0
    print(f"Frame {total_frames:4d} | Level {current_level} | "
          f"V3(Full): {v3_size:6,} B | "
          f"V4({'FULL' if send_full else 'BBOX'}): {v4_size:6,} B | "
          f"Saving: {saving_pct:+.1f}% | "
          f"Thresh: {adaptive_thresh}")

    # ---- UI 顯示 ----
    disp_v3 = frame_resized.copy()
    cv2.putText(disp_v3, f"V3 FULL | Level {current_level}",
                (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(disp_v3, f"Size: {v3_size:,} B",
                (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(disp_v3, f"ratio={ratio} | quality={quality}",
                (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)

    disp_v4 = frame_resized.copy()
    if pkt_type == 1:
        cv2.rectangle(disp_v4, (crop_x, crop_y), (crop_x+cw_c, crop_y+ch_c), (0,0,255), 2)
        cv2.putText(disp_v4, f"V4 BBOX (NoSky) | Saved {area_saving:.0f}%",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    else:
        cv2.putText(disp_v4, f"V4 FULL (NoSky) | Level {current_level}",
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(disp_v4, f"Size: {v4_size:,} B",
                (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,255), 2)
    cv2.putText(disp_v4, f"Thresh: {adaptive_thresh}",
                (5, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,220), 1)

    combined = np.hstack([disp_v3, disp_v4])
    cv2.putText(combined, "V3: Full Frame (left)   |   V4: Motion Diff No SkyMask (right)",
                (5, combined.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.imshow("Motion_experiment_noSkyMask | q=quit  f=force-full  +/-=level", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key in (ord('+'), ord('=')):
        if current_level < 4:
            current_level += 1; force_full_frame = True
            print(f">>> Switched to Level {current_level}")
    elif key in (ord('-'), ord('_')):
        if current_level > 0:
            current_level -= 1; force_full_frame = True
            print(f">>> Switched to Level {current_level}")
    elif key == ord('f'):
        force_full_frame = True

cap.release()
cv2.destroyAllWindows()

# ============================================================
# 統計與圖表
# ============================================================
max_len = max(len(v3_frame_sizes), len(v4_frame_sizes))
v3_frame_sizes += [0] * (max_len - len(v3_frame_sizes))
v4_frame_sizes += [0] * (max_len - len(v4_frame_sizes))

v3_total     = sum(v3_frame_sizes)
v4_total     = sum(v4_frame_sizes)
total_saving = (1 - v4_total / v3_total) * 100 if v3_total > 0 else 0

print("\n" + "="*60)
print(f"{'Experiment Results (No Sky Mask)':^60}")
print("="*60)
print(f"  Video  : {args.video}")
print(f"  Level  : {current_level}  |  Ratio: {settings[current_level][0]}  |  Quality: {settings[current_level][1]}")
print(f"  Frames : {total_frames} frames")
print(f"  V3 Full Frame        -> Total: {v3_total:,} bytes  ({v3_total/1024:.1f} KB)")
print(f"  V4 Motion Diff(NoSky)-> Total: {v4_total:,} bytes  ({v4_total/1024:.1f} KB)")
print(f"  Saved                : {v3_total - v4_total:,} bytes ({total_saving:.1f}%)")
print("="*60 + "\n")

frames_axis = list(range(1, max_len + 1))
fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    f"AVIF Benchmark (No Sky Mask)  |  Video: {args.video}  |  Level {current_level}  "
    f"(ratio={settings[current_level][0]}, quality={settings[current_level][1]})",
    fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(frames_axis, v3_frame_sizes, color='tomato', linewidth=0.8, label='V3 Full Frame')
ax1.set_title('V3: Full Frame AVIF - Size per Frame', fontsize=11)
ax1.set_xlabel('Frame Number'); ax1.set_ylabel('Bytes')
ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)
ax1.grid(True, linestyle='--', alpha=0.6); ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(frames_axis, v4_frame_sizes, color='steelblue', linewidth=0.8, label='V4 Motion Diff (No SkyMask)')
ax2.set_title('V4: Motion Diff (No Sky Mask) - Size per Frame', fontsize=11)
ax2.set_xlabel('Frame Number'); ax2.set_ylabel('Bytes')
ax2.set_xlim(left=0); ax2.set_ylim(bottom=0)
ax2.grid(True, linestyle='--', alpha=0.6); ax2.legend()

ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(frames_axis, v3_frame_sizes, color='tomato',    linewidth=0.8, alpha=0.8, label='V3 Full Frame')
ax3.plot(frames_axis, v4_frame_sizes, color='steelblue', linewidth=0.8, alpha=0.8, label='V4 Motion Diff (No SkyMask)')
ax3.set_title('V3 vs V4 (No Sky Mask) - Size per Frame Comparison', fontsize=11)
ax3.set_xlabel('Frame Number'); ax3.set_ylabel('Bytes')
ax3.set_xlim(left=0); ax3.set_ylim(bottom=0)
ax3.grid(True, linestyle='--', alpha=0.6); ax3.legend()

ax4 = fig.add_subplot(gs[1, 1])
bars = ax4.bar(['V3\nFull Frame', 'V4\nMotion Diff\n(No SkyMask)'],
               [v3_total/1024, v4_total/1024],
               color=['tomato', 'steelblue'], width=0.5, edgecolor='white')
for bar, val in zip(bars, [v3_total, v4_total]):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{val/1024:.1f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax4.set_title('Total Data Size Comparison', fontsize=11)
ax4.set_ylabel('Total Size (KB)'); ax4.set_ylim(bottom=0)
ax4.grid(True, axis='y', linestyle='--', alpha=0.6)
ax4.text(0.5, 0.92,
         f"V4 saves {total_saving:.1f}%  ({(v3_total-v4_total)/1024:.1f} KB)",
         transform=ax4.transAxes, ha='center', fontsize=11,
         color='green' if total_saving > 0 else 'red', fontweight='bold')

out_img = "motion_experiment_noSkyMask_result.png"
plt.savefig(out_img, dpi=150, bbox_inches='tight')
print(f"Chart saved as {out_img}")
plt.show()
