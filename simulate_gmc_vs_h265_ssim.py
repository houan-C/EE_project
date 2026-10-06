import cv2
import numpy as np
from PIL import Image
import pillow_avif
import io
import json
import random
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

VIDEO_PATH = 'g:/code/EE_project/filghtRecord/2.mp4'
CHUNK_SIZE = 812
GOP_SIZE = 60
FPS = 30.0
BAUD_RATE_BPS = 921600
MAX_PACKETS_PER_SEC = (BAUD_RATE_BPS / 10.0) / CHUNK_SIZE
SSIM_THRESHOLD = 0.60  # Threshold to consider a frame 'valid'

# --- 1. Extract Frames ---
print("Extracting frames...")
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
orig_w, orig_h = 0, 0
while len(frames) < 180:  # Simulate 180 frames (3 GOPs) to save time, or more? Let's do 180 frames.
    ret, frame = cap.read()
    if not ret: break
    if len(frames) == 0:
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if orig_w > 640:
            scale = 640.0 / orig_w
            new_w, new_h = 640, int(orig_h * scale)
        else:
            new_w, new_h = orig_w, orig_h
    frame = cv2.resize(frame, (new_w, new_h))
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
cap.release()
print(f"Loaded {len(frames)} frames.")

# --- 2. Pre-calculate GMC AVIF Payloads ---
print("Pre-calculating GMC AVIF Payloads...")
gmc_payloads = [] # stores dict with size, dx, dy, crop_x, crop_y, decoded_crop, is_full
prev_gray = None
rx_mock_bg = None
total_camera_dx = 0.0
total_camera_dy = 0.0
last_sent_dx = 0.0
last_sent_dy = 0.0
smooth_flow_dx = 0.0
smooth_flow_dy = 0.0
FLOW_EMA = 0.4
import collections
_dx_history = collections.deque(maxlen=5)
_dy_history = collections.deque(maxlen=5)

sky_mask_cache = np.zeros((new_h, new_w), dtype=np.uint8)
# simplified sky mask (just top 30% as sky for simulation speed)
sky_mask_cache[:int(new_h*0.3), :] = 255
ground_mask = cv2.bitwise_not(sky_mask_cache)

frames_since_full = 0
SKY_REFRESH_INTERVAL = 60 # Set to 60 to match GOP size for fair comparison

for i, curr_gray in enumerate(frames):
    force_full = False
    if prev_gray is None or frames_since_full >= SKY_REFRESH_INTERVAL:
        prev_gray = curr_gray.copy()
        rx_mock_bg = curr_gray.copy()
        force_full = True
        frames_since_full = 0
    
    raw_dx, raw_dy = 0.0, 0.0
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
    
    M_shift = np.array([[1.0, 0.0, float(int_acc_dx)], [0.0, 1.0, float(int_acc_dy)]], dtype=np.float32)
    aligned_rx_bg = cv2.warpAffine(rx_mock_bg, M_shift, (new_w, new_h))
    
    diff = cv2.absdiff(curr_gray, aligned_rx_bg)
    ground_diff = cv2.bitwise_and(diff, ground_mask)
    ground_thresh = 15
    _, ground_fg = cv2.threshold(ground_diff, ground_thresh, 255, cv2.THRESH_BINARY)
    
    sky_diff = cv2.bitwise_and(diff, sky_mask_cache)
    _, sky_fg = cv2.threshold(sky_diff, 50, 255, cv2.THRESH_BINARY)
    
    fg_mask = cv2.bitwise_or(ground_fg, sky_fg)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    fg_mask = cv2.dilate(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    
    send_full = force_full
    if send_full:
        crop_img = curr_gray
        crop_x, crop_y = 0, 0
    else:
        coords = cv2.findNonZero(fg_mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            x, y = max(0, x-10), max(0, y-10)
            w, h = min(new_w-x, w+20), min(new_h-y, h+20)
            crop_img = curr_gray[y:y+h, x:x+w]
            crop_x, crop_y = x, y
        else:
            crop_img = np.zeros((2, 2), dtype=np.uint8)
            crop_x, crop_y = 0, 0

    # Encode to get size and decoded image
    pil_img = Image.fromarray(crop_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='AVIF', quality=30)
    avif_bytes = buf.getvalue()
    size_bytes = len(avif_bytes) + 17 # header
    
    # decode for receiver simulation
    decoded_crop = np.array(Image.open(io.BytesIO(avif_bytes)))
    if len(decoded_crop.shape) == 3:
        decoded_crop = cv2.cvtColor(decoded_crop, cv2.COLOR_RGB2GRAY)
    elif len(decoded_crop.shape) == 2 and decoded_crop.dtype != np.uint8:
        decoded_crop = decoded_crop.astype(np.uint8)

    gmc_payloads.append({
        'is_full': send_full,
        'size': size_bytes,
        'dx': int_acc_dx,
        'dy': int_acc_dy,
        'cx': crop_x,
        'cy': crop_y,
        'crop': decoded_crop
    })
    
    # Update sender rx_mock_bg
    last_sent_dx += int_acc_dx
    last_sent_dy += int_acc_dy
    if send_full:
        rx_mock_bg = decoded_crop.copy()
    else:
        M_sent = np.array([[1.0, 0.0, float(int_acc_dx)], [0.0, 1.0, float(int_acc_dy)]], dtype=np.float32)
        rx_mock_bg = cv2.warpAffine(rx_mock_bg, M_sent, (new_w, new_h))
        cy, cx = crop_y, crop_x
        ch, cw = decoded_crop.shape[:2]
        rx_mock_bg[cy:cy+ch, cx:cx+cw] = decoded_crop
    
    frames_since_full += 1

# Calculate Avg GMC Pkts
gmc_avg_pkts = np.mean([np.ceil(p['size'] / CHUNK_SIZE) for p in gmc_payloads])
print(f"GMC AVIF Avg Pkts per frame: {gmc_avg_pkts:.2f}")

# Calculate Raw AVIF Avg Pkts based ONLY on the full I-frames to be perfectly fair
raw_avif_avg_pkts = np.mean([np.ceil(p['size'] / CHUNK_SIZE) for p in gmc_payloads if p['is_full']])
print(f"Raw AVIF Avg Pkts per frame: {raw_avif_avg_pkts:.2f}")

# H.265 baseline sizes from previous experiment (Average 349 bytes)
h265_i_size = 3240  # ~4 packets
h265_p_size = 300   # 1 packet
h265_avg_pkts = ((h265_i_size/CHUNK_SIZE) + 59*(h265_p_size/CHUNK_SIZE)) / 60
print(f"H.265 Avg Pkts per frame: {h265_avg_pkts:.2f}")

def get_channel_metrics(d):
    if d <= 0.05: return -37.0, 1.0
    elif d <= 0.1: return -40.0, 0.999
    elif d <= 0.2: return -43.0, 0.995
    elif d <= 0.5: return -47.0, 0.985
    else:
        rssi = -50.0 - 10.0 * np.log2(d)
        p = 0.65 + (rssi + 60.0) * 0.03
    return rssi, float(np.clip(p, 0.001, 1.0))

distances = [0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
results = []

for d in distances:
    rssi, p_succ = get_channel_metrics(d)
    
    # 1. Raw AVIF Mathematical Survival
    raw_avif_survival = p_succ ** raw_avif_avg_pkts
    
    # 2. GMC AVIF Mathematical Survival
    # Logic: I-frame must survive. P-frame survives if I-frame survived AND P-frame survives.
    gmc_valid_frames = 0
    current_i_frame_prob = 0
    for st in gmc_payloads:
        pkts = np.ceil(st['size'] / CHUNK_SIZE)
        if st['is_full']:
            current_i_frame_prob = p_succ ** pkts
            gmc_valid_frames += current_i_frame_prob
        else:
            gmc_valid_frames += (current_i_frame_prob * (p_succ ** pkts))
    gmc_survival_rate = gmc_valid_frames / len(gmc_payloads)
    
    # 3. H.265 Mathematical Survival
    h265_valid_frames = 0
    num_gops = int(np.ceil(len(gmc_payloads) / GOP_SIZE))
    for g in range(num_gops):
        for f in range(GOP_SIZE):
            if g * GOP_SIZE + f >= len(gmc_payloads): break
            if f == 0:
                cum_pkts = np.ceil(h265_i_size / CHUNK_SIZE)
            else:
                cum_pkts += np.ceil(h265_p_size / CHUNK_SIZE)
            h265_valid_frames += (p_succ ** cum_pkts)
    h265_survival_rate = h265_valid_frames / len(gmc_payloads)
    
    # 4. H.265 (GOP=1) Mathematical Survival (All-Intra)
    h265_gop1_pkts = np.ceil(h265_i_size / CHUNK_SIZE)
    h265_gop1_survival = p_succ ** h265_gop1_pkts
    
    # Bandwidth limited max FPS
    raw_tx_fps = min(FPS, MAX_PACKETS_PER_SEC / raw_avif_avg_pkts)
    gmc_tx_fps = min(FPS, MAX_PACKETS_PER_SEC / gmc_avg_pkts)
    h265_tx_fps = min(FPS, MAX_PACKETS_PER_SEC / h265_avg_pkts)
    h265_gop1_tx_fps = min(FPS, MAX_PACKETS_PER_SEC / h265_gop1_pkts)
    
    # Final FPS
    raw_final_fps = raw_tx_fps * raw_avif_survival
    gmc_final_fps = gmc_tx_fps * gmc_survival_rate
    h265_final_fps = h265_tx_fps * h265_survival_rate
    h265_gop1_final_fps = h265_gop1_tx_fps * h265_gop1_survival
    
    results.append({
        'dist': d,
        'p': p_succ,
        'raw_fps': raw_final_fps,
        'gmc_fps': gmc_final_fps,
        'h265_fps': h265_final_fps,
        'h265_gop1_fps': h265_gop1_final_fps
    })

print("\n=== RESULTS ===")
print(f"{'Dist':<5} | {'P_Succ':<6} | {'Raw AVIF':<10} | {'GMC FPS':<8} | {'H265 (GOP 60)':<14} | {'H265 (GOP 1)':<14}")
for r in results:
    print(f"{r['dist']:<5} | {r['p']:<6.2f} | {r['raw_fps']:<10.2f} | {r['gmc_fps']:<8.2f} | {r['h265_fps']:<14.2f} | {r['h265_gop1_fps']:<14.2f}")

# Plot
dists = [r['dist'] for r in results]
plt.figure(figsize=(10, 6))
plt.plot(dists, [r['raw_fps'] for r in results], marker='^', color='purple', label='Raw AVIF (No GMC)')
plt.plot(dists, [r['gmc_fps'] for r in results], marker='o', color='blue', label='GMC AVIF')
plt.plot(dists, [r['h265_fps'] for r in results], marker='s', color='red', label='H.265 (GOP 60)')
plt.plot(dists, [r['h265_gop1_fps'] for r in results], marker='D', color='orange', linestyle='-.', label='H.265 (GOP 1)')
plt.title('Effective Valid FPS (Logical Dependency Rule) vs Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Effective FPS')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.axvline(x=0.8, color='green', linestyle=':', label='Approx Threshold (0.8 km)')
plt.savefig('gmc_vs_h265_logical.png')
print("Saved plot to gmc_vs_h265_logical.png")
