import cv2
import numpy as np
from PIL import Image
import pillow_avif
import io
import os
import subprocess
import imageio_ffmpeg
import json
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

VIDEO_PATH = 'g:/code/EE_project/filghtRecord/2.mp4'
CHUNK_SIZE = 812  # bytes per packet
GOP_SIZE = 60
AVIF_QUALITY = 30
H265_CRF = 37  # Similar SSIM (~0.68) to AVIF Q30
BAUD_RATE_BPS = 921600  # Baud rate from Tx_AVIF_v5_Motion_Camera.py
# Effective throughput in Bytes/sec (10 bits per byte start/stop framing)
MAX_THROUGHPUT_BYTES_PER_SEC = BAUD_RATE_BPS / 10.0  # 92,160 Bytes/sec
MAX_PACKETS_PER_SEC = MAX_THROUGHPUT_BYTES_PER_SEC / CHUNK_SIZE  # ~113.5 packets/sec

# 1. Load video frames
print(f"Loading video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
FPS = fps if fps > 0 else 30.0

if orig_w > 640:
    scale = 640.0 / orig_w
    new_w = 640
    new_h = int(orig_h * scale)
else:
    new_w = orig_w
    new_h = orig_h

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (new_w, new_h))
    frames.append(frame)
cap.release()

num_frames = len(frames)
print(f"Loaded {num_frames} frames @ {FPS:.2f} FPS, Resolution: {new_w}x{new_h}")

# Save raw video for H265 GOP=60 encoding
raw_video_path = 'raw_resized.mp4'
out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (new_w, new_h))
for f in frames:
    out.write(f)
out.release()

# 2. Encode AVIF frame-by-frame
print(f"Encoding raw AVIF (Quality {AVIF_QUALITY})...")
avif_sizes = []
avif_packets_per_frame = []
for f in frames:
    frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="AVIF", quality=AVIF_QUALITY, speed=10)
    size = len(buf.getvalue())
    avif_sizes.append(size)
    pkts = int(np.ceil(size / CHUNK_SIZE))
    avif_packets_per_frame.append(pkts)

avg_avif_size = np.mean(avif_sizes)
avg_avif_pkts = np.mean(avif_packets_per_frame)
print(f"AVIF Avg Frame Size: {avg_avif_size:.0f} bytes ({avg_avif_pkts:.2f} packets)")

# 3. Encode H.265 GOP=60
h265_out_path = 'h265_gop60.mp4'
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [
    ffmpeg_exe, '-y', '-i', raw_video_path,
    '-c:v', 'libx265',
    '-x265-params', f'keyint={GOP_SIZE}:min-keyint={GOP_SIZE}',
    '-crf', str(H265_CRF),
    '-preset', 'fast',
    h265_out_path
]
subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

h265_size = os.path.getsize(h265_out_path)
avg_h265_size = h265_size / num_frames

# Estimate I-frame vs P-frame sizes for GOP=60
# Size = (1/60)*S_I + (59/60)*S_P. S_I ~ 10 * S_P
# avg = (10/60 + 59/60) * S_P = 1.15 * S_P
sp_est = avg_h265_size / 1.15
si_est = sp_est * 10.0

p_pkts = max(1, int(np.ceil(sp_est / CHUNK_SIZE)))
i_pkts = max(1, int(np.ceil(si_est / CHUNK_SIZE)))

h265_frame_pkts = []
for i in range(num_frames):
    if i % GOP_SIZE == 0:
        h265_frame_pkts.append(i_pkts)
    else:
        h265_frame_pkts.append(p_pkts)

print(f"H.265 Avg Frame Size: {avg_h265_size:.0f} bytes (I-frame: {i_pkts} pkts, P-frame: {p_pkts} pkts)")

# 4. Extended Distance bounds: 0.05 km (50m) to 6.0 km
distances = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])

def get_channel_metrics(d):
    """
    Log-distance path loss model:
    RSSI(d) = -50 - 10*log2(d)
    Packet success rate p mapped realistically:
    At 0.05 km (50m): -37 dBm -> 100.0%
    At 0.1 km (100m): -40 dBm -> 99.9%
    At 0.2 km (200m): -43 dBm -> 99.5%
    At 0.5 km (500m): -50 dBm -> 98.5%
    At 1.0 km: -50 dBm -> 95.0%
    At 2.0 km: -60 dBm -> 65.0%
    At 4.0 km: -70 dBm -> 35.0%
    At 8.0 km: -80 dBm -> 5.0%
    """
    if d <= 0.05:
        p = 1.0
        rssi = -37.0
    elif d <= 0.1:
        p = 0.999
        rssi = -40.0
    elif d <= 0.2:
        p = 0.995
        rssi = -43.0
    elif d <= 0.5:
        p = 0.985
        rssi = -47.0
    else:
        rssi = -50.0 - 10.0 * np.log2(d)
        p = 0.65 + (rssi + 60.0) * 0.03
    p = float(np.clip(p, 0.001, 1.0))
    return rssi, p

results = []

for d in distances:
    rssi, p_success = get_channel_metrics(d)
    
    # --- A. Loss-limited Valid FPS (Source at 30 FPS) ---
    # AVIF
    avif_valid_frames = sum([p_success ** k for k in avif_packets_per_frame])
    avif_loss_fps = (avif_valid_frames / num_frames) * FPS
    
    # H.265 (GOP=60)
    h265_valid_frames = 0
    num_gops = int(np.ceil(num_frames / GOP_SIZE))
    for g in range(num_gops):
        g_start = g * GOP_SIZE
        g_frames = h265_frame_pkts[g_start : min(g_start + GOP_SIZE, num_frames)]
        cum_pkts = 0
        for fk in g_frames:
            cum_pkts += fk
            h265_valid_frames += (p_success ** cum_pkts)
    h265_loss_fps = (h265_valid_frames / num_frames) * FPS
    
    # --- Final Delivered FPS (Bandwidth + Loss combined) ---
    avg_h265_pkts = avg_h265_size / CHUNK_SIZE
    # Max FPS we can transmit bounded by Source (30) and Serial Link
    avif_tx_fps = min(FPS, MAX_PACKETS_PER_SEC / avg_avif_pkts)
    h265_tx_fps = min(FPS, MAX_PACKETS_PER_SEC / avg_h265_pkts)
    
    # Survival rate (fraction of transmitted frames that are validly decoded)
    avif_survival_rate = avif_valid_frames / num_frames
    h265_survival_rate = h265_valid_frames / num_frames
    
    # Final delivered effective valid FPS
    avif_final_fps = avif_tx_fps * avif_survival_rate
    h265_final_fps = h265_tx_fps * h265_survival_rate
    
    results.append({
        'distance_km': d,
        'rssi_dbm': round(rssi, 1),
        'packet_success_rate': round(p_success * 100, 2),
        'avif_valid_fps': round(avif_final_fps, 2),
        'h265_valid_fps': round(h265_final_fps, 2),
        'avif_bw_limited_fps': round(avif_final_fps, 2), # Keeping keys for compatibility
        'h265_bw_limited_fps': round(h265_final_fps, 2),
        'best_choice': 'H.265' if h265_final_fps > avif_final_fps else 'AVIF'
    })

print("\n=== EXPANDED DISTANCE SIMULATION (0.05 km to 8.0 km) ===")
print(f"{'Dist (km)':<10} | {'RSSI':<8} | {'Pkt Succ':<10} | {'AVIF Valid FPS':<15} | {'H265 Valid FPS':<15} | {'AVIF BW FPS':<12} | {'H265 BW FPS':<12} | {'Winner':<8}")
print("-" * 105)
for r in results:
    print(f"{r['distance_km']:<10} | {r['rssi_dbm']:<8} | {r['packet_success_rate']:5.1f}%     | {r['avif_valid_fps']:<15} | {r['h265_valid_fps']:<15} | {r['avif_bw_limited_fps']:<12} | {r['h265_bw_limited_fps']:<12} | {r['best_choice']:<8}")

with open('expanded_distance_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# 5. Generate Graph
dists = [r['distance_km'] for r in results]
avif_fps = [r['avif_bw_limited_fps'] for r in results]
h265_fps = [r['h265_bw_limited_fps'] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(dists, avif_fps, marker='o', linestyle='-', color='blue', label='AVIF (Raw)')
plt.plot(dists, h265_fps, marker='s', linestyle='--', color='red', label='H.265 (GOP 60)')

plt.title('Effective Valid FPS vs. Distance (921,600 Baud Limit)')
plt.xlabel('Distance (km)')
plt.ylabel('Effective Delivered FPS (Log Scale)')
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.axvline(x=0.8, color='green', linestyle=':', label='Approx Threshold (0.8 km)')
plt.xlim(0, 4.0)


# Save the plot
graph_path = os.path.abspath('fps_vs_distance.png')
plt.savefig(graph_path, dpi=300, bbox_inches='tight')
print(f"Graph saved to {graph_path}")

