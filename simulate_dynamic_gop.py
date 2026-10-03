import numpy as np
import matplotlib.pyplot as plt
import os
import json

CHUNK_SIZE = 812  # bytes per packet
BAUD_RATE_BPS = 921600
MAX_THROUGHPUT_BYTES_PER_SEC = BAUD_RATE_BPS / 10.0  # 92,160 B/s
MAX_PACKETS_PER_SEC = MAX_THROUGHPUT_BYTES_PER_SEC / CHUNK_SIZE  # ~113.5 pkts/s
MAX_VIDEO_FPS = 30.0

# H.265 frame sizes (CRF 37)
H265_I_SIZE = 3240  # bytes -> 4 packets
H265_P_SIZE = 300   # bytes -> 1 packet

PKTS_I = int(np.ceil(H265_I_SIZE / CHUNK_SIZE))  # 4
PKTS_P = int(np.ceil(H265_P_SIZE / CHUNK_SIZE))  # 1

def get_channel_metrics(d):
    """
    Log-distance path loss model for packet success rate p(d).
    d in kilometers (0.05 km to 4.0 km)
    """
    rssi = -50.0 - 10.0 * np.log2(d)
    # p = 0.95 at 1.0 km (rssi = -50). Scales down with distance.
    p = 0.65 + (rssi + 60.0) * 0.03
    p = float(np.clip(p, 0.001, 1.0))
    return rssi, p

def calculate_effective_fps(gop_size, p_succ):
    """
    Calculate effective valid FPS for a given GOP size and packet success probability.
    """
    avg_pkts = (PKTS_I + (gop_size - 1) * PKTS_P) / gop_size
    tx_fps = min(MAX_VIDEO_FPS, MAX_PACKETS_PER_SEC / avg_pkts)
    
    # Frame k (0 to gop_size - 1) requires I-frame and 1..k P-frames: cum_pkts = PKTS_I + k * PKTS_P
    cum_pkts = PKTS_I + np.arange(gop_size) * PKTS_P
    frame_survival = p_succ ** cum_pkts
    
    avg_survival = np.mean(frame_survival)
    effective_fps = tx_fps * avg_survival
    return effective_fps

# Dense distance grid for smooth curves & precise threshold search
distances_dense = np.linspace(0.05, 4.0, 1000)
gops = np.arange(1, 61)  # 1 to 60

# Matrix of shape (len(gops), len(distances_dense))
fps_matrix = np.zeros((len(gops), len(distances_dense)))

for g_idx, g in enumerate(gops):
    for d_idx, d in enumerate(distances_dense):
        _, p_succ = get_channel_metrics(d)
        fps_matrix[g_idx, d_idx] = calculate_effective_fps(g, p_succ)

# Calculate optimal GOP and compositional envelope curve
optimal_gop_indices = np.argmax(fps_matrix, axis=0)
optimal_gops = gops[optimal_gop_indices]
envelope_fps = np.max(fps_matrix, axis=0)

# Find exact switching point between GOPs
lut_entries = []
current_gop = optimal_gops[0]
start_dist = distances_dense[0]

for i in range(1, len(distances_dense)):
    if optimal_gops[i] != current_gop:
        end_dist = distances_dense[i-1]
        mid_dist = (start_dist + end_dist) / 2.0
        _, mid_p = get_channel_metrics(mid_dist)
        mid_fps = calculate_effective_fps(current_gop, mid_p)
        lut_entries.append({
            'start_km': round(start_dist, 3),
            'end_km': round(end_dist, 3),
            'optimal_gop': int(current_gop),
            'avg_fps_at_mid': round(mid_fps, 2)
        })
        start_dist = distances_dense[i]
        current_gop = optimal_gops[i]

# Final entry
end_dist = distances_dense[-1]
mid_dist = (start_dist + end_dist) / 2.0
_, mid_p = get_channel_metrics(mid_dist)
mid_fps = calculate_effective_fps(current_gop, mid_p)
lut_entries.append({
    'start_km': round(start_dist, 3),
    'end_km': round(end_dist, 3),
    'optimal_gop': int(current_gop),
    'avg_fps_at_mid': round(mid_fps, 2)
})

print("=== DYNAMIC GOP LOOKUP TABLE (LUT) ===")
print(f"{'Distance Range (km)':<22} | {'Optimal GOP':<12} | {'Sample Effective FPS':<20}")
print("-" * 60)
for entry in lut_entries:
    dist_str = f"{entry['start_km']:.2f} km - {entry['end_km']:.2f} km"
    print(f"{dist_str:<22} | {entry['optimal_gop']:<12} | {entry['avg_fps_at_mid']:<20.2f}")

# Discrete standard evaluation points for tabular comparison
eval_dists = [0.1, 0.5, 0.8, 1.0, 1.15, 1.5, 2.0, 2.5, 3.0, 4.0]
print("\n=== DETAILED MULTI-GOP COMPARISON AT KEY DISTANCES ===")
print(f"{'Dist (km)':<10} | {'Opt GOP':<8} | {'Dynamic FPS':<12} | {'GOP 60':<9} | {'GOP 30':<9} | {'GOP 10':<9} | {'GOP 2':<9} | {'GOP 1':<9}")
print("-" * 90)
for d in eval_dists:
    _, p = get_channel_metrics(d)
    fps_all = [calculate_effective_fps(g, p) for g in gops]
    opt_g = gops[np.argmax(fps_all)]
    dyn_fps = max(fps_all)
    g60_fps = calculate_effective_fps(60, p)
    g30_fps = calculate_effective_fps(30, p)
    g10_fps = calculate_effective_fps(10, p)
    g2_fps  = calculate_effective_fps(2, p)
    g1_fps  = calculate_effective_fps(1, p)
    print(f"{d:<10.2f} | {opt_g:<8} | {dyn_fps:<12.2f} | {g60_fps:<9.2f} | {g30_fps:<9.2f} | {g10_fps:<9.2f} | {g2_fps:<9.2f} | {g1_fps:<9.2f}")

# Save LUT to JSON
with open('dynamic_gop_lut.json', 'w') as f:
    json.dump(lut_entries, f, indent=4)

# --- PLOTTING ---
plt.figure(figsize=(12, 7), dpi=300)

# Colors and key GOP highlighting
cmap = plt.colormaps['plasma']
featured_gops = [1, 2, 5, 10, 20, 30, 60]

for g_idx, g in enumerate(gops):
    if g in featured_gops:
        continue
    plt.plot(distances_dense, fps_matrix[g_idx], color='lightgray', alpha=0.3, linewidth=0.6)

# Plot key GOP lines with distinction
colors = {1: '#d62728', 2: '#e377c2', 5: '#9467bd', 10: '#8c564b', 20: '#1f77b4', 30: '#2ca02c', 60: '#ff7f0e'}
for g in featured_gops:
    g_idx = g - 1
    plt.plot(distances_dense, fps_matrix[g_idx], color=colors[g], alpha=0.75, linewidth=1.5, linestyle='--', label=f'GOP {g}')

# Highlight Top-Most Compositional Envelope (Dynamic GOP)
plt.plot(distances_dense, envelope_fps, color='#00ffcc', linewidth=3.5, linestyle='-', label='Dynamic GOP Envelope (Optimal)', zorder=10)

# Customize plot aesthetics
plt.title('Dynamic GOP Optimization: Upper Compositional Envelope (GOP 1 to 60)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Distance (km)', fontsize=12, labelpad=8)
plt.ylabel('Effective Delivered FPS (Log Scale)', fontsize=12, labelpad=8)
plt.yscale('log')
plt.ylim(0.01, 35.0)
plt.xlim(0.05, 4.0)

# Add grid lines
plt.grid(True, which="both", linestyle="--", alpha=0.5)

# Switch threshold annotation
plt.axvline(x=1.14, color='cyan', linestyle=':', linewidth=2, label='Optimal Switch Point (~1.14 km)')
plt.text(0.3, 20, 'Short Range: GOP = 2\n(30 FPS Max, Low Risk)', fontsize=10, color='darkgreen', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
plt.text(1.5, 0.5, 'Long Range: GOP = 1\n(All-Intra Immunity)', fontsize=10, color='darkred', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

# Legend
plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

plot_path = os.path.abspath('dynamic_gop_envelope.png')
plt.tight_layout()
plt.savefig(plot_path)
print(f"\nPlot successfully generated and saved to: {plot_path}")
