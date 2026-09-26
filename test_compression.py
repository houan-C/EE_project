import cv2
import time
import numpy as np
from PIL import Image
import pillow_avif
import io
import os
import struct
import collections
from skimage.metrics import structural_similarity as ssim
import subprocess
import json

# Settings
VIDEO_PATH = 'g:/code/EE_project/filghtRecord/2.mp4'
GOP = 10
AVIF_QUALITIES = [30, 40, 50, 60, 70, 80, 90]
H265_CRFS = [24, 26, 28, 30, 32, 34, 36, 38, 40]

def get_ssim(img1, img2):
    # Convert to grayscale for SSIM
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(g1, g2, data_range=255)

# 1. Extract frames from video and resize
print(f"Reading video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

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

print(f"Loaded {len(frames)} frames. Resolution: {new_w}x{new_h}")

# Save these frames to a raw video for H265 encoding
raw_video_path = 'raw_resized.avi'
out = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*'FFV1'), fps, (new_w, new_h))
for f in frames:
    out.write(f)
out.release()

def test_custom_avif(quality):
    frames_since_full = 0
    prev_gray = None
    rx_mock_bg = None
    rx_color_bg = None
    prev_bbox = None
    BBOX_SMOOTH = 0.4
    
    total_camera_dx = 0.0
    total_camera_dy = 0.0
    last_sent_dx = 0.0
    last_sent_dy = 0.0
    smooth_flow_dx = 0.0
    smooth_flow_dy = 0.0
    FLOW_EMA = 0.4
    
    _dx_history = collections.deque(maxlen=5)
    _dy_history = collections.deque(maxlen=5)
    
    total_size = 0
    ssims = []
    
    for idx, frame in enumerate(frames):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Disabled sky mask
        ground_mask = np.ones((new_h, new_w), dtype=np.uint8) * 255
        
        if prev_gray is None:
            prev_gray = curr_gray.copy()
            rx_mock_bg = curr_gray.copy()
            rx_color_bg = frame.copy()
            force_full_frame = True
        else:
            force_full_frame = False
            
        if frames_since_full >= GOP:
            force_full_frame = True
            
        raw_dx, raw_dy = 0.0, 0.0
        ground_gray_for_features = curr_gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.2, minDistance=7)
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
        
        ground_diff = diff
        if np.any(ground_diff):
            ground_noise_std = np.std(ground_diff)
        else:
            ground_noise_std = 10
        ground_thresh = max(8, min(25, int(ground_noise_std * 2.5 + 5)))
        _, ground_fg = cv2.threshold(ground_diff, ground_thresh, 255, cv2.THRESH_BINARY)
        
        fg_mask = ground_fg
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fg_mask = cv2.dilate(fg_mask, dilate_kernel, iterations=1)
        
        send_full = force_full_frame
        crop_x, crop_y = 0, 0
        target_crop = frame
        
        if not send_full:
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_contours = [c for c in contours if cv2.contourArea(c) > 30]
            
            if not motion_contours:
                if prev_bbox is not None:
                    x_min, y_min, x_max, y_max = prev_bbox
                    crop_x, crop_y = x_min, y_min
                    target_crop = frame[y_min:y_max, x_min:x_max]
                    prev_bbox = None
                else:
                    if frames_since_full % 5 == 0:
                        y_min_g, y_max_g = 0, new_h
                        crop_x, crop_y = 0, y_min_g
                        target_crop = frame[y_min_g:y_max_g, 0:new_w]
                    else:
                        ssims.append(get_ssim(frame, rx_color_bg))
                        frames_since_full += 1
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
                if bbox_area > new_w * new_h * 0.75:
                    send_full = True
                else:
                    pad = 20
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(new_w, x_max + pad)
                    y_max = min(new_h, y_max + pad)
                    crop_x, crop_y = x_min, y_min
                    target_crop = frame[y_min:y_max, x_min:x_max]
                    prev_bbox = (x_min, y_min, x_max, y_max)
                    
        frame_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="AVIF", quality=quality, speed=10)
        frame_bytes = buffer.getvalue()
        
        total_size += len(frame_bytes)
        
        transmitted_pil = Image.open(io.BytesIO(frame_bytes))
        transmitted_cv = cv2.cvtColor(np.array(transmitted_pil), cv2.COLOR_RGB2BGR)
        
        if send_full:
            rx_mock_bg = cv2.cvtColor(transmitted_cv, cv2.COLOR_BGR2GRAY)
            rx_color_bg = transmitted_cv
            last_sent_dx = total_camera_dx
            last_sent_dy = total_camera_dy
            frames_since_full = 0
            prev_bbox = None
        else:
            tx_dx = max(-32768, min(32767, int_acc_dx))
            tx_dy = max(-32768, min(32767, int_acc_dy))
            last_sent_dx += tx_dx
            last_sent_dy += tx_dy
            
            M_sent = np.array([[1.0, 0.0, float(tx_dx)], [0.0, 1.0, float(tx_dy)]], dtype=np.float32)
            rx_mock_bg = cv2.warpAffine(rx_mock_bg, M_sent, (new_w, new_h))
            rx_color_bg = cv2.warpAffine(rx_color_bg, M_sent, (new_w, new_h))
            
            ch, cw = transmitted_cv.shape[:2]
            rx_mock_bg[crop_y:crop_y+ch, crop_x:crop_x+cw] = cv2.cvtColor(transmitted_cv, cv2.COLOR_BGR2GRAY)
            rx_color_bg[crop_y:crop_y+ch, crop_x:crop_x+cw] = transmitted_cv
            frames_since_full += 1
            
        ssims.append(get_ssim(frame, rx_color_bg))
        
    return total_size, np.mean(ssims)

def test_h265(crf):
    output_path = f'h265_crf{crf}.mp4'
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, '-y', '-i', raw_video_path,
        '-c:v', 'libx265',
        '-x265-params', f'keyint={GOP}:min-keyint={GOP}',
        '-crf', str(crf),
        '-preset', 'fast',
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    size = os.path.getsize(output_path)
    
    cap = cv2.VideoCapture(output_path)
    ssims = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx >= len(frames):
            break
        ssims.append(get_ssim(frames[idx], frame))
        idx += 1
    cap.release()
    
    return size, np.mean(ssims)

print("Testing Custom AVIF Method...")
avif_results = []
for q in AVIF_QUALITIES:
    size, avg_ssim = test_custom_avif(q)
    print(f"AVIF Quality {q}: Size = {size} bytes, SSIM = {avg_ssim:.4f}")
    avif_results.append((q, size, avg_ssim))
    
print("Testing H265 Method...")
h265_results = []
for crf in H265_CRFS:
    size, avg_ssim = test_h265(crf)
    print(f"H265 CRF {crf}: Size = {size} bytes, SSIM = {avg_ssim:.4f}")
    h265_results.append((crf, size, avg_ssim))
    
with open('compression_results.json', 'w') as f:
    json.dump({'avif': avif_results, 'h265': h265_results}, f, indent=4)
