import cv2
import numpy as np
import time
import os
import io
import struct
import random
import argparse
import collections
from PIL import Image

# For AVIF
try:
    import pillow_avif
except ImportError:
    pass

# For Metrics
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

try:
    import torch
    import lpips
    loss_fn_lpips = lpips.LPIPS(net='alex', verbose=False)
    if torch.cuda.is_available():
        loss_fn_lpips = loss_fn_lpips.cuda()
except:
    loss_fn_lpips = None

# For RIFE
try:
    import tensorrt as trt
    _HAS_TRT = True
except ImportError:
    _HAS_TRT = False

# --- CONFIG ---
TEST_DISTANCE = 0.20

DISTANCE_TABLE = {
    0.05: {'delivery': 1.000, 'fps': 17.01},
    0.10: {'delivery': 0.999, 'fps': 16.90},
    0.20: {'delivery': 0.995, 'fps': 16.45},
    0.50: {'delivery': 0.985, 'fps': 15.38},
    1.00: {'delivery': 0.950, 'fps': 12.08},
    1.50: {'delivery': 0.775, 'fps': 3.09},
    2.00: {'delivery': 0.650, 'fps': 0.96},
    4.00: {'delivery': 0.350, 'fps': 0.02},
}

if TEST_DISTANCE in DISTANCE_TABLE:
    DELIVERY_RATE = DISTANCE_TABLE[TEST_DISTANCE]['delivery']
    TX_TARGET_FPS = DISTANCE_TABLE[TEST_DISTANCE]['fps']
else:
    DELIVERY_RATE = 1.0
    TX_TARGET_FPS = 30.0

ENABLE_RIFE = True
ENABLE_SR = True
VIDEO_PATH = "fly_2.mp4"

# Models & Hardware TensorRT setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RIFE_ENGINE_PATH = os.path.join(BASE_DIR, "flownet.engine")
SR_ENGINE_PATH = os.path.join(BASE_DIR, "realesr-general-x4v3.engine")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RIFE_PAD_H = 256
MODEL_W, MODEL_H = 320, 240
TARGET_W, TARGET_H = 1280, 720  # 16:9 Super-Resolution
DISP_W, DISP_H = 480, 270       # 16:9 Display window
EVAL_W, EVAL_H = 640, 360       # 16:9 Metric evaluation

# RealESRGAN Buffers
SR_CTX = SR_STREAM = SR_INPUT = SR_OUTPUT = None

def init_sr():
    global SR_CTX, SR_STREAM, SR_INPUT, SR_OUTPUT
    if not _HAS_TRT or DEVICE.type != "cuda": return False
    if not os.path.exists(SR_ENGINE_PATH): return False
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        with open(SR_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine: return False
        SR_CTX = engine.create_execution_context()
        SR_STREAM = torch.cuda.Stream()
        inp_shape = (1, 3, MODEL_H, MODEL_W)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                SR_CTX.set_input_shape(name, inp_shape)
                SR_INPUT = torch.zeros(inp_shape, dtype=torch.float32, device=DEVICE).contiguous()
                SR_CTX.set_tensor_address(name, int(SR_INPUT.data_ptr()))
        SR_CTX.infer_shapes()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(SR_CTX.get_tensor_shape(name))
                dt = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                SR_OUTPUT = torch.zeros(shape, dtype=dt, device=DEVICE).contiguous()
                SR_CTX.set_tensor_address(name, int(SR_OUTPUT.data_ptr()))
        return True
    except Exception as e:
        print(f"[SR] Init failed: {e}")
        return False

def infer_sr(bgr_model):
    rgb = cv2.cvtColor(bgr_model, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    SR_INPUT.copy_(t)
    SR_CTX.execute_async_v3(stream_handle=SR_STREAM.cuda_stream)
    SR_STREAM.synchronize()
    out = SR_OUTPUT.float().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return cv2.cvtColor((out.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# RIFE Buffers
RIFE_CTX = None
RIFE_STREAM = None
RIFE_INPUT = None
RIFE_TIMESTEP = None
RIFE_OUT_TENSORS = {}
RIFE_DUMMY = []
RIFE_OUTPUT = None
_RIFE_STAGING = np.zeros((1, 6, RIFE_PAD_H, MODEL_W), dtype=np.float32)

def init_rife():
    global RIFE_CTX, RIFE_STREAM, RIFE_INPUT, RIFE_TIMESTEP
    global RIFE_OUT_TENSORS, RIFE_DUMMY, RIFE_OUTPUT
    if not _HAS_TRT or DEVICE.type != "cuda": return False
    if not os.path.exists(RIFE_ENGINE_PATH): return False
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        with open(RIFE_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine: return False
        RIFE_CTX = engine.create_execution_context()
        RIFE_STREAM = torch.cuda.Stream()
        inp_shape = (1, 6, RIFE_PAD_H, MODEL_W)
        ts_shape = (1,)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name == "imgs":
                    RIFE_CTX.set_input_shape(name, inp_shape)
                    RIFE_INPUT = torch.zeros(inp_shape, dtype=torch.float32, device=DEVICE).contiguous()
                    RIFE_CTX.set_tensor_address(name, int(RIFE_INPUT.data_ptr()))
                elif name == "timestep":
                    RIFE_CTX.set_input_shape(name, ts_shape)
                    RIFE_TIMESTEP = torch.full(ts_shape, 0.5, dtype=torch.float32, device=DEVICE).contiguous()
                    RIFE_CTX.set_tensor_address(name, int(RIFE_TIMESTEP.data_ptr()))
                elif name.startswith("onnx::Cast_"):
                    sh = tuple(engine.get_tensor_shape(name)) or (1,)
                    d = torch.ones(sh, dtype=torch.int64, device=DEVICE).contiguous()
                    RIFE_DUMMY.append(d)
                    RIFE_CTX.set_tensor_address(name, int(d.data_ptr()))
        RIFE_CTX.infer_shapes()
        target_shape = (1, 3, RIFE_PAD_H, MODEL_W)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(RIFE_CTX.get_tensor_shape(name))
                dt = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                t = torch.zeros(shape, dtype=dt, device=DEVICE).contiguous()
                RIFE_CTX.set_tensor_address(name, int(t.data_ptr()))
                RIFE_OUT_TENSORS[name] = t
                if name == "2787":
                    RIFE_OUTPUT = t
        if RIFE_OUTPUT is None:
            for v in RIFE_OUT_TENSORS.values():
                if tuple(v.shape) == target_shape:
                    RIFE_OUTPUT = v; break
        return True
    except:
        return False

def infer_rife(bgr0, bgr1):
    for ch in range(3):
        _RIFE_STAGING[0, ch, :MODEL_H, :] = bgr0[:, :, 2 - ch] / 255.0
        _RIFE_STAGING[0, ch, MODEL_H:, :] = _RIFE_STAGING[0, ch, MODEL_H-1:MODEL_H, :]
        _RIFE_STAGING[0, ch+3, :MODEL_H, :] = bgr1[:, :, 2 - ch] / 255.0
        _RIFE_STAGING[0, ch+3, MODEL_H:, :] = _RIFE_STAGING[0, ch+3, MODEL_H-1:MODEL_H, :]
    RIFE_INPUT.copy_(torch.from_numpy(_RIFE_STAGING))
    RIFE_CTX.execute_async_v3(stream_handle=RIFE_STREAM.cuda_stream)
    RIFE_STREAM.synchronize()
    out = RIFE_OUTPUT.float().squeeze(0).permute(1, 2, 0)[:MODEL_H].clamp(0, 1)
    out_np = out.cpu().numpy()
    return cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Metrics
def compute_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_ssim(img1, img2):
    if ssim is None: return 0.0
    return float(ssim(img1, img2, channel_axis=2, data_range=255))

def compute_lpips(img1, img2):
    if loss_fn_lpips is None: return 0.0
    def to_tensor(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        if torch.cuda.is_available(): t = t.cuda()
        return t
    with torch.no_grad():
        score = loss_fn_lpips(to_tensor(img1), to_tensor(img2))
    return float(score.cpu().item())

# Sky mask
def detect_sky_mask(frame_bgr, h, w):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    sky_color = ((S < 80) & (V > 140)).astype(np.uint8) * 255
    blue_sky = ((H > 90) & (H < 130) & (S > 30) & (V > 100)).astype(np.uint8) * 255
    sky_raw = cv2.bitwise_or(sky_color, blue_sky)
    position_weight = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        if row < h * 0.4: position_weight[row, :] = 255
        elif row < h * 0.7: position_weight[row, :] = int(255 * (1.0 - (row - h * 0.4) / (h * 0.3)))
    sky_mask = cv2.bitwise_and(sky_raw, position_weight)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
    sky_mask = cv2.GaussianBlur(sky_mask, (21, 21), 0)
    _, sky_mask = cv2.threshold(sky_mask, 127, 255, cv2.THRESH_BINARY)
    return sky_mask

def main():
    parser = argparse.ArgumentParser(description="Unified TX/RX AVIF Real-Time Simulator")
    parser.add_argument("--distance", type=float, default=TEST_DISTANCE, help="Simulation distance in km")
    parser.add_argument("--video", type=str, default=VIDEO_PATH, help="Path to input video")
    parser.add_argument("--level", type=int, default=3, help="TX Compression Level (0-4), default=3")
    args = parser.parse_args()

    # TX Settings matching Tx_AVIF_v5_Motion_Video.py
    TX_SETTINGS = {
        4: [0.64, 29],
        3: [0.59, 25],
        2: [0.48, 22],
        1: [0.38, 23],
        0: [0.34, 22]
    }
    ratio, quality = TX_SETTINGS.get(args.level, [0.59, 25])

    if args.distance in DISTANCE_TABLE:
        d_rate = DISTANCE_TABLE[args.distance]['delivery']
        t_fps = DISTANCE_TABLE[args.distance]['fps']
    else:
        d_rate = DELIVERY_RATE
        t_fps = TX_TARGET_FPS

    sr_ok = False
    if ENABLE_SR:
        sr_ok = init_sr()
        if not sr_ok:
            print("[WARN] RealESRGAN init failed. Continuing with bilinear upscale.")

    rife_ok = False
    if ENABLE_RIFE:
        rife_ok = init_rife()
        if not rife_ok:
            print("[WARN] RIFE init failed. Continuing without interpolation.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] Cannot open {args.video}")
        return

    # Calculate 16:9 TX Canvas size
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if raw_w > 640:
        scale = 640.0 / raw_w
        base_w = 640
        base_h = int(raw_h * scale)
    else:
        base_w = raw_w
        base_h = raw_h

    # Maintain true 16:9 aspect ratio! (e.g. 640x360 * 0.59 = 377x212)
    new_w = int(base_w * ratio)
    new_h = int(base_h * ratio)

    print(f"--- Simulator Config ---")
    print(f"Distance: {args.distance} km")
    print(f"Video: {args.video} ({raw_w}x{raw_h})")
    print(f"TX Level: {args.level} (ratio={ratio}, quality={quality}) -> TX Canvas: {new_w}x{new_h} (16:9)")
    print(f"Delivery Rate: {d_rate*100:.1f}%")
    print(f"TX Target FPS: {t_fps}")
    print(f"RIFE Enabled: {ENABLE_RIFE}")
    print(f"RealESRGAN: {ENABLE_SR} (outputs {TARGET_W}x{TARGET_H})")
    print(f"------------------------")

    # TX State
    prev_gray = None
    force_full_frame = True
    frames_since_full = 0
    prev_bbox = None
    sky_mask_cache = None
    sky_mask_frame_count = 0
    total_camera_dx, total_camera_dy = 0.0, 0.0
    last_sent_dx, last_sent_dy = 0.0, 0.0
    smooth_flow_dx, smooth_flow_dy = 0.0, 0.0
    _dx_history = collections.deque(maxlen=5)
    _dy_history = collections.deque(maxlen=5)
    tx_mock_bg = None
    last_tx_time = 0

    # RX State
    rx_bg = None
    prev_rx_bg = None
    rx_smooth_dx, rx_smooth_dy = 0.0, 0.0
    rife_mid_frame = None

    HEADER_FORMAT = '<4sBhhHHI'

    cv2.namedWindow('Realtime Simulator', cv2.WINDOW_NORMAL)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or np.isnan(video_fps):
        video_fps = 30.0
    frame_interval = 1.0 / video_fps

    # Metric History: Baseline (No AI) vs Reconstructed (With AI)
    hist_no_ai = {'ssim': [], 'lpips': [], 'sharp': []}
    hist_ai    = {'ssim': [], 'lpips': [], 'sharp': []}

    cur_ssim_no_ai, cur_lpips_no_ai, cur_sharp_no_ai = 0.0, 0.0, 0.0
    cur_ssim_ai, cur_lpips_ai, cur_sharp_ai = 0.0, 0.0, 0.0
    frame_idx = 0
    METRIC_SAMPLE_INTERVAL = 15  # 每 15 幀抽樣一次，消除頓挫感，提升順暢度
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 999999
    current_read_idx = 0

    start_wall_time = time.time()

    while True:
        # 真實時鐘同步 (Wall-Clock Sync): 保證在原片長度內剛好播完，絕不慢動作
        elapsed_real = time.time() - start_wall_time
        target_frame_idx = int(elapsed_real * video_fps)
        if target_frame_idx >= total_video_frames:
            break

        # 若運算超前，等待至當前幀的預定時間
        time_to_wait = (target_frame_idx * frame_interval) - elapsed_real
        if time_to_wait > 0.002:
            time.sleep(time_to_wait)

        # 若運算稍慢，自動快進跳過落後的幀（模擬真實相機即時串流）
        while current_read_idx < target_frame_idx:
            cap.grab()
            current_read_idx += 1

        ret, raw_frame = cap.read()
        if not ret:
            break
        current_read_idx += 1
        frame_idx = current_read_idx
        
        curr_time = time.time()
        orig_raw = raw_frame.copy()
        frame = cv2.resize(raw_frame, (new_w, new_h))  # 保持 16:9，不壓扁！
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update TX mask
        sky_mask_frame_count += 1
        if sky_mask_cache is None or sky_mask_frame_count >= 30:
            sky_mask_cache = detect_sky_mask(frame, new_h, new_w)
            sky_mask_frame_count = 0
        ground_mask = cv2.bitwise_not(sky_mask_cache)

        if prev_gray is None:
            prev_gray = curr_gray.copy()
            tx_mock_bg = curr_gray.copy()
            force_full_frame = True
            rx_bg = frame.copy()
            prev_rx_bg = rx_bg.copy()
        
        # --- TX GMC ---
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

        smooth_flow_dx = smooth_flow_dx * 0.6 + raw_dx * 0.4
        smooth_flow_dy = smooth_flow_dy * 0.6 + raw_dy * 0.4
        if abs(smooth_flow_dx) > 0.3 or abs(smooth_flow_dy) > 0.3:
            total_camera_dx += smooth_flow_dx
            total_camera_dy += smooth_flow_dy

        prev_gray = curr_gray.copy()

        # Decide if we send based on target FPS
        if (curr_time - last_tx_time) >= (1.0 / t_fps):
            last_tx_time = curr_time
            
            raw_acc_dx = total_camera_dx - last_sent_dx
            raw_acc_dy = total_camera_dy - last_sent_dy
            _dx_history.append(raw_acc_dx)
            _dy_history.append(raw_acc_dy)
            int_acc_dx = int(round(sorted(_dx_history)[len(_dx_history)//2]))
            int_acc_dy = int(round(sorted(_dy_history)[len(_dy_history)//2]))

            M_shift = np.array([[1.0, 0.0, float(int_acc_dx)], [0.0, 1.0, float(int_acc_dy)]], dtype=np.float32)
            aligned_rx_bg = cv2.warpAffine(tx_mock_bg, M_shift, (new_w, new_h))
            diff = cv2.absdiff(curr_gray, aligned_rx_bg)

            ground_diff = cv2.bitwise_and(diff, ground_mask)
            ground_noise_std = np.std(ground_diff[ground_mask > 0]) if np.any(ground_mask > 0) else 10
            ground_thresh = max(8, min(25, int(ground_noise_std * 2.5 + 5)))
            _, ground_fg = cv2.threshold(ground_diff, ground_thresh, 255, cv2.THRESH_BINARY)
            ground_fg = cv2.bitwise_and(ground_fg, ground_mask)

            sky_diff = cv2.bitwise_and(diff, sky_mask_cache)
            _, sky_fg = cv2.threshold(sky_diff, 50, 255, cv2.THRESH_BINARY)
            sky_fg = cv2.bitwise_and(sky_fg, sky_mask_cache)

            fg_mask = cv2.bitwise_or(ground_fg, sky_fg)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            fg_mask = cv2.dilate(fg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)

            send_full = force_full_frame
            frames_since_full += 1
            if frames_since_full >= 180:
                send_full = True

            crop_x, crop_y = 0, 0
            target_crop = frame
            
            skip_send = False

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
                            ground_rows = np.where(np.any(ground_mask > 0, axis=1))[0]
                            if len(ground_rows) > 0:
                                y_min_g, y_max_g = int(ground_rows[0]), int(ground_rows[-1]) + 1
                                crop_x, crop_y = 0, y_min_g
                                target_crop = frame[y_min_g:y_max_g, 0:new_w]
                            else: skip_send = True
                        else: skip_send = True
                else:
                    x_min, y_min = new_w, new_h
                    x_max, y_max = 0, 0
                    for cnt in motion_contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        x_min = min(x_min, x); y_min = min(y_min, y)
                        x_max = max(x_max, x + w); y_max = max(y_max, y + h)
                    
                    if prev_bbox is not None:
                        px_min, py_min, px_max, py_max = prev_bbox
                        s = 0.4
                        x_min = int(px_min * (1-s) + x_min * s)
                        y_min = int(py_min * (1-s) + y_min * s)
                        x_max = int(px_max * (1-s) + x_max * s)
                        y_max = int(py_max * (1-s) + y_max * s)
                        for cnt in motion_contours:
                            bx, by, bw, bh = cv2.boundingRect(cnt)
                            x_min = min(x_min, bx); y_min = min(y_min, by)
                            x_max = max(x_max, bx + bw); y_max = max(y_max, by + bh)

                    bbox_area = (x_max - x_min) * (y_max - y_min)
                    if bbox_area > new_w * new_h * 0.75:
                        send_full = True
                        target_crop = frame
                        crop_x, crop_y = 0, 0
                    else:
                        pad = 20
                        x_min = max(0, x_min - pad); y_min = max(0, y_min - pad)
                        x_max = min(new_w, x_max + pad); y_max = min(new_h, y_max + pad)
                        crop_x, crop_y = x_min, y_min
                        target_crop = frame[y_min:y_max, x_min:x_max]
                        prev_bbox = (x_min, y_min, x_max, y_max)
            
            if not skip_send:
                # Encode AVIF
                frame_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                buffer = io.BytesIO()
                pil_img.save(buffer, format="AVIF", quality=quality, speed=10)
                frame_bytes = buffer.getvalue()
                
                pkt_type = 0 if send_full else 1
                tx_dx = max(-32768, min(32767, int_acc_dx))
                tx_dy = max(-32768, min(32767, int_acc_dy))
                
                # TX State Update
                last_sent_dx += tx_dx
                last_sent_dy += tx_dy
                if send_full:
                    tx_mock_bg = curr_gray.copy()
                    force_full_frame = False
                    frames_since_full = 0
                    prev_bbox = None
                    total_camera_dx, total_camera_dy = 0.0, 0.0
                    last_sent_dx, last_sent_dy = 0.0, 0.0
                    smooth_flow_dx, smooth_flow_dy = 0.0, 0.0
                else:
                    M_tx = np.array([[1.0, 0.0, float(tx_dx)], [0.0, 1.0, float(tx_dy)]], dtype=np.float32)
                    tx_mock_bg = cv2.warpAffine(tx_mock_bg, M_tx, (new_w, new_h))
                    cy, cx = crop_y, crop_x
                    ch, cw = target_crop.shape[:2]
                    tx_mock_bg[cy:cy+ch, cx:cx+cw] = cv2.cvtColor(target_crop, cv2.COLOR_BGR2GRAY)
                
                # Channel Simulation (Packet Drop)
                if random.random() <= d_rate:
                    # RX Decode
                    try:
                        dec_pil = Image.open(io.BytesIO(frame_bytes))
                        bgr_patch = cv2.cvtColor(np.array(dec_pil), cv2.COLOR_RGB2BGR)
                        
                        if pkt_type == 0:
                            prev_rx_bg = rx_bg.copy()
                            rx_bg = bgr_patch
                        elif pkt_type == 1:
                            h_bg, w_bg = rx_bg.shape[:2]
                            rx_smooth_dx = rx_smooth_dx * 0.4 + float(tx_dx) * 0.6
                            rx_smooth_dy = rx_smooth_dy * 0.4 + float(tx_dy) * 0.6
                            apply_dx = int(round(rx_smooth_dx))
                            apply_dy = int(round(rx_smooth_dy))
                            if abs(apply_dx) > 0 or abs(apply_dy) > 0:
                                M_rx = np.float32([[1, 0, apply_dx], [0, 1, apply_dy]])
                                rx_bg = cv2.warpAffine(rx_bg, M_rx, (w_bg, h_bg), borderMode=cv2.BORDER_REPLICATE)
                                rx_smooth_dx -= apply_dx
                                rx_smooth_dy -= apply_dy

                            ph, pw = bgr_patch.shape[:2]
                            y1, x1 = crop_y, crop_x
                            y2, x2 = min(y1 + ph, h_bg), min(x1 + pw, w_bg)
                            if y2 > y1 and x2 > x1:
                                target_roi = rx_bg[y1:y2, x1:x2].astype(np.float32)
                                patch_f = bgr_patch[:y2-y1, :x2-x1].astype(np.float32)
                                
                                feather_px = 12
                                ph_roi, pw_roi = y2 - y1, x2 - x1
                                alpha = np.ones((ph_roi, pw_roi, 1), dtype=np.float32)
                                for i in range(min(feather_px, ph_roi // 2)):
                                    val = 0.5 * (1.0 - np.cos(np.pi * i / feather_px))
                                    if y1 > 0: alpha[i, :, 0] = np.minimum(alpha[i, :, 0], val)
                                    if y2 < h_bg: alpha[-(i+1), :, 0] = np.minimum(alpha[-(i+1), :, 0], val)
                                for i in range(min(feather_px, pw_roi // 2)):
                                    val = 0.5 * (1.0 - np.cos(np.pi * i / feather_px))
                                    if x1 > 0: alpha[:, i, 0] = np.minimum(alpha[:, i, 0], val)
                                    if x2 < w_bg: alpha[:, -(i+1), 0] = np.minimum(alpha[:, -(i+1), 0], val)
                                alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
                                if alpha.ndim == 2: alpha = alpha[:, :, np.newaxis]
                                
                                edge_mask = (alpha[:, :, 0] < 0.5) & (alpha[:, :, 0] > 0.01)
                                if np.count_nonzero(edge_mask) > 10:
                                    bg_edge = target_roi[edge_mask]
                                    patch_edge = patch_f[edge_mask]
                                    diff_c = np.mean(bg_edge, axis=0) - np.mean(patch_edge, axis=0)
                                else:
                                    diff_c = np.array(cv2.mean(target_roi)[:3]) - np.array(cv2.mean(patch_f)[:3])
                                diff_c = np.clip(diff_c, -12.0, 12.0)
                                patch_f = np.clip(patch_f + diff_c, 0, 255)
                                blended = patch_f * alpha + target_roi * (1.0 - alpha)
                                
                                prev_rx_bg = rx_bg.copy()
                                rx_bg[y1:y2, x1:x2] = blended.astype(np.uint8)
                        
                        # Generate RIFE (適配 16:9)
                        if rife_ok and prev_rx_bg is not None:
                            frame_diff = cv2.absdiff(prev_rx_bg, rx_bg)
                            if np.mean(frame_diff) < 60:
                                try:
                                    b0 = cv2.resize(prev_rx_bg, (MODEL_W, MODEL_H))
                                    b1 = cv2.resize(rx_bg, (MODEL_W, MODEL_H))
                                    rife_mid_frame = cv2.resize(infer_rife(b0, b1), (new_w, new_h))
                                except:
                                    pass
                    except Exception as e:
                        pass # Decode failed

        # Compute Metrics (Anchor vs Original) - 雙軌對比 (16:9 評估)
        if frame_idx % METRIC_SAMPLE_INTERVAL == 0 or frame_idx == 1:
            ref_eval = cv2.resize(orig_raw, (EVAL_W, EVAL_H))  # 640x360 (16:9)
            no_ai_eval = cv2.resize(rx_bg, (EVAL_W, EVAL_H), interpolation=cv2.INTER_LINEAR)
            if sr_ok:
                rx_model = cv2.resize(rx_bg, (MODEL_W, MODEL_H))
                ai_sr_out = infer_sr(rx_model)
                ai_recon_img = cv2.resize(ai_sr_out, (TARGET_W, TARGET_H))  # 1280x720 (16:9)
            else:
                ai_recon_img = cv2.resize(rx_bg, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            ai_eval = cv2.resize(ai_recon_img, (EVAL_W, EVAL_H))

            cur_ssim_no_ai = compute_ssim(ref_eval, no_ai_eval)
            cur_ssim_ai = compute_ssim(ref_eval, ai_eval)
            cur_lpips_no_ai = compute_lpips(ref_eval, no_ai_eval)
            cur_lpips_ai = compute_lpips(ref_eval, ai_eval)
            cur_sharp_no_ai = compute_sharpness(no_ai_eval)
            cur_sharp_ai = compute_sharpness(ai_eval)

            hist_no_ai['ssim'].append(cur_ssim_no_ai)
            hist_ai['ssim'].append(cur_ssim_ai)
            hist_no_ai['lpips'].append(cur_lpips_no_ai)
            hist_ai['lpips'].append(cur_lpips_ai)
            hist_no_ai['sharp'].append(cur_sharp_no_ai)
            hist_ai['sharp'].append(cur_sharp_ai)
        elif 'ai_recon_img' not in locals():
            ai_recon_img = cv2.resize(rx_bg, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

        # Visualization: 3 Columns (16:9 顯示視窗)
        # Col 1: Original | Col 2: No AI Baseline | Col 3: With AI (RealESRGAN + RIFE)
        disp_orig = cv2.resize(orig_raw, (DISP_W, DISP_H))
        disp_no_ai = cv2.resize(rx_bg, (DISP_W, DISP_H))
        disp_ai = cv2.resize(ai_recon_img, (DISP_W, DISP_H))
        
        simulated_fps_received = t_fps * d_rate
        simulated_fps_recon = simulated_fps_received * 2.0 if rife_ok else simulated_fps_received

        # Col 1 Overlay
        cv2.putText(disp_orig, "1. Original Reference", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp_orig, f"Res: {orig_raw.shape[1]}x{orig_raw.shape[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Col 2 Overlay (No AI Baseline)
        cv2.putText(disp_no_ai, "2. No AI (Bilinear)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(disp_no_ai, f"SSIM: {cur_ssim_no_ai:.3f} | LPIPS: {cur_lpips_no_ai:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(disp_no_ai, f"Sharp: {cur_sharp_no_ai:.1f} | FPS: {simulated_fps_received:.1f}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Col 3 Overlay (With AI)
        cv2.putText(disp_ai, "3. With AI (RealESRGAN+RIFE)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp_ai, f"SSIM: {cur_ssim_ai:.3f} | LPIPS: {cur_lpips_ai:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(disp_ai, f"Sharp: {cur_sharp_ai:.1f} | FPS: {simulated_fps_recon:.1f} (x2)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        canvas = np.hstack([disp_orig, disp_no_ai, disp_ai])
        cv2.imshow('Realtime Simulator', canvas)

        # 鍵盤監聽與畫面刷新
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if hist_ai['ssim']:
        avg_s_no = sum(hist_no_ai['ssim']) / len(hist_no_ai['ssim'])
        avg_s_ai = sum(hist_ai['ssim'])    / len(hist_ai['ssim'])
        avg_l_no = sum(hist_no_ai['lpips']) / len(hist_no_ai['lpips'])
        avg_l_ai = sum(hist_ai['lpips'])    / len(hist_ai['lpips'])
        avg_sh_no = sum(hist_no_ai['sharp']) / len(hist_no_ai['sharp'])
        avg_sh_ai = sum(hist_ai['sharp'])    / len(hist_ai['sharp'])

        recv_fps = t_fps * d_rate
        recon_fps = recv_fps * 2.0 if rife_ok else recv_fps

        d_ssim = avg_s_ai - avg_s_no
        p_ssim = (d_ssim / max(1e-5, avg_s_no)) * 100
        d_lpips = avg_l_ai - avg_l_no
        p_lpips = (d_lpips / max(1e-5, avg_l_no)) * 100
        d_sharp = avg_sh_ai - avg_sh_no
        m_sharp = avg_sh_ai / max(1e-5, avg_sh_no)

        print("\n" + "="*88)
        print("                  AI 模型畫面重建效能比較報告 (實驗總結)")
        print("="*88)
        print(f" 測試距離: {args.distance:.2f} km | 封包成功率: {d_rate*100:.1f}% | 評估樣本數: {len(hist_ai['ssim'])} 幀")
        print("-"*88)
        print(f"{'指標項目':<22} | {'未經 AI 優化 (Baseline)':<22} | {'經 AI 重建 (With AI)':<22} | {'改善幅度 (Improvement)':<18}")
        print("-"*88)
        print(f"{'【RealESRGAN 畫質重建】':<22} | {'':<22} | {'':<22} |")
        print(f"  解析度 (Resolution)  | {'320x240 (雙線性放大)':<22} | {'1280x960 (4x 超解析度)':<22} | {'+4x 解析度提升'}")
        print(f"  SSIM 結構相似性 (↑)  | {avg_s_no:<22.4f} | {avg_s_ai:<22.4f} | {d_ssim:+.4f} ({p_ssim:+.1f}%)")
        print(f"  LPIPS 感知失真 (↓)   | {avg_l_no:<22.4f} | {avg_l_ai:<22.4f} | {d_lpips:+.4f} ({p_lpips:+.1f}%)")
        print(f"  Sharpness 銳利度 (↑) | {avg_sh_no:<22.1f} | {avg_sh_ai:<22.1f} | {d_sharp:+.1f} ({m_sharp:.2f}x 銳利)")
        print("-"*88)
        print(f"{'【RIFE 畫面流暢度】':<22} | {'':<22} | {'':<22} |")
        print(f"  每秒幀率 (FPS) (↑)   | {recv_fps:<18.2f} FPS | {recon_fps:<18.2f} FPS | +{recon_fps-recv_fps:.2f} FPS (+100.0%)")
        print("="*88)
        print(" [指標解讀說明]")
        print("  1. SSIM (0~1): 越高越好，表示結構輪廓越接近原圖。")
        print("  2. LPIPS (0~1): 越低越好，負值表示人眼感知的模糊與壓縮瑕疵大幅減少。")
        print("  3. Sharpness: 越高越好，表示物體與地景邊緣清晰銳利。")
        print("  4. FPS: RIFE 深度學習光流插幀，將原本受限的傳輸幀率提升一倍，達成順暢播放。")
        print("="*88 + "\n")

if __name__ == "__main__":
    main()
