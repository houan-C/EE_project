import cv2
import serial
import numpy as np
import time
import os
import warnings
import sys
import threading
import queue
import io
import collections
import struct

warnings.filterwarnings("ignore")

try:
    from PIL import Image
    import pillow_avif
except ImportError:
    pass

try:
    import tensorrt as trt
    import torch 
except ImportError:
    pass

COM_PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"

BAUD_RATE = 921600
CHUNK_SIZE = 1024 
TIMEOUT = 0.01

ENGINE_INPUT_W, ENGINE_INPUT_H = 320, 240
DISPLAY_SCALE = 0.5 

TRT_LOGGER = None
if 'trt' in globals():
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

DEVICE = None
if 'torch' in globals():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_FILENAME = "realesr-general-x4v3.engine" 
ENGINE_PATH = os.path.join(BASE_DIR, ENGINE_FILENAME)

TRT_STREAM = None
TRT_INPUT_TENSOR = None
TRT_OUTPUT_TENSOR = None
TRT_CONTEXT = None

frame_queue = queue.Queue(maxsize=1)
sliding_window = collections.deque()
display_queue = []
display_lock = threading.Lock()
running = True
right_count = 0
wrong_count = 0
dropped_frames_since_last_success = 0

super_res_enabled = True

def init_trt():
    global TRT_CONTEXT, TRT_STREAM, TRT_INPUT_TENSOR, TRT_OUTPUT_TENSOR
    if DEVICE is None or DEVICE.type != 'cuda':
        return False
    engine_file = ENGINE_PATH
    if not os.path.exists(engine_file):
        fallback_a = r"D:\python\TX RX\RealESRGAN_x4plus_fp16_NEW.engine"
        fallback_b = r"D:\python\TX RX\RealESRGAN_x4plus_fp16.engine"
        if os.path.exists(fallback_a):
            engine_file = fallback_a
        elif os.path.exists(fallback_b):
            engine_file = fallback_b
        else:
            return False
    try:
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if not engine:
            return False
        TRT_CONTEXT = engine.create_execution_context()
        TRT_STREAM = torch.cuda.Stream()
        inp_shape = (1, 3, ENGINE_INPUT_H, ENGINE_INPUT_W)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                 TRT_CONTEXT.set_input_shape(name, inp_shape)
                 in_dtype = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                 TRT_INPUT_TENSOR = torch.zeros(inp_shape, dtype=in_dtype, device=DEVICE).contiguous()
                 TRT_CONTEXT.set_tensor_address(name, int(TRT_INPUT_TENSOR.data_ptr()))
        TRT_CONTEXT.infer_shapes()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                 out_shape = tuple(TRT_CONTEXT.get_tensor_shape(name))
                 out_dtype = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                 TRT_OUTPUT_TENSOR = torch.zeros(out_shape, dtype=out_dtype, device=DEVICE).contiguous()
                 TRT_CONTEXT.set_tensor_address(name, int(TRT_OUTPUT_TENSOR.data_ptr()))
        return True
    except Exception:
        return False

def infer_trt(frame_cv2):
    h, w = frame_cv2.shape[:2]
    if w != ENGINE_INPUT_W or h != ENGINE_INPUT_H:
        frame_cv2 = cv2.resize(frame_cv2, (ENGINE_INPUT_W, ENGINE_INPUT_H))
    rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    inp_np = np.array(rgb).astype(np.float32) / 255.0
    inp_np = np.transpose(inp_np, (2, 0, 1))
    inp_np = np.expand_dims(inp_np, 0)       
    TRT_INPUT_TENSOR.copy_(torch.from_numpy(inp_np))
    TRT_CONTEXT.execute_async_v3(stream_handle=TRT_STREAM.cuda_stream)
    TRT_STREAM.synchronize()
    out_np = TRT_OUTPUT_TENSOR.float().cpu().numpy()
    out_np = np.squeeze(out_np, 0)
    out_np = np.transpose(out_np, (1, 2, 0))
    
    # 偵測神經網路權重是否崩潰 (FP16 Overflow)
    nan_count = np.isnan(out_np).sum()
    if nan_count > 1000:
        print("[TRT WARNING] Engine output is corrupted (NaNs detected). Falling back to cv2.resize.")
        # 降級使用 OpenCV 放大，確保畫面不會爛掉
        upscaled = cv2.resize(frame_cv2, (ENGINE_INPUT_W * 4, ENGINE_INPUT_H * 4), interpolation=cv2.INTER_CUBIC)
        return upscaled
    
    # FP16 TensorRT 模型常見溢位修復
    out_np = np.nan_to_num(out_np, nan=0.0, posinf=1.0, neginf=0.0)
    
    if np.max(out_np) > 2.0:
        out_np = np.clip(out_np, 0, 255)
    else:
        out_np = np.clip(out_np, 0, 1) * 255.0
    out_np = out_np.astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

def compute_optical_flow(prev_gray, curr_gray):
    h, w = prev_gray.shape
    small_prev = cv2.resize(prev_gray, (w // 4, h // 4))
    small_curr = cv2.resize(curr_gray, (w // 4, h // 4))
    low_flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(low_flow[..., 0], low_flow[..., 1])
    full_flow = np.zeros((h, w, 2), dtype=np.float32)
    weight_map = np.zeros((h, w, 1), dtype=np.float32)
    TILE_H, TILE_W = h // 2, w // 2
    OVERLAP = 16 
    for i in range(2):
        for j in range(2):
            y1 = max(0, i * TILE_H - OVERLAP)
            y2 = min(h, (i + 1) * TILE_H + OVERLAP)
            x1 = max(0, j * TILE_W - OVERLAP)
            x2 = min(w, (j + 1) * TILE_W + OVERLAP)
            th, tw = y2 - y1, x2 - x1
            wy = np.ones(th, dtype=np.float32)
            wx = np.ones(tw, dtype=np.float32)
            if i > 0: wy[:OVERLAP] = np.linspace(0, 1, OVERLAP)
            if i < 1: wy[-OVERLAP:] = np.linspace(1, 0, OVERLAP)
            if j > 0: wx[:OVERLAP] = np.linspace(0, 1, OVERLAP)
            if j < 1: wx[-OVERLAP:] = np.linspace(1, 0, OVERLAP)
            w_tile = (wy[:, None] * wx[None, :])[:, :, None]
            lr_y1, lr_y2 = i * (TILE_H // 4), (i + 1) * (TILE_H // 4)
            lr_x1, lr_x2 = j * (TILE_W // 4), (j + 1) * (TILE_W // 4)
            if np.max(mag[lr_y1:lr_y2, lr_x1:lr_x2]) > 0.4:
                t_prev = prev_gray[y1:y2, x1:x2]
                t_curr = curr_gray[y1:y2, x1:x2]
                t_flow = cv2.calcOpticalFlowFarneback(t_prev, t_curr, None, 0.5, 5, 21, 10, 5, 1.2, 0)
                full_flow[y1:y2, x1:x2] += t_flow * w_tile
            weight_map[y1:y2, x1:x2] += w_tile
    weight_map = np.clip(weight_map, 1e-5, None)
    full_flow /= weight_map 
    return full_flow

def warp_frame(frame_prev, frame_curr, flow, last_flow=None, alpha=0.5):
    h, w = frame_prev.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    if last_flow is not None:
        flow_fw_x = last_flow[:, :, 0] * alpha + (flow[:, :, 0] - last_flow[:, :, 0]) * (alpha ** 2)
        flow_fw_y = last_flow[:, :, 1] * alpha + (flow[:, :, 1] - last_flow[:, :, 1]) * (alpha ** 2)
        flow_bw_x = flow[:, :, 0] - flow_fw_x
        flow_bw_y = flow[:, :, 1] - flow_fw_y
    else:
        flow_fw_x = flow[:, :, 0] * alpha
        flow_fw_y = flow[:, :, 1] * alpha
        flow_bw_x = flow[:, :, 0] * (1.0 - alpha)
        flow_bw_y = flow[:, :, 1] * (1.0 - alpha)
    map_x_prev = map_x.astype(np.float32) - flow_fw_x
    map_y_prev = map_y.astype(np.float32) - flow_fw_y
    map_x_curr = map_x.astype(np.float32) + flow_bw_x
    map_y_curr = map_y.astype(np.float32) + flow_bw_y
    warp_prev = cv2.remap(frame_prev, map_x_prev, map_y_prev, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp_curr = cv2.remap(frame_curr, map_x_curr, map_y_curr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    interp_frame = cv2.addWeighted(warp_prev, 1.0 - alpha, warp_curr, alpha, 0)
    return interp_frame

def serial_reader_thread():
    global running, right_count, wrong_count, dropped_frames_since_last_success
    buffer = bytearray()
    chunk = bytearray() 
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
    except Exception:
        running = False
        return
    while running:
        try:
            read_data = ser.read(CHUNK_SIZE)
            if not read_data:
                continue
            chunk.extend(read_data)
            while len(chunk) >= 3:
                payloadLen = chunk[0]
                if payloadLen > 255:
                    chunk = chunk[1:]
                    continue
                if len(chunk) < payloadLen + 3:
                    break
                payload = chunk[1 : payloadLen + 1]
                rssi = chunk[payloadLen + 1] - 256
                buffer.extend(payload)
                chunk = chunk[payloadLen + 3:]
            while True:
                header_sig = b'ftypavif'
                match_idx = buffer.find(header_sig)
                if match_idx == -1:
                    break
                start_offset = match_idx - 4
                if start_offset > 0:
                    buffer = buffer[start_offset:]
                    match_idx = 4 
                elif start_offset < 0:
                     pass
                next_match_idx = buffer.find(header_sig, match_idx + len(header_sig))
                if next_match_idx == -1:
                    break 
                frame_end = next_match_idx - 4
                frame_data = buffer[:frame_end]
                buffer = buffer[frame_end:] 
                try:
                    try:
                        pil_img = Image.open(io.BytesIO(frame_data))
                        frame_rgb = np.array(pil_img)
                        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    except Exception:
                        frame = None
                    if frame is not None:
                        right_count += 1
                        if frame_queue.full():
                            try:
                                frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        frame_queue.put((frame, rssi, dropped_frames_since_last_success))
                        dropped_frames_since_last_success = 0 
                    else:
                        wrong_count += 1
                        dropped_frames_since_last_success += 1
                except Exception:
                    wrong_count += 1
                    dropped_frames_since_last_success += 1
        except Exception:
            time.sleep(0.1)
    try: ser.close()
    except: pass

def flow_warp_thread():
    global running, sliding_window, display_queue, display_lock
    last_flow = None
    while running:
        try:
            if len(sliding_window) >= 2:
                if time.time() - sliding_window[0]["time"] >= 0.7:
                    prev_data = sliding_window.popleft()
                    curr_data = sliding_window[0] 
                    flow = compute_optical_flow(prev_data["gray"], curr_data["gray"])
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    max_mag = np.max(mag)
                    mag = np.clip(mag, 0, 30.0) 
                    flow[..., 0], flow[..., 1] = cv2.polarToCart(mag, ang)
                    prev_flow_unmodified = last_flow
                    if last_flow is not None:
                        flow = 0.6 * flow + 0.4 * last_flow
                    last_flow = flow
                    dropped_frames = curr_data["dropped"]
                    if max_mag > 40.0 or dropped_frames == 0:
                        num_interp = 0
                    else: 
                        num_interp = min(dropped_frames, 10) 
                    with display_lock:
                        display_queue.append((prev_data["sr"], prev_data["rssi"]))
                    if num_interp > 0:
                        h_sr, w_sr = prev_data["sr"].shape[:2]
                        flow_sr = cv2.resize(flow, (w_sr, h_sr))
                        flow_sr[:, :, 0] *= (w_sr / 320.0)
                        flow_sr[:, :, 1] *= (h_sr / 240.0)
                        if prev_flow_unmodified is not None:
                            last_flow_sr = cv2.resize(prev_flow_unmodified, (w_sr, h_sr))
                            last_flow_sr[:, :, 0] *= (w_sr / 320.0)
                            last_flow_sr[:, :, 1] *= (h_sr / 240.0)
                        else:
                            last_flow_sr = None
                        for i in range(1, num_interp + 1):
                            alpha = i / (num_interp + 1.0)
                            interp_bgr_sr = warp_frame(prev_data["sr"], curr_data["sr"], flow_sr, last_flow=last_flow_sr, alpha=alpha)
                            with display_lock:
                                display_queue.append((interp_bgr_sr, prev_data["rssi"])) 
                    continue  
        except Exception:
            pass
        time.sleep(0.005)

def stdin_thread():
    global running, super_res_enabled
    while running:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line == "SR_ON":
                super_res_enabled = True
            elif line == "SR_OFF":
                super_res_enabled = False
            elif line == "QUIT":
                running = False
                break
        except:
            pass

def main():
    global running, display_queue, super_res_enabled, right_count, wrong_count
    
    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    trt_ok = init_trt()
    t = threading.Thread(target=serial_reader_thread)
    t.daemon = True
    t.start()
    t_warp = threading.Thread(target=flow_warp_thread)
    t_warp.daemon = True
    t_warp.start()
    t_std = threading.Thread(target=stdin_thread)
    t_std.daemon = True
    t_std.start()
    prev_time = time.time()
    fps = 0
    rssi = 0
    target_display_w = 1280
    target_display_h = 960
    while running:
        has_new_frame = False
        try:
            raw_frame, new_rssi, dropped_count = frame_queue.get_nowait()
            rssi = new_rssi
            has_new_frame = True
            while not frame_queue.empty():
                 raw_frame, rssi, dropped_count = frame_queue.get_nowait()
        except queue.Empty:
            pass
        if has_new_frame:
            curr_bgr = cv2.resize(raw_frame, (320, 240))
            curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
            if trt_ok and super_res_enabled:
                try: curr_bgr_sr = infer_trt(curr_bgr)
                except: curr_bgr_sr = curr_bgr.copy()
            else:
                curr_bgr_sr = curr_bgr.copy()
            sliding_window.append({
                "bgr": curr_bgr, "gray": curr_gray, "sr": curr_bgr_sr,
                "rssi": rssi, "dropped": dropped_count, "time": time.time()
            })
        current_time = time.time()
        if current_time - prev_time >= (1.0 / 24.0): 
            dt = current_time - prev_time
            fps = 1.0 / dt if dt > 0 else 0
            prev_time = current_time
            with display_lock:
                if len(display_queue) > 0:
                    proc_frame, frame_rssi = display_queue.pop(0)
                    rssi = frame_rssi
                    if proc_frame.shape[1] != target_display_w or proc_frame.shape[0] != target_display_h:
                        proc_frame = cv2.resize(proc_frame, (target_display_w, target_display_h), interpolation=cv2.INTER_LINEAR)
                    rgb_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                    header = struct.pack('<4sIIfiiii', b'SYNC', target_display_w, target_display_h, float(fps), int(rssi), int(super_res_enabled), int(right_count), int(wrong_count))
                    try:
                        sys.stdout.buffer.write(header)
                        sys.stdout.buffer.write(rgb_frame.tobytes())
                        sys.stdout.buffer.flush()
                    except:
                        running = False
                        break
                if len(display_queue) > 30: 
                    display_queue = [display_queue[-1]]
        time.sleep(0.001)

if __name__ == "__main__":
    main()
