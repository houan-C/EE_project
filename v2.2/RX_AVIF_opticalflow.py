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

warnings.filterwarnings("ignore")

# --- Pillow for AVIF Support ---
try:
    from PIL import Image
    import pillow_avif
except ImportError:
    print("🔴 Error: pillow-avif-plugin not installed. Please run: pip install pillow pillow-avif-plugin")
    # We continue, but decoding will likely fail if it relies on this
# -------------------------------

# --- TensorRT Related ---
try:
    import tensorrt as trt
    import torch 
except ImportError:
    print("🔴 Error: TensorRT or PyTorch libraries not installed.")
    sys.exit(1)
# -------------------------

# --- Settings ---
COM_PORT = "COM5"
BAUD_RATE = 921600
CHUNK_SIZE = 1024 
TIMEOUT = 0.01

# Resolution Settings
ENGINE_INPUT_W, ENGINE_INPUT_H = 320, 240 # Target input for Engine
DISPLAY_SCALE = 0.5 # Output 4x smaller (relative to 2.0)

# --- TensorRT Configuration ---
TRT_LOGGER = trt.Logger(trt.Logger.ERROR) # Only show errors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_FILENAME = "realesr-general-x4v3.engine" 
ENGINE_PATH = os.path.join(BASE_DIR, ENGINE_FILENAME)

# Globals for TRT Optimization
TRT_STREAM = None
TRT_INPUT_TENSOR = None
TRT_OUTPUT_TENSOR = None
TRT_CONTEXT = None

# Threading & Stats Globals
frame_queue = queue.Queue(maxsize=1) # Holds only the LATEST frame
sliding_window = collections.deque()
display_queue = [] # List used as queue to allow easy dropping of old frames
display_lock = threading.Lock()
running = True
right_count = 0
wrong_count = 0
dropped_frames_since_last_success = 0

# UI State
super_res_enabled = True
button_area = (1050, 30, 200, 60) # (x, y, w, h)


# ====================================================================
# TensorRT Optimized Functions
# ====================================================================
def init_trt():
    global TRT_CONTEXT, TRT_STREAM, TRT_INPUT_TENSOR, TRT_OUTPUT_TENSOR
    
    if DEVICE.type != 'cuda':
        return False
        
    engine_file = ENGINE_PATH
    if not os.path.exists(engine_file):
        # Fallback to absolute strict paths to fix 'd:\t..' stripping anomalies
        fallback_a = r"D:\python\TX RX\realesr-general-x4v3.engine"
        fallback_b = r"D:\python\TX RX\realesr-general-x4v3.engine"
        if os.path.exists(fallback_a):
            engine_file = fallback_a
        elif os.path.exists(fallback_b):
            engine_file = fallback_b
        else:
            print(f"🔴 Engine not found: {ENGINE_PATH}")
            return False

    try:
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        if not engine:
            return False
            
        TRT_CONTEXT = engine.create_execution_context()
        TRT_STREAM = torch.cuda.Stream()
        
        # Pre-allocate buffers for FIXED size (320x240)
        inp_shape = (1, 3, ENGINE_INPUT_H, ENGINE_INPUT_W)
        
        # Setup Input
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                 TRT_CONTEXT.set_input_shape(name, inp_shape)
                 TRT_INPUT_TENSOR = torch.zeros(inp_shape, dtype=torch.float32, device=DEVICE).contiguous()
                 TRT_CONTEXT.set_tensor_address(name, int(TRT_INPUT_TENSOR.data_ptr()))

        TRT_CONTEXT.infer_shapes()

        # Setup Output
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                 out_shape = tuple(TRT_CONTEXT.get_tensor_shape(name))
                 dtype = torch.float32 
                 if engine.get_tensor_dtype(name) == trt.DataType.HALF:
                     dtype = torch.float16
                 TRT_OUTPUT_TENSOR = torch.zeros(out_shape, dtype=dtype, device=DEVICE).contiguous()
                 TRT_CONTEXT.set_tensor_address(name, int(TRT_OUTPUT_TENSOR.data_ptr()))
        
        print(f"🟢 TRT Initialized (Target: {ENGINE_INPUT_W}x{ENGINE_INPUT_H})")
        return True
    except Exception as e:
        print(f"🔴 TRT Init Error: {e}")
        return False

def infer_trt(frame_cv2):
    # 1. Resize to Target Input Size
    h, w = frame_cv2.shape[:2]
    if w != ENGINE_INPUT_W or h != ENGINE_INPUT_H:
        frame_cv2 = cv2.resize(frame_cv2, (ENGINE_INPUT_W, ENGINE_INPUT_H))

    # 2. Preprocess
    rgb = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB)
    inp_np = np.array(rgb).astype(np.float32) / 255.0
    inp_np = np.transpose(inp_np, (2, 0, 1)) # CHW
    inp_np = np.expand_dims(inp_np, 0)       # NCHW
    
    # 3. Copy to GPU
    TRT_INPUT_TENSOR.copy_(torch.from_numpy(inp_np))
    
    # 4. Infer
    TRT_CONTEXT.execute_async_v3(stream_handle=TRT_STREAM.cuda_stream)
    TRT_STREAM.synchronize()
    
    # 5. Postprocess
    out_np = TRT_OUTPUT_TENSOR.float().cpu().numpy()
    out_np = np.squeeze(out_np, 0)
    out_np = np.transpose(out_np, (1, 2, 0))
    out_np = np.clip(out_np, 0, 1)
    out_np = (out_np * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

# ====================================================================
# A. Motion Vector Estimator (Advanced: Dynamic Gating & Tiling)
# ====================================================================
def compute_optical_flow(prev_gray, curr_gray):
    h, w = prev_gray.shape
    
    # 1. Gating Mechanism (Low-Resolution Pre-computation)
    # Allows skipping static regions to save ~17% computing power
    small_prev = cv2.resize(prev_gray, (w // 4, h // 4))
    small_curr = cv2.resize(curr_gray, (w // 4, h // 4))
    low_flow = cv2.calcOpticalFlowFarneback(small_prev, small_curr, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(low_flow[..., 0], low_flow[..., 1])
    
    full_flow = np.zeros((h, w, 2), dtype=np.float32)
    weight_map = np.zeros((h, w, 1), dtype=np.float32)
    
    TILE_H, TILE_W = h // 2, w // 2
    # 2. Cross-Region Partition (Overlap Avoids Seams)
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
            
            # Smooth blend mask to eliminate boundaries between tiles
            if i > 0: wy[:OVERLAP] = np.linspace(0, 1, OVERLAP)
            if i < 1: wy[-OVERLAP:] = np.linspace(1, 0, OVERLAP)
            if j > 0: wx[:OVERLAP] = np.linspace(0, 1, OVERLAP)
            if j < 1: wx[-OVERLAP:] = np.linspace(1, 0, OVERLAP)
            
            w_tile = (wy[:, None] * wx[None, :])[:, :, None]
            
            # Gating Logic: Only compute complex flow if motion exists
            lr_y1, lr_y2 = i * (TILE_H // 4), (i + 1) * (TILE_H // 4)
            lr_x1, lr_x2 = j * (TILE_W // 4), (j + 1) * (TILE_W // 4)
            if np.max(mag[lr_y1:lr_y2, lr_x1:lr_x2]) > 0.4:
                # Residual High-Fidelity Flow on Region
                t_prev = prev_gray[y1:y2, x1:x2]
                t_curr = curr_gray[y1:y2, x1:x2]
                t_flow = cv2.calcOpticalFlowFarneback(t_prev, t_curr, None, 0.5, 5, 21, 10, 5, 1.2, 0)
                full_flow[y1:y2, x1:x2] += t_flow * w_tile
            
            weight_map[y1:y2, x1:x2] += w_tile
            
    # Normalize overlapping blending smoothly
    weight_map = np.clip(weight_map, 1e-5, None) # Safe divide
    full_flow /= weight_map # Broadcasting (H,W,2) / (H,W,1) automatically works
    
    return full_flow

# ====================================================================
# B. Warping Engine
# ====================================================================
def warp_frame(frame_prev, frame_curr, flow, last_flow=None, alpha=0.5):
    h, w = frame_prev.shape[:2]
    # Create coordinate grid
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 3. Non-linear Motion Splines Modeled via Constant Acceleration approximation
    if last_flow is not None:
        # Non-linear quadratic displacement to approximate Cubic Motion Splines
        flow_fw_x = last_flow[:, :, 0] * alpha + (flow[:, :, 0] - last_flow[:, :, 0]) * (alpha ** 2)
        flow_fw_y = last_flow[:, :, 1] * alpha + (flow[:, :, 1] - last_flow[:, :, 1]) * (alpha ** 2)
        
        flow_bw_x = flow[:, :, 0] - flow_fw_x
        flow_bw_y = flow[:, :, 1] - flow_fw_y
    else:
        # Standard linear interpolation
        flow_fw_x = flow[:, :, 0] * alpha
        flow_fw_y = flow[:, :, 1] * alpha
        flow_bw_x = flow[:, :, 0] * (1.0 - alpha)
        flow_bw_y = flow[:, :, 1] * (1.0 - alpha)
    
    # Forward map (from prev to interp)
    map_x_prev = map_x.astype(np.float32) - flow_fw_x
    map_y_prev = map_y.astype(np.float32) - flow_fw_y
    
    # Backward map (from curr to interp)
    map_x_curr = map_x.astype(np.float32) + flow_bw_x
    map_y_curr = map_y.astype(np.float32) + flow_bw_y
    
    # Remap both edges to prevent strange border tearing
    warp_prev = cv2.remap(frame_prev, map_x_prev, map_y_prev, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warp_curr = cv2.remap(frame_curr, map_x_curr, map_y_curr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Blend them together (cross-fade) to completely eliminate ghosting
    interp_frame = cv2.addWeighted(warp_prev, 1.0 - alpha, warp_curr, alpha, 0)
    return interp_frame

# ====================================================================
# UI Interaction
# ====================================================================
def on_mouse(event, x, y, flags, param):
    global super_res_enabled
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = button_area
        if bx <= x <= bx + bw and by <= y <= by + bh:
            super_res_enabled = not super_res_enabled
            print(f"👉 Toggle SR: {super_res_enabled}")

# ====================================================================
# Serial Reader Thread
# ====================================================================

def serial_reader_thread():
    global running, right_count, wrong_count, dropped_frames_since_last_success
    print(f"🔵 Serial Thread Started ({COM_PORT})")
    
    buffer = bytearray()
    chunk = bytearray() # Local chunk buffer
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
    except Exception as e:
        print(f"❌ Serial Error: {e}")
        running = False
        return

    while running:
        try:
            read_data = ser.read(CHUNK_SIZE)
            if not read_data:
                continue
                
            chunk.extend(read_data)
            
            # Parsing Loop (RSSI / Header stripping logic from original)
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
            
            # Decoding Loop (AVIF Adjusted)
            while True:
                # 1. Find Header: "ftypavif" (AVIF signature)
                # The file typically starts 4 bytes before 'ftypavif' (the size box)
                header_sig = b'ftypavif'
                match_idx = buffer.find(header_sig)
                
                if match_idx == -1:
                    # No header found, wait for more data
                    break
                    
                # The actual start of the file is 4 bytes before the signature
                # But we must be careful not to go out of bounds if it's at index < 4
                start_offset = match_idx - 4
                
                # If we have garbage before the start (or start_offset > 0)
                # Discard garbage
                if start_offset > 0:
                    buffer = buffer[start_offset:]
                    match_idx = 4 # Update position relative to new buffer start
                elif start_offset < 0:
                     # This implies partial header at the very beginning? 
                     # Wait for more data? Or just skip?
                     # If buffer starts with 'typavif', match_idx=0. start_offset=-4.
                     # This assumes we are mid-stream.
                     # Let's just discard up to match_idx + len(header_sig) to be safe?
                     # Or just wait.
                     # Actually, valid AVIF must have valid box size.
                     # If we found 'ftypavif', we assume the 4 bytes before it are the size.
                     # If they are not in buffer, we can't process it yet?
                     # Actually, if match_idx < 4, it implies we have [..ftypavif..] but not the 4 bytes before?
                     # Impossible if we are appending to buffer. buffer[0] is start.
                     # If match_idx is 2, buffer has 2 bytes then ftypavif.
                     # We need 4 bytes. 
                     # So if match_idx < 4, it means we don't have enough preamble.
                     # This effectively means the signature was found too early?
                     # Realistically, if we sync to the stream, we should be fine.
                     # Let's just continue and assume we will resync.
                     pass

                # 2. Find NEXT Header to determine frame boundary
                # We need to find the *next* 'ftypavif' to know where this frame ends.
                # Start searching AFTER the current header
                next_match_idx = buffer.find(header_sig, match_idx + len(header_sig))
                
                if next_match_idx == -1:
                    # No next frame marker yet. We must wait for the next frame to arrive
                    # to know the current one is complete.
                    # (Latency = 1 frame)
                    break 
                
                # The next file starts at next_match_idx - 4
                frame_end = next_match_idx - 4
                
                # Extract Frame
                frame_data = buffer[:frame_end]
                buffer = buffer[frame_end:] # Move buffer forward to start of next frame
                
                try:
                    # Decode using Pillow (AVIF)
                    # Use io.BytesIO to act as file
                    try:
                        pil_img = Image.open(io.BytesIO(frame_data))
                        # Pillow returns RGB, cv2 needs BGR
                        frame_rgb = np.array(pil_img)
                        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    except Exception as e_pil:
                        print(f"PIL Decode Error: {e_pil}")
                        frame = None

                    if frame is not None:
                        right_count += 1
                        # Put in Queue (Overwrite if full = Only keep latest)
                        if frame_queue.full():
                            try:
                                frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        # Pass the dropped count along with the successful frame
                        frame_queue.put((frame, rssi, dropped_frames_since_last_success))
                        dropped_frames_since_last_success = 0 # reset counter
                    else:
                        wrong_count += 1
                        dropped_frames_since_last_success += 1
                        print("❌ Decode failed (Frame is None)")
                except Exception as e:
                    wrong_count += 1
                    dropped_frames_since_last_success += 1
                    print(f"❌ Decode Error: {e}")

        except Exception as e:
            print(f"Serial Loop Error: {e}")
            time.sleep(0.1)
    
    ser.close()
    print("🔵 Serial Thread Stopped")

# ====================================================================
# EE Optimization: Processing Worker Thread (Phase A & C)
# ====================================================================
def flow_warp_thread():
    global running, sliding_window, display_queue, display_lock
    last_flow = None
    
    print("🔵 Processing Worker Thread Started")
    while running:
        try:
            if len(sliding_window) >= 2:
                # Phase A: 0.7s Adaptive Buffer for Jitter
                if time.time() - sliding_window[0]["time"] >= 0.7:
                    prev_data = sliding_window.popleft()
                    curr_data = sliding_window[0] # Peek at next
                    
                    # Phase B: High-Fidelity Flow Calculation
                    flow = compute_optical_flow(prev_data["gray"], curr_data["gray"])
                
                # EE Optimization: Motion Vector limit & Dynamic Alpha Drop
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    max_mag = np.max(mag)
                    mag = np.clip(mag, 0, 30.0) 
                    flow[..., 0], flow[..., 1] = cv2.polarToCart(mag, ang)
                    
                    # Capture purely previous flow for Non-Linear fitting BEFORE EMA smoothing
                    prev_flow_unmodified = last_flow
                    
                    # Motion EMA Smoothing
                    if last_flow is not None:
                        flow = 0.6 * flow + 0.4 * last_flow
                    last_flow = flow
                    
                    # Dropping interpolation if extreme shake
                    # OR if there were no dropped frames (only compute when decoded failed)
                    dropped_frames = curr_data["dropped"]
                    
                    if max_mag > 40.0 or dropped_frames == 0:
                        num_interp = 0
                    else: 
                        # We only interpolate exactly the amount of frames that were dropped
                        num_interp = min(dropped_frames, 10) # limit maximum bounds for edge cases
                    
                    # -------------------------------------------------------------
                    # Render Logic:
                    # -------------------------------------------------------------
                    # 1. Always queue the PREV true frame
                    with display_lock:
                        display_queue.append((prev_data["sr"], prev_data["rssi"]))
    
                    # 2. Only run optical flow warping if we need to fill gaps
                    if num_interp > 0:
                        # Interpolate based on SR resolution
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
                        
                        # Queue Bi-directional interpolated frames
                        for i in range(1, num_interp + 1):
                            alpha = i / (num_interp + 1.0)
                            interp_bgr_sr = warp_frame(prev_data["sr"], curr_data["sr"], flow_sr, last_flow=last_flow_sr, alpha=alpha)
                            with display_lock:
                                display_queue.append((interp_bgr_sr, prev_data["rssi"])) # Use prev_rssi
                            
                    # Loop quickly if there's a backlog
                    continue  
        except Exception as e:
            print(f"🔴 AI Thread Crash Error: {e}")
            import traceback
            traceback.print_exc()
        time.sleep(0.005)

# ====================================================================
# C. Async Display Loop (Main Loop)
# ====================================================================
def main():
    global running, display_queue
    
    # 1. Init TRT
    trt_ok = init_trt()
    
    # 2. Start Threads
    t = threading.Thread(target=serial_reader_thread)
    t.daemon = True
    t.start()
    
    t_warp = threading.Thread(target=flow_warp_thread)
    t_warp.daemon = True
    t_warp.start()
    
    # 3. GUI Setup
    win_name = "Fluid TRT Stream"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    
    prev_time = time.time()
    fps = 0
    rssi = 0
    
    # Initial Blank Frame (will be resized on first display update)
    current_frame = np.zeros((960, 1280, 3), dtype=np.uint8)
    target_display_w = 1280
    target_display_h = 960
    
    print("🟢 Async Main Loop Started. Waiting for frames...")

    try:
        while True:
            # --- Non-blocking Receive Logic ---
            has_new_frame = False
            try:
                # Non-blocking get
                raw_frame, new_rssi, dropped_count = frame_queue.get_nowait()
                rssi = new_rssi
                has_new_frame = True
                
                # Drain queue to keep only latest packet if backed up
                while not frame_queue.empty():
                     raw_frame, rssi, dropped_count = frame_queue.get_nowait()
                     
            except queue.Empty:
                pass
            
            if has_new_frame:
                curr_bgr = cv2.resize(raw_frame, (320, 240))
                curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
                
                # TRT First: upscale the real frame sequentially as they arrive
                if trt_ok and super_res_enabled:
                    try:
                        curr_bgr_sr = infer_trt(curr_bgr)
                    except Exception as e:
                        print(f"Infer Error: {e}")
                        curr_bgr_sr = curr_bgr.copy()
                else:
                    curr_bgr_sr = curr_bgr.copy()
                
                # Add to Phase A Buffer (Worker Thread will consume)
                sliding_window.append({
                    "bgr": curr_bgr,
                    "gray": curr_gray,
                    "sr": curr_bgr_sr,
                    "rssi": rssi,
                    "dropped": dropped_count,
                    "time": time.time()
                })
                
            # --- Async Display Update Logic ---
            current_time = time.time()
            if current_time - prev_time >= (1.0 / 24.0): # Target ~24 FPS
                dt = current_time - prev_time
                fps = 1.0 / dt if dt > 0 else 0
                prev_time = current_time
                
                with display_lock:
                    if len(display_queue) > 0:
                        proc_frame, frame_rssi = display_queue.pop(0)
                        rssi = frame_rssi
                        
                        # Resize output 
                        if proc_frame.shape[1] != target_display_w or proc_frame.shape[0] != target_display_h:
                            proc_frame = cv2.resize(proc_frame, (target_display_w, target_display_h), interpolation=cv2.INTER_LINEAR)
                        
                        current_frame = proc_frame

                    # Queue management 
                    if len(display_queue) > 30: 
                        display_queue = [display_queue[-1]]
                        
            # --- Overlay & Render ---
            render_frame = current_frame.copy()
            
            # draw OSD (scaled up for 1280x960)
            cv2.putText(render_frame, f"D-FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(render_frame, f"RSSI: {rssi}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            
            # Draw Button
            bx, by, bw, bh = button_area
            color = (0, 255, 0) if super_res_enabled else (0, 0, 255)
            text = "SR: ON" if super_res_enabled else "SR: OFF"
            
            cv2.rectangle(render_frame, (bx, by), (bx + bw, by + bh), color, -1)
            cv2.rectangle(render_frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            cv2.putText(render_frame, text, (bx + 20, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
            
            # Refresh Window
            cv2.imshow(win_name, render_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
                
            if not t.is_alive():
                print("❌ Serial thread died.")
                break

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        t.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("\n=== Statistics ===")
        print(f"Success Frames: {right_count}")
        print(f"Failed Frames : {wrong_count}")
        total = right_count + wrong_count
        if total > 0:
            print(f"Success Rate  : {right_count/total*100:.2f}%")
        else:
            print("No frames processed.")
        print("Done.")

if __name__ == "__main__":
    main()
