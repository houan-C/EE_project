import cv2
import numpy as np
import time
import os
import threading
import queue
import io
import collections
import serial
from auto_discover import get_role_port

try:
    from PIL import Image
    import pillow_avif
except ImportError:
    print("[ERROR] pillow-avif-plugin not installed.")

try:
    import tensorrt as trt
    import torch
    _HAS_TRT = True
except ImportError:
    print("[WARN] TensorRT/PyTorch not found. SR and RIFE disabled.")
    trt = None
    _HAS_TRT = False

# ====================================================================
# === USER SETTINGS ===
# ====================================================================
COM_PORT = get_role_port("RX")
BAUD_RATE = 921600
CHUNK_SIZE = 1024
TIMEOUT = 0.01

TARGET_W, TARGET_H = 1280, 960
SRC_W, SRC_H = 320, 240
RIFE_PAD_H = 256  # 240 padded up to 256

# ====================================================================
# === TRT GLOBALS ===
# ====================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

if _HAS_TRT:
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SR_ENGINE_PATH  = os.path.join(MODELS_DIR, "realesr-general-x4v3.engine")
SR_CTX = None; SR_STREAM = None; SR_INPUT = None; SR_OUTPUT = None

RIFE_ENGINE_PATH  = os.path.join(MODELS_DIR, "flownet.engine")
RIFE_CTX = None; RIFE_STREAM = None; RIFE_INPUT = None; RIFE_TIMESTEP = None
RIFE_OUT_TENSORS = {}; RIFE_DUMMY = []; RIFE_OUTPUT = None

# ====================================================================
# === QUEUES & STATE ===
# ====================================================================
rife_input_q = queue.Queue(maxsize=30)
display_q    = queue.Queue(maxsize=60)

running        = True
right_count    = 0
wrong_count    = 0
super_res_on   = True
sr_btn_area    = (1050, 30, 200, 60)

_RIFE_STAGING = np.zeros((1, 6, RIFE_PAD_H, SRC_W), dtype=np.float32)

def init_sr():
    global SR_CTX, SR_STREAM, SR_INPUT, SR_OUTPUT
    if not _HAS_TRT or DEVICE.type != "cuda": return False
    if not os.path.exists(SR_ENGINE_PATH): return False
    try:
        with open(SR_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine: return False
        SR_CTX = engine.create_execution_context()
        SR_STREAM = torch.cuda.Stream()
        inp_shape = (1, 3, SRC_H, SRC_W)
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
        print(f"[SR ] Initialized ({SRC_W}x{SRC_H} -> {TARGET_W}x{TARGET_H})")
        return True
    except Exception as e:
        print(f"[SR ] Init failed: {e}")
        return False

def infer_sr(bgr_320x240):
    rgb = cv2.cvtColor(bgr_320x240, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    SR_INPUT.copy_(t)
    SR_CTX.execute_async_v3(stream_handle=SR_STREAM.cuda_stream)
    SR_STREAM.synchronize()
    out = SR_OUTPUT.float().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return cv2.cvtColor((out.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def init_rife():
    global RIFE_CTX, RIFE_STREAM, RIFE_INPUT, RIFE_TIMESTEP
    global RIFE_OUT_TENSORS, RIFE_DUMMY, RIFE_OUTPUT
    if not _HAS_TRT or DEVICE.type != "cuda": return False
    if not os.path.exists(RIFE_ENGINE_PATH): return False
    try:
        with open(RIFE_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine: return False
        RIFE_CTX = engine.create_execution_context()
        RIFE_STREAM = torch.cuda.Stream()
        inp_shape = (1, 6, RIFE_PAD_H, SRC_W)
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
        target_shape = (1, 3, RIFE_PAD_H, SRC_W)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(RIFE_CTX.get_tensor_shape(name))
                dt = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                t = torch.zeros(shape, dtype=dt, device=DEVICE).contiguous()
                RIFE_CTX.set_tensor_address(name, int(t.data_ptr()))
                RIFE_OUT_TENSORS[name] = t
                if name == "2787": RIFE_OUTPUT = t
        if RIFE_OUTPUT is None:
            for v in RIFE_OUT_TENSORS.values():
                if tuple(v.shape) == target_shape:
                    RIFE_OUTPUT = v
                    break
        print(f"[RIFE] Initialized ({SRC_W}x{SRC_H} padded to {SRC_W}x{RIFE_PAD_H})")
        return True
    except Exception as e:
        print(f"[RIFE] Init failed: {e}")
        return False

def infer_rife(bgr0, bgr1):
    pad = RIFE_PAD_H - SRC_H
    for ch in range(3):
        _RIFE_STAGING[0, ch, :SRC_H, :] = bgr0[:, :, 2 - ch] / 255.0
        _RIFE_STAGING[0, ch, SRC_H:, :] = _RIFE_STAGING[0, ch, SRC_H-1:SRC_H, :]
        _RIFE_STAGING[0, ch+3, :SRC_H, :] = bgr1[:, :, 2 - ch] / 255.0
        _RIFE_STAGING[0, ch+3, SRC_H:, :] = _RIFE_STAGING[0, ch+3, SRC_H-1:SRC_H, :]
    RIFE_INPUT.copy_(torch.from_numpy(_RIFE_STAGING))
    RIFE_CTX.execute_async_v3(stream_handle=RIFE_STREAM.cuda_stream)
    RIFE_STREAM.synchronize()
    out = RIFE_OUTPUT.float().squeeze(0).permute(1, 2, 0)[:SRC_H].clamp(0, 1)
    out_np = out.cpu().numpy()
    return cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def on_mouse(event, x, y, flags, param):
    global super_res_on
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = sr_btn_area
        # Offset click bounds by TARGET_W since button is duplicated on the right pane
        if bx + TARGET_W <= x <= bx + TARGET_W + bw and by <= y <= by + bh:
            super_res_on = not super_res_on

def serial_reader_thread():
    global running, right_count, wrong_count
    print(f"[Serial] Started on {COM_PORT}")
    buffer = bytearray()
    chunk_buffer = bytearray()
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
    except Exception as e:
        print(f"[Serial] Open failed: {e}")
        running = False
        return

    while running:
        try:
            data = ser.read(CHUNK_SIZE)
            if not data:
                time.sleep(0.001)
                continue
            
            chunk_buffer.extend(data)

            # --- Protocol Parsing (CC1310) ---
            while len(chunk_buffer) >= 3:
                payloadLen = chunk_buffer[0]
                if len(chunk_buffer) < payloadLen + 3:
                    break
                rssi = chunk_buffer[payloadLen + 1] - 256
                buffer.extend(chunk_buffer[1 : payloadLen + 1])
                chunk_buffer = chunk_buffer[payloadLen + 3 :]

            # --- AVIF Frame Extraction ---
            header = b'ftypavif'
            while True:
                idx = buffer.find(header)
                if idx == -1:
                    break
                
                # Locate start of the box size bytes
                start = idx - 4
                if start < 0:
                    buffer = buffer[idx:]
                    continue
                
                # Check for next header to find boundaries
                nxt = buffer.find(header, idx + len(header))
                if nxt == -1:
                    break # Wait for more data to ensure full frame has arrived
                
                # Look backwards from 'start' to see if there's a 8-byte mode flag
                is_hq = False
                has_person = False
                if start >= 8:
                    marker = buffer[start-8:start]
                    if marker == b'MODE_HQP':
                        is_hq, has_person = True, True
                    elif marker == b'MODE_HQN':
                        is_hq, has_person = True, False
                    elif marker == b'MODE_LQP':
                        is_hq, has_person = False, True
                    elif marker == b'MODE_LQN':
                        is_hq, has_person = False, False
                
                cut_idx = nxt - 4
                if buffer[nxt - 12 : nxt - 4] in (b'MODE_HQP', b'MODE_HQN', b'MODE_LQP', b'MODE_LQN'):
                    cut_idx = nxt - 12
                    
                frame_data = buffer[start:cut_idx]
                buffer = buffer[cut_idx:]

                # --- AVIF Decode ---
                try:
                    img = Image.open(io.BytesIO(frame_data))
                    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    
                    # FIX: Only resize to AI input dimensions if it's a Low Quality frame.
                    # HQ frames keep their native 640x480 resolution.
                    if not is_hq:
                        bgr = cv2.resize(bgr, (SRC_W, SRC_H))
                        
                    right_count += 1
                    if not rife_input_q.full():
                        rife_input_q.put((bgr, rssi, is_hq, has_person)) # passing mode into queue pipeline
                except Exception:
                    wrong_count += 1

        except Exception as e:
            time.sleep(0.05)

    ser.close()
    print("[Serial] Stopped")

def rife_thread(rife_ok):
    global running
    prev = None
    while running:
        try:
            frm, rs, is_hq, has_person = rife_input_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if prev is None:
            prev = (frm, rs, is_hq, has_person)
            if not display_q.full():
                display_q.put((frm, frm, rs, is_hq, has_person, True))
            continue

        if rife_ok and not is_hq and not prev[2]:
            try:
                mid = infer_rife(prev[0], frm)
                if not display_q.full():
                    # for intermediate frame, inherit has_person and raw frame from prev, and mark it as NOT a new native frame
                    display_q.put((mid, prev[0], prev[1], False, prev[3], False))
            except Exception:
                pass

        if not display_q.full():
            display_q.put((frm, frm, rs, is_hq, has_person, True))
        prev = (frm, rs, is_hq, has_person)

def main():
    global running
    print("=== Sim RX AVIF / RIFE Real-Time ===")
    sr_ok = init_sr()
    rife_ok = init_rife()

    t1 = threading.Thread(target=serial_reader_thread, daemon=True)
    t2 = threading.Thread(target=rife_thread, args=(rife_ok,), daemon=True)
    t1.start()
    t2.start()

    win = "Sim RX Stream"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback(win, on_mouse)

    TARGET_FPS = 24.0 if rife_ok else 15.0
    pace_interval = 1.0 / TARGET_FPS

    frame_times_ai = collections.deque(maxlen=30)
    frame_times_raw = collections.deque(maxlen=30)
    fps_ai = 0.0
    fps_raw = 0.0
    last_render = np.zeros((TARGET_H, TARGET_W * 2, 3), dtype=np.uint8)
    rssi = 0
    prev_time = time.perf_counter()

    is_hq = False
    has_person = False
    is_new_raw = False
    try:
        while True:
            new_frame = None
            new_ai = None
            new_raw = None
            try:
                ai, raw, rssi, frame_hq, frame_person, frame_is_new_raw = display_q.get_nowait()
                new_ai = ai
                new_raw = raw
                is_hq = frame_hq
                has_person = frame_person
                is_new_raw = frame_is_new_raw
            except queue.Empty:
                pass

            while display_q.qsize() > 17:
                try:
                    ai, raw, rssi, frame_hq, frame_person, frame_is_new_raw = display_q.get_nowait()
                    new_ai = ai
                    new_raw = raw
                    is_hq = frame_hq
                    has_person = frame_person
                    is_new_raw = frame_is_new_raw
                except queue.Empty:
                    break

            if new_ai is not None:
                ai_frame = new_ai
                raw_frame = new_raw
                if sr_ok and super_res_on and not is_hq:
                    try:
                        ai_frame = infer_sr(ai_frame)
                    except Exception:
                        pass
                
                h_ai, w_ai = ai_frame.shape[:2]
                if w_ai != TARGET_W or h_ai != TARGET_H:
                    ai_frame_up = cv2.resize(ai_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                else:
                    ai_frame_up = ai_frame

                h_r, w_r = raw_frame.shape[:2]
                if w_r != TARGET_W or h_r != TARGET_H:
                    raw_frame_up = cv2.resize(raw_frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
                else:
                    raw_frame_up = raw_frame

                frame_times_ai.append(time.perf_counter())
                if len(frame_times_ai) > 1:
                    fps_ai = (len(frame_times_ai) - 1) / max(1e-6, frame_times_ai[-1] - frame_times_ai[0])

                if is_new_raw:
                    frame_times_raw.append(time.perf_counter())
                    if len(frame_times_raw) > 1:
                        fps_raw = (len(frame_times_raw) - 1) / max(1e-6, frame_times_raw[-1] - frame_times_raw[0])
                
                # Combine horizontally side-by-side!
                last_render = np.hstack((raw_frame_up, ai_frame_up))

            render = last_render.copy()
            # Left pane text (Naive)
            cv2.putText(render, "NAIVE DECODE (NO AI)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(render, f"FPS: {fps_raw:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(render, f"RSSI: {rssi}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 80, 80), 2)
            cv2.putText(render, f"Q:{display_q.qsize()}  OK:{right_count}  ERR:{wrong_count}", (20, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)

            # Right pane text (AI)
            bx, by, bw, bh = sr_btn_area
            btn_bx = bx + TARGET_W
            btn_c = (0, 200, 0) if super_res_on else (0, 0, 200)
            if is_hq: btn_c = (200, 100, 0) # Orange bypass indicator
            cv2.rectangle(render, (btn_bx, by), (btn_bx + bw, by + bh), btn_c, -1)
            cv2.rectangle(render, (btn_bx, by), (btn_bx + bw, by + bh), (255, 255, 255), 2)
            
            mode_text = "HQ: NATIVE" if is_hq else ("SR: ON" if super_res_on else "SR: OFF")
            cv2.putText(render, mode_text, (btn_bx + 15, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
            cv2.putText(render, "AI ENHANCED (RIFE + SR)", (TARGET_W + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            cv2.putText(render, f"FPS: {fps_ai:.1f}", (TARGET_W + 20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2)

            if has_person:
                cv2.putText(render, "HUMAN DETECTED", ((TARGET_W * 2) // 2 - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            cv2.imshow(win, render)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if not t1.is_alive() or not t2.is_alive():
                break

            now = time.perf_counter()
            elapsed = now - prev_time
            sleep_t = pace_interval - elapsed
            if sleep_t > 0.002:
                time.sleep(sleep_t - 0.001)
            prev_time = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
