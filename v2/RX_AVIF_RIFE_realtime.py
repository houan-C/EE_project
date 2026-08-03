import cv2
import serial
import numpy as np
import time
import os
import threading
import queue
import io
import collections

# --- Pillow for AVIF Support ---
try:
    from PIL import Image
    import pillow_avif
except ImportError:
    print("[ERROR] pillow-avif-plugin not installed. Run: pip install pillow pillow-avif-plugin")

# --- TensorRT / Torch ---
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
COM_PORT   = "COM5"
BAUD_RATE  = 921600
CHUNK_SIZE = 1024
TIMEOUT    = 0.01

# Target display size
TARGET_W, TARGET_H = 1280, 960

# Source frame size (from hardware)
SRC_W, SRC_H = 320, 240

# RIFE engine pads height to nearest 32-multiple
RIFE_PAD_H = 256  # 240 padded up to 256

# ====================================================================
# === TRT GLOBALS ===
# ====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if _HAS_TRT:
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RealESRGAN (Super-Resolution) ---
SR_ENGINE_PATH  = os.path.join(BASE_DIR, "realesr-general-x4v3.engine")
SR_CTX          = None
SR_STREAM       = None
SR_INPUT        = None   # (1,3,240,320) float32 CUDA
SR_OUTPUT       = None   # (1,3,960,1280) float32 CUDA

# --- RIFE flownet (Frame Interpolation) ---
RIFE_ENGINE_PATH  = os.path.join(BASE_DIR, "flownet.engine")
RIFE_CTX          = None
RIFE_STREAM       = None
RIFE_INPUT        = None   # (1,6,256,320) float32 CUDA
RIFE_TIMESTEP     = None   # (1,) float32 CUDA
RIFE_OUT_TENSORS  = {}     # name -> tensor  (keeps all alive for TRT)
RIFE_DUMMY        = []     # int64 constants (kept alive)
RIFE_OUTPUT       = None   # (1,3,256,320) final RGB tensor

# ====================================================================
# === QUEUES & STATE ===
# ====================================================================
rife_input_q = queue.Queue(maxsize=30)
display_q    = queue.Queue(maxsize=60)

running        = True
right_count    = 0
wrong_count    = 0
super_res_on   = True
sr_btn_area    = (1050, 30, 200, 60)  # x,y,w,h

# Pre-allocated numpy staging buffer for RIFE input (avoids per-frame malloc)
_RIFE_STAGING = np.zeros((1, 6, RIFE_PAD_H, SRC_W), dtype=np.float32)


# ====================================================================
# === INIT: RealESRGAN TRT ===
# ====================================================================
def init_sr():
    global SR_CTX, SR_STREAM, SR_INPUT, SR_OUTPUT
    if not _HAS_TRT or DEVICE.type != "cuda":
        return False
    if not os.path.exists(SR_ENGINE_PATH):
        print(f"[SR] Engine not found: {SR_ENGINE_PATH}")
        return False
    try:
        with open(SR_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine:
            return False
        SR_CTX    = engine.create_execution_context()
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
                dt    = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                SR_OUTPUT = torch.zeros(shape, dtype=dt, device=DEVICE).contiguous()
                SR_CTX.set_tensor_address(name, int(SR_OUTPUT.data_ptr()))
        print(f"[SR ] Initialized  ({SRC_W}x{SRC_H} -> {TARGET_W}x{TARGET_H})")
        return True
    except Exception as e:
        print(f"[SR ] Init failed: {e}")
        return False


def infer_sr(bgr_320x240):
    """Returns 1280x960 BGR uint8"""
    rgb = cv2.cvtColor(bgr_320x240, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    SR_INPUT.copy_(t)
    SR_CTX.execute_async_v3(stream_handle=SR_STREAM.cuda_stream)
    SR_STREAM.synchronize()
    out = SR_OUTPUT.float().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return cv2.cvtColor((out.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ====================================================================
# === INIT: RIFE TRT ===
# ====================================================================
def init_rife():
    global RIFE_CTX, RIFE_STREAM, RIFE_INPUT, RIFE_TIMESTEP
    global RIFE_OUT_TENSORS, RIFE_DUMMY, RIFE_OUTPUT
    if not _HAS_TRT or DEVICE.type != "cuda":
        return False
    if not os.path.exists(RIFE_ENGINE_PATH):
        print(f"[RIFE] Engine not found: {RIFE_ENGINE_PATH}")
        return False
    try:
        with open(RIFE_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine:
            return False
        RIFE_CTX    = engine.create_execution_context()
        RIFE_STREAM = torch.cuda.Stream()

        inp_shape = (1, 6, RIFE_PAD_H, SRC_W)
        ts_shape  = (1,)

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
                    d  = torch.ones(sh, dtype=torch.int64, device=DEVICE).contiguous()
                    RIFE_DUMMY.append(d)
                    RIFE_CTX.set_tensor_address(name, int(d.data_ptr()))

        RIFE_CTX.infer_shapes()

        target_shape = (1, 3, RIFE_PAD_H, SRC_W)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(RIFE_CTX.get_tensor_shape(name))
                dt    = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                t     = torch.zeros(shape, dtype=dt, device=DEVICE).contiguous()
                RIFE_CTX.set_tensor_address(name, int(t.data_ptr()))
                RIFE_OUT_TENSORS[name] = t
                if name == "2787":
                    RIFE_OUTPUT = t

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
    """
    Interpolates a mid-frame between bgr0 and bgr1 (both SRC_W x SRC_H BGR uint8).
    Uses the pre-allocated _RIFE_STAGING buffer to avoid per-call malloc.
    Returns SRC_W x SRC_H BGR uint8.
    """
    # Pad height from 240 -> 256
    # (pad bottom only, BORDER_REPLICATE avoids color discontinuity)
    pad = RIFE_PAD_H - SRC_H  # 16 pixels

    # --- numpy HWC -> CHW float32 directly into staging ---
    for ch in range(3):
        _RIFE_STAGING[0, ch,    :SRC_H, :] = bgr0[:, :, 2 - ch] / 255.0   # BGR->RGB
        _RIFE_STAGING[0, ch,   SRC_H:, :] = _RIFE_STAGING[0, ch, SRC_H-1:SRC_H, :]
        _RIFE_STAGING[0, ch+3, :SRC_H, :] = bgr1[:, :, 2 - ch] / 255.0
        _RIFE_STAGING[0, ch+3, SRC_H:, :] = _RIFE_STAGING[0, ch+3, SRC_H-1:SRC_H, :]

    RIFE_INPUT.copy_(torch.from_numpy(_RIFE_STAGING))
    RIFE_CTX.execute_async_v3(stream_handle=RIFE_STREAM.cuda_stream)
    RIFE_STREAM.synchronize()

    out = RIFE_OUTPUT.float().squeeze(0).permute(1, 2, 0)[:SRC_H].clamp(0, 1)
    # RGB -> BGR
    out_np = out.cpu().numpy()
    return cv2.cvtColor((out_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# ====================================================================
# === UI ===
# ====================================================================
def on_mouse(event, x, y, flags, param):
    global super_res_on
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = sr_btn_area
        if bx <= x <= bx + bw and by <= y <= by + bh:
            super_res_on = not super_res_on
            print(f"[UI] Super-Resolution: {'ON' if super_res_on else 'OFF'}")


# ====================================================================
# === THREAD 1: Serial Reader ===
# ====================================================================
def serial_reader_thread():
    global running, right_count, wrong_count
    print(f"[Serial] Started on {COM_PORT}")
    buffer = bytearray()
    chunk  = bytearray()

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
            chunk.extend(data)

            # --- Protocol Parsing ---
            while len(chunk) >= 3:
                payloadLen = chunk[0]
                if len(chunk) < payloadLen + 3:
                    break
                rssi = chunk[payloadLen + 1] - 256
                buffer.extend(chunk[1: payloadLen + 1])
                chunk = chunk[payloadLen + 3:]

            # --- AVIF Frame Extraction ---
            header = b'ftypavif'
            while True:
                idx = buffer.find(header)
                if idx == -1:
                    break
                start = idx - 4
                if start > 0:
                    buffer = buffer[start:]
                    idx = 4
                nxt = buffer.find(header, idx + len(header))
                if nxt == -1:
                    break
                frame_data = buffer[:nxt - 4]
                buffer     = buffer[nxt - 4:]

                # --- AVIF Decode ---
                try:
                    img  = Image.open(io.BytesIO(frame_data))
                    bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    bgr  = cv2.resize(bgr, (SRC_W, SRC_H))
                    right_count += 1
                    if not rife_input_q.full():
                        rife_input_q.put((bgr, rssi))
                except Exception:
                    wrong_count += 1
                    # NOTE: decode failures are counted but not printed to avoid I/O cost.
                    # To enable: uncomment below (may reduce throughput on bad signal)
                    # print(f"[Serial] Decode failed (total={wrong_count})")

        except Exception:
            time.sleep(0.05)

    ser.close()
    print("[Serial] Stopped")


# ====================================================================
# === THREAD 2: RIFE Interpolator ===
# ====================================================================
def rife_thread(rife_ok):
    global running
    print(f"[RIFE] Thread started (interpolation={'ON' if rife_ok else 'OFF'})")
    prev = None

    while running:
        try:
            frm, rs = rife_input_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if prev is None:
            prev = (frm, rs)
            if not display_q.full():
                display_q.put((frm, rs))
            continue

        # --- Interpolate midpoint frame ---
        if rife_ok:
            try:
                mid = infer_rife(prev[0], frm)
                if not display_q.full():
                    display_q.put((mid, prev[1]))
            except Exception as e:
                print(f"[RIFE] Inference error: {e}")

        # --- Push current frame ---
        if not display_q.full():
            display_q.put((frm, rs))
        prev = (frm, rs)

    print("[RIFE] Thread stopped")


# ====================================================================
# === MAIN: Display Loop ===
# ====================================================================
def main():
    global running

    print("=== RX AVIF / RIFE Real-Time ===")
    sr_ok   = init_sr()
    rife_ok = init_rife()

    if not rife_ok:
        print("[WARN] RIFE disabled. Displaying raw frames only.")

    t1 = threading.Thread(target=serial_reader_thread, daemon=True)
    t2 = threading.Thread(target=rife_thread, args=(rife_ok,), daemon=True)
    t1.start()
    t2.start()

    win = "RX AVIF + RIFE Stream"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    # Target playback fps. With RIFE ON: source ~12fps * 2 = 24fps target.
    TARGET_FPS    = 24.0 if rife_ok else 15.0
    pace_interval = 1.0 / TARGET_FPS

    frame_times = collections.deque(maxlen=30)
    actual_fps  = 0.0
    last_render = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    rssi        = 0
    prev_time   = time.perf_counter()

    print(f"[Main] Display loop at target {TARGET_FPS:.0f} FPS")

    try:
        while True:
            # === Check for new frame (NON-BLOCKING) ===
            new_frame = None
            try:
                proc_frame, rssi = display_q.get_nowait()
                new_frame = proc_frame
            except queue.Empty:
                pass

            # === Drain excess to respect <700ms latency ===
            # At 24fps, 17 frames ≈ 700ms
            while display_q.qsize() > 17:
                try:
                    proc_frame, rssi = display_q.get_nowait()
                    new_frame = proc_frame  # always keep the newest
                except queue.Empty:
                    break

            # === Process frame if we have a new one ===
            if new_frame is not None:
                frame = new_frame
                if sr_ok and super_res_on:
                    try:
                        frame = infer_sr(frame)
                    except Exception:
                        pass
                h, w = frame.shape[:2]
                if w != TARGET_W or h != TARGET_H:
                    frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

                # FPS counter
                frame_times.append(time.perf_counter())
                if len(frame_times) > 1:
                    actual_fps = (len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0])

                last_render = frame

            # === Draw HUD on every frame (prevents flickering) ===
            render = last_render.copy()
            cv2.putText(render, f"FPS: {actual_fps:.1f}",
                        (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(render, f"RSSI: {rssi}",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 80, 80), 2)
            cv2.putText(render, f"Q:{display_q.qsize()}  OK:{right_count}  ERR:{wrong_count}",
                        (20, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 1)

            bx, by, bw, bh = sr_btn_area
            btn_c = (0, 200, 0) if super_res_on else (0, 0, 200)
            cv2.rectangle(render, (bx, by), (bx + bw, by + bh), btn_c, -1)
            cv2.rectangle(render, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            cv2.putText(render, "SR: ON" if super_res_on else "SR: OFF",
                        (bx + 20, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

            cv2.imshow(win, render)

            # === Pump events & check quit ===
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if not t1.is_alive() or not t2.is_alive():
                print("[Main] Worker thread died.")
                break

            # === Precise pace to TARGET_FPS ===
            now     = time.perf_counter()
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
        print(f"\n=== Done  |  OK={right_count}  ERR={wrong_count} ===")


if __name__ == "__main__":
    main()
