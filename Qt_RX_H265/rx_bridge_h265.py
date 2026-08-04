"""
rx_bridge_h265.py  —  RX Bridge for Qt (H.265)

Protocol matches TX_H265.py:
  - DSSS MAC Layer (optional)
  - AES-GCM encryption layer  (MAGIC_HEADER = b'FRIEREN')
  - H.265 NALU stream decoding via PyAV

AI Upscaling:
  - RealESRGAN  (realesr-general-x4v3.engine)
  - RIFE        (flownet.engine)

stdin commands from Qt:
  SR_ON / SR_OFF
  RIFE_ON / RIFE_OFF
  QUIT

stdout binary stream to Qt:
  SyncHeader ('<4sIIfiiiiii') + JPEG frame bytes
"""

import cv2
import serial
import numpy as np
import time
import os
import warnings
import sys
import threading
import queue
import collections
import struct

warnings.filterwarnings("ignore")

# ── PyAV (H.265) ──────────────────────────────────────────────────────────────
try:
    import av
    _HAS_AV = True
except ImportError:
    _HAS_AV = False
    print("[WARN] PyAV not installed — H.265 decode disabled.", flush=True)

# ── AES-GCM ───────────────────────────────────────────────────────────────────
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    AES_KEY  = b'this_is_a_32_byte_secret_key_!!!'
    aes_gcm  = AESGCM(AES_KEY)
    _HAS_AES = True
    print("[OK] AES-GCM ready.", flush=True)
except ImportError:
    _HAS_AES = False
    aes_gcm  = None
    print("[WARN] cryptography not installed — AES-GCM disabled (plaintext fallback).", flush=True)

# ── TensorRT / Torch ──────────────────────────────────────────────────────────
try:
    import tensorrt as trt
    import torch
    _HAS_TRT = True
except ImportError:
    trt   = None
    torch = None
    _HAS_TRT = False
    print("[WARN] TensorRT/PyTorch not found. SR and RIFE disabled.", flush=True)

# ==============================================================================
#  CONFIGURATION  (argv: python rx_bridge_h265.py <COM_PORT>)
# ==============================================================================
COM_PORT   = sys.argv[1] if len(sys.argv) > 1 else "COM5"
BAUD_RATE  = 921600
CHUNK_SIZE = 1024
TIMEOUT    = 0.01

ENABLE_DSSS_MAC_PARSING = True  # Matches TX_H265.py

MAGIC_HEADER   = b'FRIEREN'   # AES-GCM packet boundary
HEADER_FORMAT  = '<7sBI'      # magic (7), pkt_level (1), payload_len (4)
HEADER_SIZE    = struct.calcsize(HEADER_FORMAT)

TARGET_W, TARGET_H = 1280, 960
SRC_W,    SRC_H    = 320,  240
RIFE_PAD_H         = 256   # 240 → nearest 32-multiple

# ==============================================================================
#  GLOBALS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SR_ENGINE_PATH   = os.path.join(BASE_DIR, "realesr-general-x4v3.engine")
RIFE_ENGINE_PATH = os.path.join(BASE_DIR, "flownet.engine")

# TRT handles — set by background init thread
SR_CTX           = None
SR_STREAM        = None
SR_INPUT         = None
SR_OUTPUT        = None
SR_READY         = threading.Event()

RIFE_CTX         = None
RIFE_STREAM      = None
RIFE_INPUT       = None
RIFE_TIMESTEP    = None
RIFE_OUT_TENSORS = {}
RIFE_DUMMY       = []
RIFE_OUTPUT      = None
RIFE_READY       = threading.Event()

sr_ok   = False
rife_ok = False

if _HAS_TRT:
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    TRT_LOGGER = None
    DEVICE     = None

# Pre-allocated RIFE staging buffer
_RIFE_STAGING = np.zeros((1, 6, RIFE_PAD_H, SRC_W), dtype=np.float32)

# Queues
rife_input_q = queue.Queue(maxsize=30)
display_q    = queue.Queue(maxsize=60)

# State flags
running      = True
super_res_on = True
rife_on      = True
right_count  = 0
wrong_count  = 0
incoming_fps = 15.0

# ==============================================================================
#  INIT: RealESRGAN TRT  (runs in background thread)
# ==============================================================================
def init_sr():
    global SR_CTX, SR_STREAM, SR_INPUT, SR_OUTPUT, sr_ok
    if not _HAS_TRT or DEVICE.type != "cuda":
        SR_READY.set()
        return
    if not os.path.exists(SR_ENGINE_PATH):
        print(f"[SR ] Engine not found: {SR_ENGINE_PATH}", flush=True)
        SR_READY.set()
        return
    try:
        print("[SR ] Loading engine...", flush=True)
        with open(SR_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine:
            SR_READY.set()
            return
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
        sr_ok = True
        print(f"[SR ] Ready ({SRC_W}x{SRC_H} → {TARGET_W}x{TARGET_H})", flush=True)
    except Exception as e:
        print(f"[SR ] Init failed: {e}", flush=True)
    finally:
        SR_READY.set()

def infer_sr(bgr_320x240):
    """Returns TARGET_W x TARGET_H BGR uint8."""
    rgb = cv2.cvtColor(bgr_320x240, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
    SR_INPUT.copy_(t)
    SR_CTX.execute_async_v3(stream_handle=SR_STREAM.cuda_stream)
    SR_STREAM.synchronize()
    out = SR_OUTPUT.float().squeeze(0).permute(1, 2, 0).clamp(0, 1)
    return cv2.cvtColor((out.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# ==============================================================================
#  INIT: RIFE TRT  (runs in background thread)
# ==============================================================================
def init_rife():
    global RIFE_CTX, RIFE_STREAM, RIFE_INPUT, RIFE_TIMESTEP
    global RIFE_OUT_TENSORS, RIFE_DUMMY, RIFE_OUTPUT, rife_ok
    if not _HAS_TRT or DEVICE.type != "cuda":
        RIFE_READY.set()
        return
    if not os.path.exists(RIFE_ENGINE_PATH):
        print(f"[RIFE] Engine not found: {RIFE_ENGINE_PATH}", flush=True)
        RIFE_READY.set()
        return
    try:
        print("[RIFE] Loading engine...", flush=True)
        with open(RIFE_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine:
            RIFE_READY.set()
            return
        RIFE_CTX    = engine.create_execution_context()
        RIFE_STREAM = torch.cuda.Stream()
        inp_shape   = (1, 6, RIFE_PAD_H, SRC_W)
        ts_shape    = (1,)

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

        rife_ok = True
        print(f"[RIFE] Ready ({SRC_W}x{SRC_H} padded to {SRC_W}x{RIFE_PAD_H})", flush=True)
    except Exception as e:
        print(f"[RIFE] Init failed: {e}", flush=True)
    finally:
        RIFE_READY.set()

def infer_rife(bgr0, bgr1):
    """Interpolates mid-frame. Input/output: SRC_W x SRC_H BGR uint8."""
    for ch in range(3):
        _RIFE_STAGING[0, ch,    :SRC_H, :] = bgr0[:, :, 2 - ch] / 255.0
        _RIFE_STAGING[0, ch,   SRC_H:, :] = _RIFE_STAGING[0, ch, SRC_H-1:SRC_H, :]
        _RIFE_STAGING[0, ch+3, :SRC_H, :] = bgr1[:, :, 2 - ch] / 255.0
        _RIFE_STAGING[0, ch+3, SRC_H:, :] = _RIFE_STAGING[0, ch+3, SRC_H-1:SRC_H, :]

    RIFE_INPUT.copy_(torch.from_numpy(_RIFE_STAGING))
    RIFE_CTX.execute_async_v3(stream_handle=RIFE_STREAM.cuda_stream)
    RIFE_STREAM.synchronize()

    out = RIFE_OUTPUT.float().squeeze(0).permute(1, 2, 0)[:SRC_H].clamp(0, 1)
    return cv2.cvtColor((out.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# ==============================================================================
#  THREAD 1: Serial Reader  (AES-GCM decrypt + H.265 Decode)
# ==============================================================================
def serial_reader_thread():
    global running, right_count, wrong_count, incoming_fps

    buffer       = bytearray()
    chunk_buffer = bytearray()
    rssi         = 0
    
    rx_times = collections.deque(maxlen=15)
    
    # Wait for PyAV
    if not _HAS_AV:
        print("[Error] PyAV not available.", flush=True)
        return

    decoder = av.CodecContext.create('hevc', 'r')
    current_level = 4

    while running:
        ser = None
        while running and ser is None:
            try:
                ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
                print(f"[Serial] Opened {COM_PORT}", flush=True)
            except Exception as e:
                print(f"[Serial] Cannot open {COM_PORT}: {e}  (retrying in 2s)", flush=True)
                time.sleep(2.0)

        if not running:
            break

        try:
            while running:
                data = ser.read(CHUNK_SIZE)
                if not data:
                    time.sleep(0.001)
                    continue

                if ENABLE_DSSS_MAC_PARSING:
                    chunk_buffer.extend(data)
                    while len(chunk_buffer) >= 3:
                        payloadLen = chunk_buffer[0]
                        if len(chunk_buffer) < payloadLen + 3:
                            break
                        rssi = chunk_buffer[payloadLen + 1] - 256
                        buffer.extend(chunk_buffer[1: payloadLen + 1])
                        chunk_buffer = chunk_buffer[payloadLen + 3:]
                else:
                    buffer.extend(data)

                while True:
                    start_idx = buffer.find(MAGIC_HEADER)
                    if start_idx == -1:
                        if len(buffer) > 2048: buffer = buffer[-512:]
                        break

                    if len(buffer) < start_idx + HEADER_SIZE:
                        break

                    unpacked = struct.unpack(HEADER_FORMAT, buffer[start_idx : start_idx + HEADER_SIZE])
                    magic, pkt_level, payload_len = unpacked

                    if payload_len > 2_000_000:
                        buffer = buffer[start_idx + 4:]
                        wrong_count += 1
                        continue

                    if len(buffer) < start_idx + HEADER_SIZE + payload_len:
                        break

                    payload = buffer[start_idx + HEADER_SIZE : start_idx + HEADER_SIZE + payload_len]
                    buffer = buffer[start_idx + HEADER_SIZE + payload_len:]
                    
                    if _HAS_AES:
                        try:
                            nonce = bytes(payload[:12])
                            ciphertext = bytes(payload[12:])
                            h265_packet_data = aes_gcm.decrypt(nonce, ciphertext, None)
                            right_count += 1
                        except Exception:
                            wrong_count += 1
                            continue
                    else:
                        h265_packet_data = payload
                        right_count += 1

                    if pkt_level != current_level:
                        current_level = pkt_level
                        decoder = av.CodecContext.create('hevc', 'r')

                    try:
                        packets = decoder.parse(h265_packet_data)
                        for packet in packets:
                            frames = decoder.decode(packet)
                            for frame in frames:
                                img = frame.to_ndarray(format='bgr24')
                                if img.shape[1] != SRC_W or img.shape[0] != SRC_H:
                                    img = cv2.resize(img, (SRC_W, SRC_H), interpolation=cv2.INTER_LINEAR)
                                if not rife_input_q.full():
                                    rife_input_q.put((img, pkt_level, len(h265_packet_data), rssi))
                                    
                                    # Track arrival rate for dynamic pacing
                                    rx_times.append(time.perf_counter())
                                    if len(rx_times) > 1:
                                        elapsed = rx_times[-1] - rx_times[0]
                                        if elapsed > 0:
                                            incoming_fps = (len(rx_times) - 1) / elapsed
                    except Exception as e:
                        print(f"[Decode Error] {e}", flush=True)

        except Exception as e:
            print(f"[Serial] Error: {e}. Reconnecting...", flush=True)
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(1.0)

    try:
        if ser:
            ser.close()
    except Exception:
        pass
    print("[Serial] Thread stopped", flush=True)

# ==============================================================================
#  THREAD 2: RIFE Interpolator
# ==============================================================================
def rife_thread_func():
    global running
    prev = None

    while running:
        try:
            frm, lvl, sz, rs = rife_input_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if prev is None:
            prev = (frm, lvl, sz, rs)
            if not display_q.full(): display_q.put((frm, rs))
            continue

        if rife_ok and rife_on and RIFE_READY.is_set():
            mean_diff = np.mean(cv2.absdiff(prev[0], frm))
            if mean_diff < 60:
                try:
                    mid = infer_rife(prev[0], frm)
                    if not display_q.full():
                        display_q.put((mid, prev[3]))
                except Exception as e:
                    print(f"[RIFE] Inference error: {e}", flush=True)

        if not display_q.full():
            display_q.put((frm, rs))
        prev = (frm, lvl, sz, rs)

# ==============================================================================
#  STDIN THREAD: reads control commands from Qt
# ==============================================================================
def stdin_thread():
    global running, super_res_on, rife_on
    while running:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if line == "SR_ON":
                super_res_on = True
            elif line == "SR_OFF":
                super_res_on = False
            elif line == "RIFE_ON":
                rife_on = True
            elif line == "RIFE_OFF":
                rife_on = False
            elif line == "QUIT":
                running = False
                break
        except Exception:
            pass

# ==============================================================================
#  SYNC HEADER  (must match SyncHeader in mainwindow.cpp)
#  '<4sIIfiiiiii'  = 40 bytes
# ==============================================================================
SYNC_FORMAT = '<4sIIfiiiiii'
SYNC_SIZE   = struct.calcsize(SYNC_FORMAT)

def _make_init_frame(phase):
    """Returns a (TARGET_H, TARGET_W, 3) BGR frame showing 'initializing' animation."""
    frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    bar_h = 4
    bar_y = TARGET_H // 2 - bar_h // 2
    bar_w = int((0.5 + 0.5 * np.sin(phase)) * TARGET_W * 0.6) + 1
    bar_x = (TARGET_W - bar_w) // 2
    frame[bar_y: bar_y + bar_h, bar_x: bar_x + bar_w] = [88, 166, 255]
    cv2.putText(frame, "INITIALIZING AI ENGINES...", (TARGET_W // 2 - 250, TARGET_H // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
    return frame

def main():
    global running

    if sys.platform == "win32":
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    print(f"=== RX Bridge H265 | COM:{COM_PORT} | AES:{_HAS_AES} | TRT:{_HAS_TRT} ===", flush=True)

    t_sr   = threading.Thread(target=init_sr,   daemon=True)
    t_rife = threading.Thread(target=init_rife, daemon=True)
    t_sr.start()
    t_rife.start()

    t_serial = threading.Thread(target=serial_reader_thread, daemon=True)
    t_rife_w = threading.Thread(target=rife_thread_func,    daemon=True)
    t_stdin  = threading.Thread(target=stdin_thread,         daemon=True)
    t_serial.start()
    t_rife_w.start()
    t_stdin.start()

    frame_times  = collections.deque(maxlen=30)
    actual_fps   = 0.0
    last_frame   = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    rssi         = 0
    prev_time    = time.perf_counter()
    init_phase   = 0.0

    print("[Main] Display loop started", flush=True)

    while running:
        new_frame = None
        try:
            proc_frame, rssi = display_q.get_nowait()
            new_frame = proc_frame
        except queue.Empty:
            pass

        while display_q.qsize() > 17:
            try:
                proc_frame, rssi = display_q.get_nowait()
                new_frame = proc_frame
            except queue.Empty:
                break

        if new_frame is not None:
            frame = new_frame

            if sr_ok and super_res_on and SR_READY.is_set():
                try:
                    frame = infer_sr(frame)
                except Exception:
                    pass

            h, w = frame.shape[:2]
            if w != TARGET_W or h != TARGET_H:
                frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)

            frame_times.append(time.perf_counter())
            if len(frame_times) > 1:
                actual_fps = (len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0])

            last_frame = frame
        else:
            both_ready = SR_READY.is_set() and RIFE_READY.is_set()
            if not both_ready:
                init_phase += 0.15
                last_frame = _make_init_frame(init_phase)

        success, encimg = cv2.imencode('.jpg', last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            jpeg_bytes = encimg.tobytes()
            jpeg_len = len(jpeg_bytes)
        else:
            jpeg_bytes = b''
            jpeg_len = 0

        header = struct.pack(
            SYNC_FORMAT,
            b'SYNC',
            0,
            jpeg_len,
            float(actual_fps),
            int(rssi),
            int(super_res_on),
            int(right_count),
            int(wrong_count),
            int(rife_on),
            int(rife_ok),
        )
        try:
            sys.stdout.buffer.write(header)
            if jpeg_len > 0:
                sys.stdout.buffer.write(jpeg_bytes)
            sys.stdout.buffer.flush()
        except Exception:
            running = False
            break

        # Dynamic pacing
        base_fps = max(5.0, incoming_fps)
        target_fps = base_fps * 2.0 if (rife_ok and rife_on) else base_fps
        pace_interval = 1.0 / target_fps

        now     = time.perf_counter()
        elapsed = now - prev_time
        sleep_t = pace_interval - elapsed
        if sleep_t > 0.002:
            time.sleep(sleep_t - 0.001)
        prev_time = time.perf_counter()

    print("[Bridge] Exiting.", flush=True)

if __name__ == "__main__":
    main()
