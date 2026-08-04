import cv2
import serial
import numpy as np
import time
import os
import threading
import queue
import io
import struct
import collections
import av
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# --- TensorRT / Torch ---
try:
    import tensorrt as trt
    import torch
    _HAS_TRT = True
except ImportError:
    print("[WARN] TensorRT/PyTorch not found. SR and RIFE disabled.")
    trt = None
    _HAS_TRT = False

import serial.tools.list_ports
def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports: return "COM4"
    for p in ports:
        if "XDS110" in p.description: return p.device
    for p in ports:
        if "USB" in p.description or "UART" in p.description: return p.device
    return ports[0].device

COM_PORT   = find_serial_port()
print(f"Auto-selected RX COM Port: {COM_PORT}")
BAUD_RATE  = 921600
CHUNK_SIZE = 1024
TIMEOUT    = 0.01

ENABLE_DSSS_MAC_PARSING = True  # Set to True if receiving via Hardware Radio that adds MAC layer bytes

AES_KEY      = b'this_is_a_32_byte_secret_key_!!!'
aes_gcm      = AESGCM(AES_KEY)
MAGIC_HEADER = b'FRIEREN'
HEADER_FORMAT = '<7sBI'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

TARGET_W, TARGET_H = 1280, 960
SRC_W, SRC_H = 320, 240 # Max source resolution (Must match TRT static dims)
RIFE_PAD_H = 256 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if _HAS_TRT:
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- RealESRGAN (Super-Resolution) ---
SR_ENGINE_PATH  = os.path.join(BASE_DIR, "realesr-general-x4v3.engine")
SR_CTX          = None
SR_STREAM       = None
SR_INPUT        = None   
SR_OUTPUT       = None   

# --- RIFE flownet (Frame Interpolation) ---
RIFE_ENGINE_PATH  = os.path.join(BASE_DIR, "flownet.engine")
RIFE_CTX          = None
RIFE_STREAM       = None
RIFE_INPUT        = None   
RIFE_TIMESTEP     = None   
RIFE_OUT_TENSORS  = {}     
RIFE_DUMMY        = []     
RIFE_OUTPUT       = None   

rife_input_q = queue.Queue(maxsize=30)
display_q    = queue.Queue(maxsize=60)

running        = True
right_count    = 0
wrong_count    = 0
super_res_on   = True
sr_btn_area    = (1050, 30, 200, 60)  
rife_on        = True
rife_btn_area  = (1050, 110, 200, 60)

_RIFE_STAGING = np.zeros((1, 6, RIFE_PAD_H, SRC_W), dtype=np.float32)

def init_sr():
    global SR_CTX, SR_STREAM, SR_INPUT, SR_OUTPUT
    if not _HAS_TRT or DEVICE.type != "cuda": return False
    if not os.path.exists(SR_ENGINE_PATH): return False
    try:
        with open(SR_ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
            engine = rt.deserialize_cuda_engine(f.read())
        if not engine: return False
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
        return True
    except Exception as e:
        print(f"[SR ] Init failed: {e}")
        return False

def infer_sr(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous()
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
                if name == "2787": RIFE_OUTPUT = t
        if RIFE_OUTPUT is None:
            for v in RIFE_OUT_TENSORS.values():
                if tuple(v.shape) == target_shape:
                    RIFE_OUTPUT = v
                    break
        return True
    except Exception as e:
        return False

def infer_rife(bgr0, bgr1):
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

def on_mouse(event, x, y, flags, param):
    global super_res_on, rife_on
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = sr_btn_area
        if bx <= x <= bx + bw and by <= y <= by + bh:
            super_res_on = not super_res_on
            
        rx, ry, rw, rh = rife_btn_area
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            rife_on = not rife_on

# Global for dynamic pacing
incoming_fps = 15.0

def serial_reader_thread():
    global running, right_count, wrong_count, incoming_fps
    buffer = bytearray()
    chunk_buffer = bytearray()
    rssi = 0
    
    rx_times = collections.deque(maxlen=15)
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
    except Exception as e:
        print(f"[Serial] Open failed: {e}")
        running = False
        return

    decoder = av.CodecContext.create('hevc', 'r')
    current_level = 4
    
    while running:
        try:
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
                
                try:
                    nonce = bytes(payload[:12])
                    ciphertext = bytes(payload[12:])
                    h265_packet_data = aes_gcm.decrypt(nonce, ciphertext, None)
                    right_count += 1
                except Exception:
                    wrong_count += 1
                    continue

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

        except Exception:
            time.sleep(0.01)

    ser.close()

def rife_thread(rife_ok):
    global running
    prev = None
    while running:
        try:
            frm, lvl, sz, rssi = rife_input_q.get(timeout=0.2)
        except queue.Empty:
            continue

        if prev is None:
            prev = (frm, lvl, sz, rssi)
            if not display_q.full(): display_q.put((frm, lvl, sz, rssi, False))
            continue

        if rife_ok and rife_on:
            mean_diff = np.mean(cv2.absdiff(prev[0], frm))
            if mean_diff < 60:
                try:
                    mid = infer_rife(prev[0], frm)
                    if not display_q.full():
                        display_q.put((mid, lvl, sz, rssi, True))
                except Exception as e:
                    print(f"[RIFE Error] {e}")

        if not display_q.full():
            display_q.put((frm, lvl, sz, rssi, False))
        prev = (frm, lvl, sz, rssi)

def main():
    global running

    sr_ok   = init_sr()
    rife_ok = init_rife()

    t1 = threading.Thread(target=serial_reader_thread, daemon=True)
    t2 = threading.Thread(target=rife_thread, args=(rife_ok,), daemon=True)
    t1.start()
    t2.start()

    win = "RX H.265 + RIFE + RealESRGAN Stream"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
    cv2.setMouseCallback(win, on_mouse)

    frame_times = collections.deque(maxlen=30)
    actual_fps  = 0.0
    last_render = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    prev_time   = time.perf_counter()
    
    current_lvl, current_sz, current_rssi = 4, 0, 0

    try:
        while True:
            new_frame = None
            is_mid = False
            try:
                frm, lvl, sz, rssi, mid = display_q.get_nowait()
                new_frame = frm
                current_lvl, current_sz, current_rssi, is_mid = lvl, sz, rssi, mid
            except queue.Empty:
                pass

            while display_q.qsize() > 17:
                try:
                    frm, lvl, sz, rssi, mid = display_q.get_nowait()
                    new_frame = frm
                    current_lvl, current_sz, current_rssi, is_mid = lvl, sz, rssi, mid
                except queue.Empty:
                    break

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

                frame_times.append(time.perf_counter())
                if len(frame_times) > 1:
                    actual_fps = (len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0])

                last_render = frame

            render = last_render.copy()
            cv2.putText(render, f"RX H.265 (PyAV)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(render, f"FPS: {actual_fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(render, f"Level: {current_lvl} | Size: {current_sz} Bytes", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
            
            status_text = f"Q:{display_q.qsize()} OK:{right_count} ERR:{wrong_count} RSSI:{current_rssi} " + ("(RIFE MID-FRAME)" if is_mid else "")
            cv2.putText(render, status_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2, cv2.LINE_AA)

            bx, by, bw, bh = sr_btn_area
            btn_c = (0, 200, 0) if super_res_on else (0, 0, 200)
            cv2.rectangle(render, (bx, by), (bx + bw, by + bh), btn_c, -1)
            cv2.rectangle(render, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
            cv2.putText(render, "SR: ON" if super_res_on else "SR: OFF", (bx + 20, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

            rx, ry, rw, rh = rife_btn_area
            rbtn_c = (0, 200, 0) if rife_on else (0, 0, 200)
            cv2.rectangle(render, (rx, ry), (rx + rw, ry + rh), rbtn_c, -1)
            cv2.rectangle(render, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)
            cv2.putText(render, "RIFE: ON" if rife_on else "RIFE: OFF", (rx + 5, ry + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

            cv2.imshow(win, render)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if not t1.is_alive() or not t2.is_alive(): break

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

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
