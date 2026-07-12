import sys
sys.stdout.reconfigure(line_buffering=True)
"""
rx_reed.py - AVIF receiver with Reed-Solomon FEC + optional TRT Super-Resolution
==================================================================================
Matches tx_reed.py protocol:
  [ MAGIC(4B) | frame_idx(4B) | total_blocks(2B) | block_idx(2B) | rs_data(255B) ]
  = 267 bytes per packet

Firmware framing (CC1310 radio board) - confirmed by sniffer:
  Each radio packet arriving on serial is wrapped as:
  [ payloadLen(1B) | payload(payloadLen B) | RSSI(1B) | 0x00(1B) ]
  The inner payload bytes are reassembled into a stream, then RS packets are parsed.

Reed-Solomon: 223 data + 32 parity = 255 bytes/block. Corrects up to 16 byte-errors.

Buttons / keys:
  SR button  / S key  - toggle TensorRT super-resolution
  CORRUPT button / C key - inject 3 corrupt bytes before RS decode every ~30 frames
  Q / ESC             - quit
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
import io
import struct
import random
from collections import defaultdict
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import pillow_avif
except ImportError:
    print("WARNING: pillow_avif not found.")

try:
    import reedsolo
    RSCodec = reedsolo.RSCodec
except ImportError:
    print("ERROR: reedsolo not found. Install: pip install reedsolo")
    sys.exit(1)

try:
    import tensorrt as trt
    import torch
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT not found - SR disabled.")

# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------
COM_PORT   = "COM4"
BAUD_RATE  = 921600
TIMEOUT    = 0.01       # slightly longer timeout for read()
CHUNK_SIZE = 1024

RS_ECC_BYTES  = 32
RS_DATA_BYTES = 255 - RS_ECC_BYTES   # 223
RS_MAGIC      = b'RSAV'
PACKET_SIZE   = 4 + 4 + 2 + 2 + 255   # 267 bytes

ENGINE_INPUT_W, ENGINE_INPUT_H = 320, 240
ENGINE_PATH = "RealESRGAN_x4plus_fp16_NEW.engine"


rsc = RSCodec(RS_ECC_BYTES)

# ----------------------------------------------------------------
# TRT globals
# ----------------------------------------------------------------
TRT_CONTEXT = TRT_STREAM = TRT_INPUT_TENSOR = TRT_OUTPUT_TENSOR = None
DEVICE = None

if TRT_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def init_trt():
    global TRT_CONTEXT, TRT_STREAM, TRT_INPUT_TENSOR, TRT_OUTPUT_TENSOR
    if not TRT_AVAILABLE or DEVICE.type != 'cuda':
        return False
    if not os.path.exists(ENGINE_PATH):
        print(f"[RX] Engine not found: {ENGINE_PATH}")
        return False
    try:
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        if not engine:
            return False
        TRT_CONTEXT = engine.create_execution_context()
        TRT_STREAM  = torch.cuda.Stream()
        inp_shape   = (1, 3, ENGINE_INPUT_H, ENGINE_INPUT_W)
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                TRT_CONTEXT.set_input_shape(name, inp_shape)
                TRT_INPUT_TENSOR = torch.zeros(inp_shape, dtype=torch.float32, device=DEVICE).contiguous()
                TRT_CONTEXT.set_tensor_address(name, int(TRT_INPUT_TENSOR.data_ptr()))
        TRT_CONTEXT.infer_shapes()
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                out_shape = tuple(TRT_CONTEXT.get_tensor_shape(name))
                dtype = torch.float16 if engine.get_tensor_dtype(name) == trt.DataType.HALF else torch.float32
                TRT_OUTPUT_TENSOR = torch.zeros(out_shape, dtype=dtype, device=DEVICE).contiguous()
                TRT_CONTEXT.set_tensor_address(name, int(TRT_OUTPUT_TENSOR.data_ptr()))
        print(f"[RX] TRT Initialized ({ENGINE_INPUT_W}x{ENGINE_INPUT_H})")
        return True
    except Exception as e:
        print(f"[RX] TRT Init Error: {e}")
        return False


def infer_trt(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if w != ENGINE_INPUT_W or h != ENGINE_INPUT_H:
        frame_bgr = cv2.resize(frame_bgr, (ENGINE_INPUT_W, ENGINE_INPUT_H))
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp_np = np.array(rgb, dtype=np.float32) / 255.0
    inp_np = np.transpose(inp_np, (2, 0, 1))
    inp_np = np.expand_dims(inp_np, 0)
    TRT_INPUT_TENSOR.copy_(torch.from_numpy(inp_np))
    TRT_CONTEXT.execute_async_v3(stream_handle=TRT_STREAM.cuda_stream)
    TRT_STREAM.synchronize()
    out_np = TRT_OUTPUT_TENSOR.float().cpu().numpy()
    out_np = np.squeeze(out_np, 0)
    out_np = np.transpose(out_np, (1, 2, 0))
    out_np = np.clip(out_np, 0, 1)
    out_np = (out_np * 255.0).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------
# UI globals
# ----------------------------------------------------------------
frame_queue       = queue.Queue(maxsize=2)
running           = True
super_res_enabled = True
right_count    = 0
wrong_count    = 0
rs_corrections = 0
last_rssi      = 0

inference_ms      = 0.0

BTN_SR      = (500, 10, 120, 38)


_mouse_click = None

def on_mouse(event, x, y, flags, param):
    global _mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click = (x, y)


def draw_info_box(img, lines, x=10, y=10):
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.48, 1
    pad, lh = 6, 20
    bw  = 290
    bh  = lh * len(lines) + pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+bw, y+bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x, y), (x+bw, y+3), (100, 200, 100), -1)
    for i, line in enumerate(lines):
        ly = y + pad + lh * (i+1)
        if ':' in line:
            lbl, _, val = line.partition(':')
            cv2.putText(img, lbl+':', (x+pad, ly), font, fs, (140, 200, 140), th)
            lw = cv2.getTextSize(lbl+': ', font, fs, th)[0][0]
            cv2.putText(img, val.strip(), (x+pad+lw, ly), font, fs, (230, 255, 230), th+1)
        else:
            cv2.putText(img, line, (x+pad, ly), font, fs, (255, 255, 255), th)


def draw_button(img, rect, label, active, on_color=(0, 200, 0), off_color=(0, 0, 200)):
    bx, by, bw, bh = rect
    c = on_color if active else off_color
    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), c, -1)
    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255, 255, 255), 1)
    cv2.putText(img, label, (bx+8, by+25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


# ----------------------------------------------------------------
# RS decode
# ----------------------------------------------------------------
def rs_decode_blocks(blocks_data, total):
    """Returns (avif_bytes, num_corrections) or (None, 0) on failure."""
    corrections = 0
    decoded_parts = []
    for bidx in range(total):
        encoded = blocks_data.get(bidx)
        if encoded is None:
            return None, 0
        try:
            result = rsc.decode(bytes(encoded))
            decoded_data = bytes(result[0])
            errata       = result[2]
            if errata:
                corrections += len(errata)
        except reedsolo.ReedSolomonError:
            return None, 0
        decoded_parts.append(decoded_data[:RS_DATA_BYTES])

    raw = b''.join(decoded_parts)
    return raw, corrections


# ----------------------------------------------------------------
# Serial reader thread
# ----------------------------------------------------------------
def serial_reader_thread():
    global running, right_count, wrong_count, rs_corrections, last_rssi

    print(f"[RX] Thread started (Serial: {COM_PORT})", flush=True)

    raw_chunk     = bytearray()
    buf           = bytearray()
    frame_buffers = defaultdict(lambda: {'total': 0, 'blocks': {}})

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
    except Exception as e:
        print(f"[RX] ERROR Cannot open serial port: {e}", flush=True)
        running = False
        return

    while running:
        try:
            data = ser.read(CHUNK_SIZE)
            if data:
                raw_chunk.extend(data)
        except Exception as e:
            print(f"[RX] Serial read error: {e}", flush=True)
            time.sleep(0.1)
            continue

        # ---- Layer 1: Firmware unwrap (exact RX_no_sr.py logic) ----
        # Format: [payloadLen(1B)] [payload(payloadLen B)] [RSSI(1B)] [0x00(1B)]
        while len(raw_chunk) >= 3:
            plen = raw_chunk[0]
            if plen > 255:
                raw_chunk = raw_chunk[1:]
                continue
            if len(raw_chunk) < plen + 3:
                break
            payload   = bytes(raw_chunk[1 : plen + 1])
            rssi_raw  = raw_chunk[plen + 1]
            last_rssi = rssi_raw - 256
            buf.extend(payload)
            raw_chunk = raw_chunk[plen + 3:]

        # ---- Layer 2: Parse RS packets from buf via RSAV magic ----
        while True:
            magic_pos = buf.find(RS_MAGIC)
            if magic_pos == -1:
                # Keep tail in case magic straddles a read boundary
                if len(buf) > PACKET_SIZE:
                    buf = buf[-(PACKET_SIZE - 1):]
                break

            if magic_pos > 0:
                buf = buf[magic_pos:]   # discard pre-magic garbage

            if len(buf) < PACKET_SIZE:
                break   # wait for full packet

            fidx  = struct.unpack('>I', buf[4:8])[0]
            total = struct.unpack('>H', buf[8:10])[0]
            bidx  = struct.unpack('>H', buf[10:12])[0]

            # Sanity check header — skip 1 byte to resync in buf only
            if total == 0 or total > 100 or bidx >= total or fidx > 2_000_000:
                buf = buf[1:]
                continue

            rs_enc = bytes(buf[12:12 + 255])
            buf    = buf[PACKET_SIZE:]


            fb = frame_buffers[fidx]
            fb['total']        = total
            fb['blocks'][bidx] = rs_enc

            if len(fb['blocks']) >= total:
                avif_raw, corrections = rs_decode_blocks(fb['blocks'], total)
                del frame_buffers[fidx]

                if avif_raw is not None:
                    try:
                        pil_img = Image.open(io.BytesIO(avif_raw))
                        frame   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        if frame is not None:
                            right_count    += 1
                            rs_corrections += corrections
                            if frame_queue.full():
                                try: frame_queue.get_nowait()
                                except queue.Empty: pass
                            frame_queue.put((frame, corrections, total))
                            if corrections:
                                print(f"[RX] Frame {fidx}: OK +{corrections} RS fixes", flush=True)
                            else:
                                print(f"[RX] Frame {fidx}: OK", flush=True)
                        else:
                            wrong_count += 1
                    except Exception as e:
                        wrong_count += 1
                        print(f"[RX] Frame {fidx} AVIF err: {e}", flush=True)
                else:
                    wrong_count += 1
                    print(f"[RX] Frame {fidx}: RS FAILED", flush=True)

        # Purge stale incomplete frames
        if frame_buffers:
            for k in sorted(frame_buffers.keys())[:-10]:
                del frame_buffers[k]

        # Overflow guards
        if len(buf) > 200_000:
            print("[RX] buf overflow, trimming", flush=True)
            buf = buf[-5000:]
        if len(raw_chunk) > 50_000:
            print("[RX] raw overflow, trimming", flush=True)
            raw_chunk = raw_chunk[-5000:]

    ser.close()
    print("[RX] Thread stopped", flush=True)


# ----------------------------------------------------------------
# Main display loop
# ----------------------------------------------------------------
def main():
    global running, inference_ms, super_res_enabled, right_count, wrong_count, rs_corrections, last_rssi

    trt_ok = init_trt()

    t = threading.Thread(target=serial_reader_thread, daemon=True)
    t.start()

    win = "RX Reed-Solomon Stream"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, 640, 480)
    cv2.setMouseCallback(win, on_mouse)

    current_frame  = np.zeros((480, 640, 3), dtype=np.uint8)
    prev_time      = time.time()
    fps            = 0.0
    last_blocks    = 0
    last_corrected = 0

    print(f"[RX] Started. Firmware: payloadLen+RSSI framing (CC1310)", flush=True)
    print(f"[RX] RS params: {RS_DATA_BYTES} data + {RS_ECC_BYTES} parity per block", flush=True)
    print("[RX] Keys: Q=quit  S=SR", flush=True)

    try:
        while True:
            has_new = False
            try:
                raw_frame, corrections, total_blocks = frame_queue.get(timeout=0.005)
                has_new        = True
                last_blocks    = total_blocks
                last_corrected = corrections

                while not frame_queue.empty():
                    raw_frame, corrections, total_blocks = frame_queue.get_nowait()
                    last_blocks    = total_blocks
                    last_corrected = corrections

                if trt_ok and super_res_enabled:
                    t0 = time.time()
                    current_frame = infer_trt(raw_frame)
                    inference_ms  = (time.time() - t0) * 1000
                else:
                    current_frame = cv2.resize(raw_frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                    inference_ms  = 0

            except queue.Empty:
                pass

            if has_new:
                now       = time.time()
                fps       = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

            dw, dh = 640, 480
            if current_frame.shape[1] != dw or current_frame.shape[0] != dh:
                current_frame = cv2.resize(current_frame, (dw, dh))

            disp  = current_frame.copy()
            total = right_count + wrong_count
            srate = (right_count / total * 100) if total > 0 else 0.0

            if trt_ok and super_res_enabled:
                sr_lbl = "ON [TRT]"
            elif super_res_enabled:
                sr_lbl = "ON [N/A]"
            else:
                sr_lbl = "OFF"

            draw_info_box(disp, [
                f"FPS: {fps:.1f}",
                f"Inference: {inference_ms:.1f} ms",
                f"Resolution: {dw}x{dh}",
                f"RSSI: {last_rssi} dBm",
                f"Frames OK: {right_count}  Fail: {wrong_count}",
                f"Success: {srate:.1f}%",
                f"RS blocks/frame: {last_blocks}",
                f"RS corrections (last): {last_corrected}",
                f"RS corrections (total): {rs_corrections}",
            ])

            draw_button(disp, BTN_SR,
                        "SR: ON" if super_res_enabled else "SR: OFF",
                        super_res_enabled, (0, 180, 0), (0, 0, 180))

            if last_corrected > 0:
                cv2.putText(disp, f"RS corrected {last_corrected} errors!",
                            (10, dh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Handle mouse clicks
            global _mouse_click
            if _mouse_click is not None:
                cx, cy = _mouse_click
                _mouse_click = None
                
                rect = cv2.getWindowImageRect(win)
                if rect and rect[2] > 0 and rect[3] > 0:
                    win_w, win_h = rect[2], rect[3]
                    scale = min(win_w / 640.0, win_h / 480.0)
                    render_w = 640.0 * scale
                    render_h = 480.0 * scale
                    dx = (win_w - render_w) / 2.0
                    dy = (win_h - render_h) / 2.0
                    
                    rx = (cx - dx) / scale
                    ry = (cy - dy) / scale

                    # SR Toggle Check
                    bx, by, bw, bh = BTN_SR
                    if bx <= rx <= bx+bw and by <= ry <= by+bh:
                        super_res_enabled = not super_res_enabled
                        print(f"[RX] SR toggle: {super_res_enabled}")

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                super_res_enabled = not super_res_enabled

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        t.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("\n=== Statistics ===")
        print(f"Frames OK      : {right_count}")
        print(f"Frames Failed  : {wrong_count}")
        print(f"RS corrections : {rs_corrections} byte-errors corrected total")
        total = right_count + wrong_count
        if total > 0:
            print(f"Success Rate   : {right_count/total*100:.2f}%")
        print("[RX] Done.")


if __name__ == "__main__":
    main()
