"""
tx_reed.py - AVIF transmitter from video file with Reed-Solomon FEC
====================================================================
Reads frames sequentially from "drone/drone vid 1.mp4" on loop.
Protocol (per block sent over serial):
  [ MAGIC(4B) | frame_idx(4B, big-endian) | total_blocks(2B) | block_idx(2B) | rs_encoded_data(255B) ]
  = 267 bytes per packet
"""

import cv2
import serial
import time
import numpy as np
import io
import sys
import os
import struct
import random
from PIL import Image

try:
    import pillow_avif
except ImportError:
    print("WARNING: pillow_avif not found. Install: pip install pillow-avif-plugin")

try:
    import reedsolo
    RSCodec = reedsolo.RSCodec
except ImportError:
    print("ERROR: reedsolo not found. Install: pip install reedsolo")
    sys.exit(1)

# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------
COM_PORT   = "COM5"
BAUD_RATE  = 921600

RS_ECC_BYTES  = 32
RS_DATA_BYTES = 255 - RS_ECC_BYTES   # 223
RS_MAGIC      = b'RSAV'

rsc = RSCodec(RS_ECC_BYTES)

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "drone", "drone vid 1.mp4")

# Quality presets: level -> (width, height, avif_quality)
QUALITY_PRESETS = {
    4: (480, 360, 29),
    3: (420, 315, 25),
    2: (360, 270, 22),
    1: (300, 225, 20),
    0: (240, 180, 18),
}

current_level     = 3
test_corrupt_mode = False
rs_enabled        = True          # RS always True for tx_reed
frame_idx         = 0
CORRUPT_EVERY_N   = 30
corrupt_countdown = CORRUPT_EVERY_N

# UI Dimensions (on 640x480 display canvas)
BTN_CORRUPT = (10, 10, 180, 35) # x, y, w, h
BTN_LEVEL_U = (10, 55, 85, 30)
BTN_LEVEL_D = (105, 55, 85, 30)


# ----------------------------------------------------------------
# Encoding
# ----------------------------------------------------------------
def encode_frame(avif_bytes, fidx, corrupt=False):
    data = bytearray(avif_bytes)
    remainder = len(data) % RS_DATA_BYTES
    if remainder:
        data.extend(b'\x00' * (RS_DATA_BYTES - remainder))

    blocks = [data[i:i + RS_DATA_BYTES] for i in range(0, len(data), RS_DATA_BYTES)]
    total  = len(blocks)

    corrupt_block_idx = random.randint(0, total - 1) if corrupt else -1

    packets = []
    for bidx, block in enumerate(blocks):
        # Always RS encode
        encoded = bytes(rsc.encode(bytes(block)))

        if bidx == corrupt_block_idx:
            ea = bytearray(encoded)
            for _ in range(3):
                pos = random.randint(0, len(ea) - 1)
                ea[pos] ^= random.randint(1, 255)
            encoded = bytes(ea)

        header = RS_MAGIC + struct.pack('>IHH', fidx, total, bidx)
        packets.append(header + encoded)   # 267 bytes

    return packets, total, corrupt_block_idx >= 0


def send_frame(ser, packets):
    payload = b''.join(packets)
    chunk_size = 812
    tx_delay = 0.03
    for i in range(0, len(payload), chunk_size):
        ser.write(payload[i:i + chunk_size])
        time.sleep(tx_delay)


def draw_ui(img, level, corrupt_on, fps, num_blocks, did_corrupt):
    # Add Corrupt Frames button
    bx, by, bw, bh = BTN_CORRUPT
    color = (0, 80, 220) if corrupt_on else (60, 60, 60)
    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), color, -1)
    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255, 255, 255), 1)
    label = "Corrupt: ON [T]" if corrupt_on else "Add Corrupt [T]"
    cv2.putText(img, label, (bx+10, by+23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Q+/Q- buttons
    for (lx, ly, lw, lh), txt in [(BTN_LEVEL_U, "Q+ [=]"), (BTN_LEVEL_D, "Q- [-]")]:
        cv2.rectangle(img, (lx, ly), (lx+lw, ly+lh), (80, 80, 80), -1)
        cv2.rectangle(img, (lx, ly), (lx+lw, ly+lh), (200, 200, 200), 1)
        cv2.putText(img, txt, (lx+10, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Corruption flash
    if did_corrupt:
        cv2.putText(img, "** CORRUPT SENT — PROTECTION ON **", (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

    # Stats overlay
    res_w, res_h, q = QUALITY_PRESETS[level]
    stats = [
        f"TX FPS: {fps:.1f}",
        f"Level: {level}  Res: {res_w}x{res_h}  Q: {q}",
        f"Blocks/frame: {num_blocks} (RS Active)",
    ]
    oy = 480 - 15 - 20 * len(stats)
    overlay = img.copy()
    cv2.rectangle(overlay, (5, oy - 5), (300, 480 - 5), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(stats):
        cv2.putText(img, line, (12, oy + i * 20 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 200), 1)


def check_button(x, y, btn):
    bx, by, bw, bh = btn
    return bx <= x <= bx + bw and by <= y <= by + bh


_mouse_click = None

def on_mouse(event, x, y, flags, param):
    global _mouse_click
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_click = (x, y)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main():
    global frame_idx, current_level, test_corrupt_mode, corrupt_countdown, _mouse_click

    if not os.path.exists(VIDEO_PATH):
        print(f"[TX] ERROR: Video not found: {VIDEO_PATH}", flush=True)
        sys.exit(1)

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"[TX] Serial opened: {COM_PORT} @ {BAUD_RATE}", flush=True)
    except Exception as e:
        print(f"[TX] ERROR Cannot open serial port {COM_PORT}: {e}", flush=True)
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[TX] ERROR Cannot open video: {VIDEO_PATH}", flush=True)
        ser.close()
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0
    print(f"[TX] Video: {VIDEO_PATH} ({total_frames} frames, {video_fps} FPS, looping)", flush=True)

    win = "TX Reed-Solomon AVIF"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(win, 640, 480)
    cv2.setMouseCallback(win, on_mouse)

    start_time             = time.time()
    internal_frame_counter = 0

    prev_time        = time.time()
    fps              = 0.0
    num_blocks       = 0
    did_corrupt_last = False

    print("[TX] Started. RS: ON (Always)", flush=True)

    while True:
        # Expected frame based on time
        real_expected_frame = int((time.time() - start_time) * video_fps)
        
        # Skip frames rapidly using cap.grab() if transmission is behind
        while internal_frame_counter < real_expected_frame:
            cap.grab()
            internal_frame_counter += 1

        ret, frame = cap.read()
        internal_frame_counter += 1
        
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(VIDEO_PATH)
            start_time = time.time()
            internal_frame_counter = 0
            print("[TX] Video re-opened & time re-synced", flush=True)
            continue

        level = current_level
        res_w, res_h, q = QUALITY_PRESETS[level]
        frame_small = cv2.resize(frame, (res_w, res_h))

        buf = io.BytesIO()
        pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
        pil.save(buf, format="AVIF", quality=q, speed=10)
        avif_bytes = buf.getvalue()

        inject_corrupt = False
        if test_corrupt_mode:
            corrupt_countdown -= 1
            if corrupt_countdown <= 0:
                inject_corrupt = True
                corrupt_countdown = CORRUPT_EVERY_N

        packets, num_blocks, injected = encode_frame(avif_bytes, frame_idx, corrupt=inject_corrupt)
        send_frame(ser, packets)
        did_corrupt_last = injected

        frame_idx += 1
        t_now = time.time()
        fps   = 1.0 / max(t_now - prev_time, 1e-6)
        prev_time = t_now

        if injected:
            print(f"[TX] Frame {frame_idx}: CORRUPT ({num_blocks} blocks)  FPS={fps:.1f}", flush=True)
        else:
            print(f"[TX] Frame {frame_idx}: {len(avif_bytes)}B AVIF -> {num_blocks} blocks  FPS={fps:.1f}", flush=True)

        disp = cv2.resize(frame, (640, 480))

        # Mouse UI translation
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

                if check_button(rx, ry, BTN_CORRUPT):
                    test_corrupt_mode = not test_corrupt_mode
                    print(f"[TX] Add Corrupt Frames: {test_corrupt_mode}", flush=True)
                elif check_button(rx, ry, BTN_LEVEL_U):
                    if current_level < 4:
                        current_level += 1
                elif check_button(rx, ry, BTN_LEVEL_D):
                    if current_level > 0:
                        current_level -= 1

        draw_ui(disp, level, test_corrupt_mode, fps, num_blocks, did_corrupt_last)
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('t'), ord('T')):
            test_corrupt_mode = not test_corrupt_mode
            print(f"[TX] Add Corrupt Frames: {test_corrupt_mode}", flush=True)
        elif key in (ord('+'), ord('=')):
            if current_level < 4:
                current_level += 1
        elif key in (ord('-'), ord('_')):
            if current_level > 0:
                current_level -= 1

    cap.release()
    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()