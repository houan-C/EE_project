import cv2
import serial
import serial.tools.list_ports
import time
import numpy as np
import sys
import os
import struct
import threading
import queue
import av
from fractions import Fraction
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ============================================================
# === COM Port Auto-Detection ===
# ============================================================
def find_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports: return "COM3"
    for p in ports:
        if "XDS110" in p.description: return p.device
    for p in ports:
        if "USB" in p.description or "UART" in p.description: return p.device
    return ports[0].device

com_port = find_serial_port()
print(f"Auto-selected TX COM Port: {com_port}")

baud_rate = 921600
timeout = 1
chunk_size = 812
VIDEO_PATH = "input.mp4" # <--- 請修改為你的影片檔案路徑)
print(f"Found COM port: {com_port}")

baud_rate = 921600
timeout = 1
chunk_size = 812

# ============================================================
# === AES-GCM ??閮剖? (敹???RX 蝡臬??其??? ===
# ============================================================
AES_KEY      = b'this_is_a_32_byte_secret_key_!!!'
aes_gcm      = AESGCM(AES_KEY)
MAGIC_HEADER = b'FRIEREN'
# Header: Magic(7s) + Level(B) + PayloadLen(I)
HEADER_FORMAT = '<7sBI'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

try:
    ser = serial.Serial(com_port, baud_rate, timeout=timeout)
except Exception as e:
    print(f"Error opening serial port {com_port}: {e}")
    sys.exit(1)

# ============================================================
# === H.265 GPU Encoder Setup ===
# ============================================================
def create_encoder(orig_w, orig_h, fps, level):
    settings = {
        4: {'scale': 1.0,  'crf': '24'},
        3: {'scale': 0.75, 'crf': '30'},
        2: {'scale': 0.5,  'crf': '36'},
        1: {'scale': 0.25, 'crf': '42'},
        0: {'scale': 0.125,'crf': '50'}
    }
    s = settings[level]
    
    enc_w = max(16, int(orig_w * s['scale']))
    enc_h = max(16, int(orig_h * s['scale']))
    enc_w -= (enc_w % 2)
    enc_h -= (enc_h % 2)
    
    try:
        # NVENC 蝖祇??楊蝣澆
        encoder = av.CodecContext.create('hevc_nvenc', 'w')
        encoder.options = {
            'preset': 'p4',       
            'tune': 'ull',        
            'rc': 'vbr',          
            'cq': s['crf']        
        }
    except Exception:
        # ??蝙??CPU 蝺函Ⅳ??        encoder = av.CodecContext.create('hevc', 'w')
        encoder.options = {
            'preset': 'ultrafast',
            'tune': 'zerolatency',
            'crf': s['crf']
        }
        
    encoder.width = enc_w
    encoder.height = enc_h
    encoder.pix_fmt = 'yuv420p'
    encoder.time_base = Fraction(1, int(fps))
    encoder.gop_size = 5 # Extremely short I-frame interval (resilient to packet loss)
    return encoder, enc_w, enc_h

# ============================================================
# === Async Threads ===
# ============================================================
latest_frame_lock = threading.Lock()
latest_frame = None
running = True

def camera_capture_thread(cap_obj):
    global latest_frame
    # We will control playback speed based on video FPS
    fps = cap_obj.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    frame_delay = 1.0 / fps

    while running:
        loop_start = time.time()
        ret, frm = cap_obj.read()
        if ret:
            with latest_frame_lock:
                latest_frame = frm
        else:
            # Loop the video when it ends
            cap_obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Ensure we don't read the file faster than its natural FPS
        elapsed = time.time() - loop_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

class PayloadState:
    def __init__(self, full_packet):
        self.full_packet = full_packet

latest_payload_lock = threading.Lock()
latest_payload_state = None

def serial_writer_thread(serial_port, c_size):
    global latest_payload_state
    while running:
        state = None
        with latest_payload_lock:
            if latest_payload_state is not None:
                state = latest_payload_state
                latest_payload_state = None

        if state is not None:
            packet = state.full_packet
            for i in range(0, len(packet), c_size):
                chunk = packet[i:i + c_size]
                try:
                    serial_port.write(chunk)
                except:
                    pass
                time.sleep(0.015) # Increased transmission frequency
        else:
            time.sleep(0.005)

# ============================================================
# === Main ===
# ============================================================
def main():
    global running, latest_payload_state
    
    print(f"Opening video file: {VIDEO_PATH} ...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{VIDEO_PATH}'. Please ensure the file exists.")
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w > 640:
        scale = 640.0 / orig_w
        orig_w = 640
        orig_h = int(orig_h * scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    print(f"Video File Initialized: {orig_w}x{orig_h} @ {fps} FPS")
    
    cap_thread = threading.Thread(target=camera_capture_thread, args=(cap,), daemon=True)
    cap_thread.start()
    
    writer_thread = threading.Thread(target=serial_writer_thread, args=(ser, chunk_size), daemon=True)
    writer_thread.start()

    current_level = 2
    encoder, enc_w, enc_h = create_encoder(orig_w, orig_h, fps, current_level)
    decoder = av.CodecContext.create('hevc', 'r')

    print("\n--- Controls ---")
    print("Press 0-4 to change quality level")
    print("Press 'q' to quit")
    print("----------------\n")
    
    last_sent_size = 0
    last_decoded_bgr = None
    window_name = 'TX_H265_Stream'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    while running:
        start_time = time.time()
        
        with latest_frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        frame = cv2.resize(frame, (orig_w, orig_h))
        
        if enc_w != orig_w or enc_h != orig_h:
            resized_frame = cv2.resize(frame, (enc_w, enc_h), interpolation=cv2.INTER_AREA)
        else:
            resized_frame = frame

        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

        # === Flow Control: Prevent dropping encoded frames ===
        # H.265 frames depend on each other (P-frames). If we encode a frame but the serial 
        # port is busy and we drop it, the RX side will fail to decode all subsequent frames 
        # until the next I-frame!
        tx_busy = False
        with latest_payload_lock:
            if latest_payload_state is not None:
                tx_busy = True
                
        if not tx_busy:
            # H.265 Encode
            packets = encoder.encode(av_frame)
            frame_bytes = bytearray()
            for p in packets:
                frame_bytes.extend(p)
                
                # Decode for local preview
                try:
                    dec_frames = decoder.decode(p)
                    for dec_frame in dec_frames:
                        dec_img = dec_frame.to_ndarray(format='rgb24')
                        last_decoded_bgr = cv2.cvtColor(dec_img, cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
                
            # AES-GCM Encrypt & Package
            if len(frame_bytes) > 0:
                nonce = os.urandom(12)
                ciphertext = aes_gcm.encrypt(nonce, bytes(frame_bytes), None)
                payload = nonce + ciphertext
                header = struct.pack(HEADER_FORMAT, MAGIC_HEADER, current_level, len(payload))
                full_packet = header + payload
                last_sent_size = len(full_packet)
                
                with latest_payload_lock:
                    latest_payload_state = PayloadState(full_packet)

        end_time = time.time()
        
        # UI Display
        if last_decoded_bgr is not None:
            if last_decoded_bgr.shape[1] != orig_w or last_decoded_bgr.shape[0] != orig_h:
                preview = cv2.resize(last_decoded_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            else:
                preview = last_decoded_bgr
        else:
            preview = np.zeros_like(frame)

        disp = np.hstack((frame, preview))
        
        # Draw on left side (Raw)
        cv2.putText(disp, f"TX H.265 (Raw Input)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        status_color = (0, 0, 255) if tx_busy else (200, 200, 200)
        cv2.putText(disp, f"Status: {'BUSY (Dropping Input)' if tx_busy else 'ENCODING'}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
        cv2.putText(disp, f"Loop Time: {end_time - start_time:.3f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Draw on right side (Compressed Preview)
        cv2.putText(disp, f"Compressed Preview (Level {current_level})", (orig_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, f"Enc Res: {enc_w}x{enc_h}", (orig_w + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, f"Pkt Size: {last_sent_size} Bytes", (orig_w + 10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow(window_name, disp)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False
            break
        elif key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4')]:
            new_level = int(chr(key))
            if new_level != current_level:
                current_level = new_level
                print(f"Switching to Level {current_level}...", flush=True)
                encoder, enc_w, enc_h = create_encoder(orig_w, orig_h, fps, current_level)
                decoder = av.CodecContext.create('hevc', 'r')
                last_sent_size = 0
                last_decoded_bgr = None

    cap_thread.join(timeout=1.0)
    writer_thread.join(timeout=1.0)
    cap.release()
    ser.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
