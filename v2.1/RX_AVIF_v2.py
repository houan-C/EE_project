import cv2
import serial
import serial.tools.list_ports
import numpy as np
import time
import os
import warnings
import sys
import threading
import queue
import io

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

def find_serial_port():
    """Automatically find the available COM port."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    
    # Priority 1: Look for XDS110 (common for CC1310 LaunchPad)
    for p in ports:
        if "XDS110" in p.description:
            return p.device
            
    # Priority 2: Look for any USB Serial port
    for p in ports:
        if "USB" in p.description or "UART" in p.description:
            return p.device
            
    # Fallback: Return the first available port
    return ports[0].device

# --- Settings ---
COM_PORT = find_serial_port()
if COM_PORT is None:
    print("🔴 Error: No available COM port found. Please check your connection.")
    sys.exit(1)
else:
    print(f"🟢 Found COM port: {COM_PORT}")

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
ENGINE_FILENAME = "RealESRGAN_x4plus_fp16_NEW.engine" 
ENGINE_PATH = os.path.join(BASE_DIR, ENGINE_FILENAME)

# Globals for TRT Optimization
TRT_STREAM = None
TRT_INPUT_TENSOR = None
TRT_OUTPUT_TENSOR = None
TRT_CONTEXT = None

# Threading & Stats Globals
frame_queue = queue.Queue(maxsize=1) # Holds only the LATEST frame
running = True
right_count = 0
wrong_count = 0

# UI State
super_res_enabled = True
button_area = (500, 10, 130, 40) # (x, y, w, h)


# ====================================================================
# TensorRT Optimized Functions
# ====================================================================
def init_trt():
    global TRT_CONTEXT, TRT_STREAM, TRT_INPUT_TENSOR, TRT_OUTPUT_TENSOR
    
    if DEVICE.type != 'cuda':
        return False
        
    if not os.path.exists(ENGINE_PATH):
        print(f"🔴 Engine not found: {ENGINE_PATH}")
        return False

    try:
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
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
    global running, right_count, wrong_count
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
                        frame_queue.put((frame, rssi))
                    else:
                        wrong_count += 1
                        print("❌ Decode failed (Frame is None)")
                except Exception as e:
                    wrong_count += 1
                    print(f"❌ Decode Error: {e}")

        except Exception as e:
            print(f"Serial Loop Error: {e}")
            time.sleep(0.1)
    
    ser.close()
    print("🔵 Serial Thread Stopped")

# ====================================================================
# Main Display Loop
# ====================================================================
def main():
    global running
    
    # 1. Init TRT
    trt_ok = init_trt()
    
    # 2. Start Serial Thread
    t = threading.Thread(target=serial_reader_thread)
    t.daemon = True
    t.start()
    
    # 3. GUI Setup
    win_name = "Fluid TRT Stream"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    
    prev_time = time.time()
    fps = 0
    rssi = 0
    
    # Initial Blank Frame (will be resized on first display update)
    current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("🟢 Main Loop Started. Waiting for frames...")

    try:
        while True:
            # Non-blocking check for new frame
            has_new_frame = False
            try:
                # Get latest frame
                raw_frame, rssi = frame_queue.get(timeout=0.005) # 5ms Poll
                has_new_frame = True
                
                # Drain queue
                while not frame_queue.empty():
                     raw_frame, rssi = frame_queue.get_nowait()
                     
            except queue.Empty:
                pass
            
            if has_new_frame:
                # --- Processing ---
                t0 = time.time()
                
                proc_frame = raw_frame
                
                # 1. TensorRT Enhance (If ready and ENABLED)
                if trt_ok and super_res_enabled:
                    try:
                        proc_frame = infer_trt(raw_frame)
                    except Exception as e:
                        print(f"Infer Error: {e}")
                        proc_frame = raw_frame 
                
                # 2. Resize
                # If SR was done, proc_frame is 4x larger (1280x960), we scale by 0.5 -> 640x480
                # If SR was NOT done, proc_frame is 320x240, we need to UPscale to match 640x480
                
                target_display_w = 640
                target_display_h = 480
                
                if proc_frame.shape[1] != target_display_w or proc_frame.shape[0] != target_display_h:
                     proc_frame = cv2.resize(proc_frame, (target_display_w, target_display_h), interpolation=cv2.INTER_LINEAR)
                
                
                dt = time.time() - t0
                
                # Update Display Frame
                current_frame = proc_frame
                
                # FPS Calc
                fps = 1.0 / (time.time() - prev_time)
                prev_time = time.time()
                
                # draw OSD
                cv2.putText(current_frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(current_frame, f"RSSI: {rssi}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw Button
                bx, by, bw, bh = button_area
                color = (0, 255, 0) if super_res_enabled else (0, 0, 255)
                text = "SR: ON" if super_res_enabled else "SR: OFF"
                
                cv2.rectangle(current_frame, (bx, by), (bx + bw, by + bh), color, -1)
                cv2.rectangle(current_frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
                cv2.putText(current_frame, text, (bx + 10, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Refresh Window (Always run this to keep UI responsive)
            cv2.imshow(win_name, current_frame)
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
