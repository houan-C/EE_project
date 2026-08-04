import cv2
import serial
import serial.tools.list_ports
import numpy as np
import time
import threading
import queue
import io
import sys
import warnings

warnings.filterwarnings("ignore")

# --- 確保有安裝 Pillow 和 pillow-avif-plugin ---
try:
    from PIL import Image
    import pillow_avif
except ImportError:
    print("Error: 找不到 pillow-avif-plugin，請執行: pip install pillow pillow-avif-plugin")
    sys.exit(1)

def find_serial_port():
    """自動尋找可用的 COM Port"""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    
    # 優先尋找 TI 開發板
    for p in ports:
        if "XDS110" in p.description:
            return p.device
            
    # 其次尋找一般 USB 序列埠
    for p in ports:
        if "USB" in p.description or "UART" in p.description:
            return p.device
            
    # 找不到就回傳第一個
    return ports[0].device

# --- 設定區 ---
COM_PORT = find_serial_port()
if COM_PORT is None:
    print("Error: 找不到可用的 COM Port，請檢查硬體連線。")
    sys.exit(1)
else:
    print(f"🟢 成功找到 COM Port: {COM_PORT}")

BAUD_RATE = 921600
CHUNK_SIZE = 1024 
TIMEOUT = 0.01

# --- 全域變數 ---
frame_queue = queue.Queue(maxsize=1) # 只保留最新一幀畫面
running = True
right_count = 0
wrong_count = 0

# ====================================================================
# 序列埠讀取與解碼執行緒 (背景執行)
# ====================================================================
def serial_reader_thread():
    global running, right_count, wrong_count
    print(f"🔵 序列埠執行緒已啟動，開始接收資料...")
    
    buffer = bytearray()
    chunk = bytearray()
    
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=TIMEOUT)
    except Exception as e:
        print(f"❌ 序列埠開啟失敗: {e}")
        running = False
        return

    while running:
        try:
            read_data = ser.read(CHUNK_SIZE)
            if not read_data:
                continue
                
            chunk.extend(read_data)
            
            # 1. 剝除 DSSS 底層協定 (payloadLen, rssi)
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
            
            # 2. 尋找 AVIF 圖片特徵碼並解碼
            while True:
                header_sig = b'ftypavif'
                match_idx = buffer.find(header_sig)
                
                if match_idx == -1:
                    break # 資料還不夠，繼續等
                    
                # 真正的 AVIF 開頭是在 ftypavif 前面 4 個 bytes (Size Box)
                start_offset = match_idx - 4
                if start_offset > 0:
                    buffer = buffer[start_offset:]
                    match_idx = 4 

                # 尋找下一張圖片的開頭，來決定目前這張圖到哪裡結束
                next_match_idx = buffer.find(header_sig, match_idx + len(header_sig))
                
                if next_match_idx == -1:
                    break # 下一張圖片還沒來，代表目前這張還沒收完
                
                frame_end = next_match_idx - 4
                frame_data = buffer[:frame_end]
                buffer = buffer[frame_end:] # 把 buffer 推進到下一張圖片
                
                # 解碼 AVIF 圖片
                try:
                    pil_img = Image.open(io.BytesIO(frame_data))
                    frame_rgb = np.array(pil_img)
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) # 轉回 OpenCV 的 BGR 格式
                    
                    if frame is not None:
                        right_count += 1
                        # 放入佇列準備顯示，如果滿了就強制抽掉舊的，保證低延遲
                        if frame_queue.full():
                            try:
                                frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        frame_queue.put((frame, rssi))
                    else:
                        wrong_count += 1
                except Exception as e:
                    # 圖片損毀或解碼失敗
                    wrong_count += 1

        except Exception as e:
            print(f"迴圈發生錯誤: {e}")
            time.sleep(0.1)
    
    ser.close()
    print("🔵 序列埠執行緒已安全停止")

# ====================================================================
# 主程式 (UI 顯示迴圈)
# ====================================================================
def main():
    global running
    
    # 啟動背景接收執行緒
    t = threading.Thread(target=serial_reader_thread)
    t.daemon = True
    t.start()
    
    win_name = "AVIF Basic Stream (No AI)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    prev_time = time.time()
    fps = 0
    rssi = 0
    
    # 建立一張初始的黑色畫面
    current_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("🟢 顯示介面已啟動，等待畫面傳入... (按 'q' 鍵離開)")

    try:
        while True:
            has_new_frame = False
            try:
                # 嘗試拿取最新畫面，Timeout 短一點讓 UI 保持流暢
                raw_frame, rssi = frame_queue.get(timeout=0.005)
                has_new_frame = True
                
                # 清空佇列中多餘的舊畫面 (Drain Queue)
                while not frame_queue.empty():
                     raw_frame, rssi = frame_queue.get_nowait()
            except queue.Empty:
                pass
            
            if has_new_frame:
                # 統一縮放到 640x480 顯示
                target_w, target_h = 640, 480
                if raw_frame.shape[1] != target_w or raw_frame.shape[0] != target_h:
                     raw_frame = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                current_frame = raw_frame
                
                # 計算即時 FPS
                fps = 1.0 / (time.time() - prev_time)
                prev_time = time.time()
                
                # 畫上 FPS 與 RSSI 資訊
                cv2.putText(current_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(current_frame, f"RSSI: {rssi} dBm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 更新視窗
            cv2.imshow(win_name, current_frame)
            
            # 按 'q' 鍵離開
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
                
            # 確保背景執行緒還活著
            if not t.is_alive():
                print("❌ 接收執行緒意外終止。")
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 安全關閉所有資源
        running = False
        t.join(timeout=1.0)
        cv2.destroyAllWindows()
        
        print("\n=== 傳輸統計 ===")
        print(f"成功接收 (Frames) : {right_count}")
        print(f"損毀遺失 (Frames) : {wrong_count}")
        total = right_count + wrong_count
        if total > 0:
            print(f"封包成功率        : {right_count/total*100:.2f}%")
        print("系統已安全關閉。")

if __name__ == "__main__":
    main()
