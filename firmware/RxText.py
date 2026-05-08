import cv2
import serial
import numpy as np
import io
from PIL import Image
import pillow_avif

import serial.tools.list_ports

# ====================================================================
# === 🐾 核心參數對齊 ===
# ====================================================================
BAUD_RATE  = 921600
# 接收格式：[Len(1)] + [Data(200)] + [RSSI(1)] + [Dummy(1)] = 203
CHUNK_SIZE = 203 

def find_board(expected_role):
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("[唔嗚～] 沒有找到任何 COM Port！請檢查裝置是否連接。")
        return None
        
    print(f"\n[喵～] 找到以下裝置，請選擇你的 {expected_role} 板子：")
    for i, port in enumerate(ports):
        print(f"[{i}] {port.device} - {port.description}")
        
    try:
        choice = input(f"\n請輸入 {expected_role} 的號碼 (例如 0): ")
        choice = int(choice)
        if 0 <= choice < len(ports):
            ser = serial.Serial(ports[choice].device, BAUD_RATE, timeout=0.01)
            print(f"[喵～] 成功開啟 {ports[choice].device}！")
            return ser
        else:
            print("[唔嗚～] 號碼超出範圍喵！")
    except Exception as e:
        print(f"[唔嗚～] 輸入無效或無法開啟序列埠: {e}")
        
    return None

def run_rx():
    ser = find_board('RX')
    if ser is None:
        print("[唔嗚～] 找不到 RX 板子，請確定已經燒錄且插上電腦喵！")
        return
    
    # 回復原本的 timeout 設定
    ser.timeout = 0.01

    buffer = bytearray()  # 存放乾淨的影像資料
    raw_buffer = bytearray() # 存放序列埠進來的原始資料
    header = b'ftypavif' # AVIF 的指紋標頭喵！

    while True:
        # 讀取目前序列埠中所有可用的資料
        data = ser.read(ser.in_waiting or 1)
        if data:
            raw_buffer.extend(data)
            
            # 🐾 按照協議拆解：每 203 Bytes 是一包
            while len(raw_buffer) >= CHUNK_SIZE:
                chunk = raw_buffer[:CHUNK_SIZE]
                raw_buffer = raw_buffer[CHUNK_SIZE:] # 移除已經處理的包
                
                payload = chunk[1:201]
                # rssi 為 int8_t，在 python 讀取為 uint8，因此如果是負數 (通常是) 需減 256
                rssi_raw = chunk[201]
                rssi = rssi_raw - 256 if rssi_raw > 127 else rssi_raw
                buffer.extend(payload)

            # 🐾 尋找並拼接完整的 AVIF 檔案
            idx = buffer.find(header)
            if idx != -1:
                # 尋找下一個標頭當作結尾
                next_idx = buffer.find(header, idx + 8)
                if next_idx != -1:
                    frame_data = buffer[idx : next_idx]
                    buffer = buffer[next_idx:] # 留下之後的資料喵！
                    
                    try:
                        img = Image.open(io.BytesIO(frame_data))
                        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        
                        # 顯示 RSSI 與畫面喵！
                        cv2.putText(bgr, f"RSSI: {rssi}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("RX Stream 喵！", bgr)
                    except Exception as e:
                        print(f"解碼失敗: {e}") # 解碼失敗就先跳過喵～

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    ser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_rx()