import cv2
import serial
import numpy as np
import io
from PIL import Image
import pillow_avif

# ====================================================================
# === 🐾 核心參數對齊 ===
# ====================================================================
COM_PORT   = "COM11" # 🐾 請改成你 RX 板子的 COM！
BAUD_RATE  = 921600
# 接收格式：[Len(1)] + [Data(200)] + [RSSI(1)] + [Dummy(1)] = 203
CHUNK_SIZE = 203 

def run_rx():
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.01)
        print(f"[喵～] 成功開啟 {COM_PORT}，守候影像碎片中...")
    except Exception as e:
        print(f"[唔嗚～] 開啟序列埠失敗: {e}"); return

    buffer = bytearray()
    header = b'ftypavif' # AVIF 的指紋標頭喵！

    while True:
        data = ser.read(CHUNK_SIZE)
        if data:
            # 🐾 按照協議拆解：中間 200 Bytes 才是資料
            if len(data) == CHUNK_SIZE:
                payload = data[1:201]
                # rssi 為 int8_t，在 python 讀取為 uint8，因此如果是負數 (通常是) 需減 256
                rssi_raw = data[201]
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
    
    ##ser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_rx()