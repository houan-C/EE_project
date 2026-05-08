import cv2
import serial
import time
import io
from PIL import Image
import pillow_avif # 🐾 確保有安裝這個喵！

import serial.tools.list_ports

# ====================================================================
# === 🐾 核心參數對齊 ===
# ====================================================================
BAUD_RATE  = 921600  # 高速通道
CHUNK_SIZE = 200     # 配合 C 程式碼的 UART_read(200)

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
            ser = serial.Serial(ports[choice].device, BAUD_RATE, timeout=1)
            print(f"[喵～] 成功開啟 {ports[choice].device}！")
            return ser
        else:
            print("[唔嗚～] 號碼超出範圍喵！")
    except Exception as e:
        print(f"[唔嗚～] 輸入無效或無法開啟序列埠: {e}")
        
    return None

def run_tx():
    ser = find_board('TX')
    if ser is None:
        print("[唔嗚～] 找不到 TX 板子，請確定已經燒錄且插上電腦喵！")
        return
    
    # 回復原本的 timeout 設定
    ser.timeout = 1

    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 🐾 1. 調整解析度 (320x240 對傳輸最友善喵)
        frame_resized = cv2.resize(frame, (320, 240))
        pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

        # 🐾 2. 壓縮成 AVIF
        buf = io.BytesIO()
        pil_img.save(buf, format="AVIF", quality=20, speed=10)
        img_bytes = buf.getvalue()

        # 🐾 3. 分塊噴發
        for i in range(0, len(img_bytes), CHUNK_SIZE):
            chunk = img_bytes[i : i + CHUNK_SIZE]
            
            # 🐾 如果不滿 200，補零讓板子不卡死喵！
            if len(chunk) < CHUNK_SIZE:
                chunk += b'\x00' * (CHUNK_SIZE - len(chunk))
            
            ser.write(chunk)
            time.sleep(0.005) # 給無線電一點喘息時間喵～

        print(f"發射完成！大小: {len(img_bytes)} Bytes")
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); ser.close()

if __name__ == "__main__":
    run_tx()