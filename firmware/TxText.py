import cv2
import serial
import time
import io
from PIL import Image
import pillow_avif # 🐾 確保有安裝這個喵！

# ====================================================================
# === 🐾 核心參數對齊 ===
# ====================================================================
COM_PORT   = "COM6" # 🐾 請改成你 TX 板子的 COM！
BAUD_RATE  = 921600  # 高速通道
CHUNK_SIZE = 200     # 配合 C 程式碼的 UART_read(200)

def run_tx():
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"[喵～] 成功開啟 {COM_PORT}，開始噴發影像數據！")
    except Exception as e:
        print(f"[唔嗚～] 開啟序列埠失敗: {e}"); return

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