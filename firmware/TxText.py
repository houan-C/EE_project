import cv2, serial, time, io
from PIL import Image
import pillow_avif

# === 🐾 核心參數 ===
COM_PORT = "COM6" 
BAUD_RATE = 921600
CHUNK_SIZE = 200

def run_tx():
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"[喵～] 成功開啟 {COM_PORT}，噴發獸準備就緒！")
    except Exception as e:
        print(f"[唔嗚～] 開 Port 失敗: {e}"); return

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 縮小圖片讓傳輸更輕快
        frame_resized = cv2.resize(frame, (320, 240))
        pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        
        buf = io.BytesIO()
        pil_img.save(buf, format="AVIF", quality=20, speed=10) #
        #pil_img.save(buf, format="WEBP", quality=30, method=6)
        #header = b'RIFF'
        img_bytes = buf.getvalue()
        total_len = len(img_bytes)

        # 🐾 分塊噴發
        chunk_count = 0
        for i in range(0, total_len, CHUNK_SIZE):
            chunk = img_bytes[i : i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                chunk += b'\x00' * (CHUNK_SIZE - len(chunk))
            
            ser.write(chunk)
            chunk_count += 1
            # 顯示進度喵！
            print(f"[發送中] 正在噴發第 {chunk_count} 塊碎片...", end='\r')
            time.sleep(0.5) # 給硬體喘息空間

        print(f"\n[喵嗚！] 發射完成！總大小: {total_len} Bytes，共 {chunk_count} 塊碎片。")

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); ser.close()

if __name__ == "__main__":
    run_tx()