import cv2, serial, io, numpy as np
from PIL import Image
import pillow_avif

COM_PORT = "COM11" 
BAUD_RATE = 921600
# 接收格式：[Len(1)] + [Data(200)] + [RSSI(1)] + [Dummy(1)] = 203
CHUNK_SIZE = 203 

def run_rx():
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        ser.flushInput()
        print(f"[喵～] 接收獸已在 {COM_PORT} 埋伏...")
    except Exception as e:
        print(f"[唔嗚～] 開 Port 失敗: {e}"); return

    image_buffer = bytearray()
    header = b'ftypavif'
    
    #header = b'RIFF'
    while True:
        # 🐾 1. 先把所有數據吸進一個原始池
        if ser.in_waiting > 0:
            image_buffer.extend(ser.read(ser.in_waiting))
            print(f"[收集中] 緩衝長度: {len(image_buffer)} Bytes", end='\r')

        # 🐾 2. 暴力搜索 AVIF 指紋，不要管封包邊界了喵！
        idx = image_buffer.find(b'ftypavif')
        if idx != -1:
            # 找到標頭了！往後找下一個標頭或 WebP 標頭
            next_idx = image_buffer.find(b'ftypavif', idx + 8)
            
            if next_idx != -1:
                print(f"\n[喵嗚！] 捕捉到完整影像數據塊！大小: {next_idx - idx}")
                frame_data = image_buffer[idx : next_idx]
                image_buffer = image_buffer[next_idx:] # 移除已處理的部分喵！
                
                try:
                    img = Image.open(io.BytesIO(frame_data))
                    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    cv2.imshow("Drone Live View", bgr)
                except Exception as e:
                    print(f"[嗚～] 解碼失敗 (數據可能缺失): {e}")
            
            # 如果緩衝區太長卻找不到下一個標頭，代表後面資料斷了喵
            elif len(image_buffer) > 15000:
                print(f"\n[警告] 緩衝區過大，可能丟包了，清理舊數據喵！")
                image_buffer = image_buffer[idx+8:] 

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    ser.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_rx()