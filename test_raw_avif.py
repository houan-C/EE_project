import cv2
import numpy as np
from PIL import Image
import pillow_avif
import io
import os
from skimage.metrics import structural_similarity as ssim

VIDEO_PATH = 'g:/code/EE_project/filghtRecord/2.mp4'
QUALITY = 30

def get_ssim(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return ssim(g1, g2, data_range=255)

print(f"Reading video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if orig_w > 640:
    scale = 640.0 / orig_w
    new_w = 640
    new_h = int(orig_h * scale)
else:
    new_w = orig_w
    new_h = orig_h

frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (new_w, new_h))
    frames.append(frame)
cap.release()

print(f"Loaded {len(frames)} frames. Testing Raw Frame-by-Frame AVIF at Quality {QUALITY}...")

total_size = 0
ssims = []

for frame in frames:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="AVIF", quality=QUALITY, speed=10)
    frame_bytes = buffer.getvalue()
    
    total_size += len(frame_bytes)
    
    decoded_pil = Image.open(io.BytesIO(frame_bytes))
    decoded_cv = cv2.cvtColor(np.array(decoded_pil), cv2.COLOR_RGB2BGR)
    
    ssims.append(get_ssim(frame, decoded_cv))

avg_ssim = np.mean(ssims)
print(f"Raw AVIF Quality {QUALITY}: Total Size = {total_size} bytes, SSIM = {avg_ssim:.4f}")
