import cv2
from PIL import Image
import pillow_avif
import io
import imageio
import os
import numpy as np

VIDEO_PATH = 'g:/code/EE_project/filghtRecord/2.mp4'

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

orig_w = int(frame.shape[1])
orig_h = int(frame.shape[0])
scale = 640.0 / orig_w
new_w, new_h = 640, int(orig_h * scale)
# Ensure height is divisible by 16 for H265 encoder
new_h = (new_h // 16) * 16
frame_resized = cv2.resize(frame, (new_w, new_h))

frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
img = Image.fromarray(frame_rgb)
img.save('original_frame.png')

# AVIF Q30
buf = io.BytesIO()
img.save(buf, format='AVIF', quality=30)
avif_bytes = buf.getvalue()
avif_dec = Image.open(io.BytesIO(avif_bytes))
avif_dec.save('sample_avif_decoded.png')
print(f"AVIF size: {len(avif_bytes)} bytes")

# H.265 CRF 37
imageio.mimwrite('sample_h265.mp4', [frame_rgb], format='FFMPEG', codec='libx265', output_params=['-crf', '37'])
h265_dec = imageio.mimread('sample_h265.mp4')[0]
Image.fromarray(h265_dec).save('sample_h265_decoded.png')
print(f"H.265 size: {os.path.getsize('sample_h265.mp4')} bytes")
