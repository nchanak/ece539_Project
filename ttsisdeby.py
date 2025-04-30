import cv2, numpy as np
from pathlib import Path

in1 = "debug_videos/original.avi"
in2 = "debug_videos/recon_epoch10.avi"
out  = "debug_videos/side_by_side.avi"

cap1 = cv2.VideoCapture(in1)
cap2 = cv2.VideoCapture(in2)

# assume both videos are same FPS & frame‐size
fps    = cap1.get(cv2.CAP_PROP_FPS)
w      = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
vw     = cv2.VideoWriter(out, fourcc, fps, (w*2, h))

while True:
    r1, f1 = cap1.read()
    r2, f2 = cap2.read()
    if not (r1 and r2):
        break
    # stack horizontally
    combo = np.concatenate([f1, f2], axis=1)
    vw.write(combo)

cap1.release()
cap2.release()
vw.release()

print("Side‑by‑side video written to", Path(out).resolve())
