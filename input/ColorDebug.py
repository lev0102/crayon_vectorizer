import cv2
import numpy as np

img = cv2.imread("D1.png", cv2.IMREAD_COLOR)
if img is None:
    raise RuntimeError("Could not read image")

colors = img.reshape(-1, 3)
unique, counts = np.unique(colors, axis=0, return_counts=True)

# Show most common colors
order = np.argsort(-counts)
for i in order[:20]:
    bgr = unique[i]
    print(f"BGR={bgr.tolist()} count={int(counts[i])}")