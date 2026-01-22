import cv2
import numpy as np
from pathlib import Path

# Usage: python pick_palette_points.py input.jpg
# Left click to sample. Press 'q' to quit.

import sys
if len(sys.argv) < 2:
    print("Usage: python pick_palette_points.py <image>")
    raise SystemExit(1)

img_path = Path(sys.argv[1])
bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
if bgr is None:
    raise FileNotFoundError(img_path)

disp = bgr.copy()
samples = []

def on_mouse(event, x, y, flags, param):
    global disp
    if event == cv2.EVENT_LBUTTONDOWN:
        # sample a small 5x5 patch median (more stable than 1 pixel)
        r = 2
        y0, y1 = max(0, y-r), min(bgr.shape[0], y+r+1)
        x0, x1 = max(0, x-r), min(bgr.shape[1], x+r+1)
        patch = bgr[y0:y1, x0:x1]
        med = np.median(patch.reshape(-1,3), axis=0).astype(int)
        B, G, R = med.tolist()
        samples.append((R, G, B))

        print(f"Clicked ({x},{y})  RGB=({R},{G},{B})  BGR=({B},{G},{R})")
        print(f'{{"palette_id": {len(samples)-1}, "name": "SET_ME", "bgr": ({B}, {G}, {R})}},')

        cv2.circle(disp, (x,y), 6, (0,0,255), 2)

cv2.namedWindow("pick")
cv2.setMouseCallback("pick", on_mouse)

while True:
    cv2.imshow("pick", disp)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()

# python pick_palette_points.py input/input.jpg
