import cv2
import numpy as np
from pathlib import Path
import json
import sys
import time

# ======================================================
# Usage:
#   python pick_palette_points_visual_export.py input.jpg
#
# Controls:
#   - Left click: sample color using current label
#   - R/O/G/B/N/E/K: set current label (red/orange/green/blue/brown/grey/black)
#   - U: undo last sample
#   - S: save palette JSON
#   - Q: quit
#
# Output:
#   output_palette_<inputstem>.json (same folder as input)
# ======================================================

LABEL_KEYS = {
    ord('r'): "RED",
    ord('o'): "ORANGE",
    ord('y'): "YELLOW",
    ord('g'): "GREEN",
    ord('b'): "BLUE",
    ord('n'): "BROWN",
    ord('t'): "SKIN",
    ord('e'): "GREY",
    ord('k'): "BLACK",
}

if len(sys.argv) < 2:
    print("Usage: python pick_palette_points_visual_export.py <image>")
    raise SystemExit(1)

img_path = Path(sys.argv[1])
bgr_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
if bgr_img is None:
    raise FileNotFoundError(img_path)

H, W = bgr_img.shape[:2]
disp = bgr_img.copy()

# UI layout
SWATCH_W = 230
SWATCH_H = 160
PADDING = 10

current_label = "BLUE"
samples = []  # list of dicts: {palette_id, name, bgr:[b,g,r], rgb:[r,g,b], pos:[x,y]}

def out_palette_path():
    return img_path.parent / f"output_palette_{img_path.stem}.json"

def median_patch_bgr(x, y, r=2):
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    patch = bgr_img[y0:y1, x0:x1]
    med = np.median(patch.reshape(-1, 3), axis=0).astype(int)
    return med.tolist()  # [B,G,R]

def draw_panel():
    global disp
    disp = bgr_img.copy()

    # draw sampled points
    for s in samples:
        x, y = s["pos"]
        B, G, R = s["bgr"]
        cv2.circle(disp, (x, y), 7, (B, G, R), -1)
        cv2.circle(disp, (x, y), 7, (0, 0, 0), 1)

    # right-side info panel background
    x0 = W - SWATCH_W - PADDING
    y0 = PADDING
    cv2.rectangle(disp, (x0 - 6, y0 - 6), (x0 + SWATCH_W + 6, y0 + SWATCH_H + 190), (245, 245, 245), -1)

    # current label
    cv2.putText(disp, f"Current: {current_label}", (x0, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # shortcut hint (on-screen)
    help_lines = [
        "R:RED  O:ORANGE  Y:YELLOW",
        "G:GREEN  B:BLUE  N:BROWN",
        "T:SKIN  E:GREY  K:BLACK",
        "U=undo  S=save  Q=quit",
        "Left click: sample current label",
    ]
    yy = y0 + 45
    for line in help_lines:
        cv2.putText(disp, line, (x0, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        yy += 18

    # sample count
    cv2.putText(disp, f"Samples: {len(samples)}", (x0, y0 + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def draw_last_swatch():
    """Draw swatch for last sample if exists."""
    if not samples:
        return
    s = samples[-1]
    B, G, R = s["bgr"]
    x0 = W - SWATCH_W - PADDING
    y0 = PADDING + 100

    # color swatch
    cv2.rectangle(disp, (x0, y0), (x0 + SWATCH_W, y0 + SWATCH_H), (B, G, R), -1)
    cv2.rectangle(disp, (x0, y0), (x0 + SWATCH_W, y0 + SWATCH_H), (0, 0, 0), 1)

    # text lines
    lines = [
        f"Last: #{s['palette_id']}  {s['name']}",
        f"RGB {tuple(s['rgb'])}",
        f"BGR {tuple(s['bgr'])}",
        f"pos {tuple(s['pos'])}",
    ]
    ty = y0 + SWATCH_H + 20
    for line in lines:
        cv2.putText(disp, line, (x0, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        ty += 18

def save_palette_json():
    path = out_palette_path()
    payload = {
        "source_image": img_path.name,
        "created_unix": int(time.time()),
        "palette_entries": [
            {"palette_id": s["palette_id"], "name": s["name"], "bgr": s["bgr"]}
            for s in samples
        ],
        # convenience summary for humans
        "summary_counts": {k: sum(1 for s in samples if s["name"] == k) for k in sorted(set(s["name"] for s in samples))}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] {path}")

def on_mouse(event, x, y, flags, param):
    global samples
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    bgr = median_patch_bgr(x, y, r=2)
    B, G, R = bgr
    rgb = [R, G, B]

    palette_id = len(samples)  # sequential unique id in this exported file
    samples.append({
        "palette_id": palette_id,
        "name": current_label,
        "bgr": [int(B), int(G), int(R)],
        "rgb": [int(R), int(G), int(B)],
        "pos": [int(x), int(y)],
    })

    print(f"Clicked ({x},{y}) label={current_label}  RGB={tuple(rgb)}  BGR={tuple(bgr)}")
    print(f'{{"palette_id": {palette_id}, "name": "{current_label}", "bgr": ({B}, {G}, {R})}},\n')

    draw_panel()
    draw_last_swatch()

# init UI
draw_panel()
cv2.namedWindow("palette picker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("palette picker", on_mouse)

while True:
    cv2.imshow("palette picker", disp)
    key = cv2.waitKey(16) & 0xFF
    if key == ord('q'):
        break
    if key in LABEL_KEYS:
        current_label = LABEL_KEYS[key]
        draw_panel()
        draw_last_swatch()
    if key == ord('u'):
        if samples:
            removed = samples.pop()
            print(f"[Undo] removed #{removed['palette_id']} {removed['name']} {removed['bgr']}")
            draw_panel()
            draw_last_swatch()
    if key == ord('s'):
        save_palette_json()

cv2.destroyAllWindows()


# python pick_palette_points.py input/Drawing0.jpg
# Shortcuts (as you requested)

# R = red

# O = orange

# G = green

# B = blue

# N = brown (“N” for brown)

# E = grey (“E” for grey)

# K = black (“K” for black)

# Also:

# U = undo last sample

# S = save JSON to disk

# Q = quit

#Orange, brown, skin tone: to farm land and maybe some trees or other objects yet to determine.
# Gray and black for roads and stuff
# Blue for cityscapes
# Red for flowers
# Green for grass and trees
# yellow for wooden houses and autumn leaves 