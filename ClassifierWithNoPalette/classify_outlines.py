import cv2
import numpy as np
from pathlib import Path
import json

def quantize_kmeans_lab(bgr, k=10, attempts=3, max_iter=30):
    """
    Quantize image into k colors using k-means in Lab space.
    Returns (labels_hxw, centers_bgr_kx3).
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    Z = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape((h, w))
    centers = centers.astype(np.uint8)

    # Convert Lab centers -> BGR for drawing
    centers_lab = centers.reshape((k, 1, 3))
    centers_bgr = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR).reshape((k, 3))
    return labels, centers_bgr

def map_centers_to_palette(centers_bgr, palette_bgr):
    """
    Map each cluster center to nearest palette color (Euclidean in Lab for perceptual match).
    Returns mapped_centers_bgr_kx3.
    """
    centers_lab = cv2.cvtColor(centers_bgr.reshape((-1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape((-1, 3)).astype(np.float32)
    palette_lab = cv2.cvtColor(palette_bgr.reshape((-1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape((-1, 3)).astype(np.float32)

    # distances: k x p
    d = ((centers_lab[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d, axis=1)
    return palette_bgr[idx]

def clean_mask(mask, close_radius=3, open_radius=1):
    """
    Clean a binary mask: close small gaps then remove small noise.
    """
    m = mask.astype(np.uint8) * 255

    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius*2+1, close_radius*2+1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if open_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius*2+1, open_radius*2+1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    return (m > 0).astype(np.uint8)

def draw_region_outlines(labels, centers_bgr, out_shape, line_thickness=2,
                         close_radius=3, open_radius=1,
                         min_area_px=200, simplify_epsilon_px=2.0,
                         draw_internal_holes=True, background="white"):
    """
    Build a 'line render' image: outlines of each color region.
    Outlines are colored by region's classified color.
    """
    h, w = out_shape[:2]
    if background == "white":
        out = np.full((h, w, 3), 255, dtype=np.uint8)
    elif background == "black":
        out = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        out = np.full((h, w, 3), 255, dtype=np.uint8)

    k = centers_bgr.shape[0]

    for ci in range(k):
        mask = (labels == ci).astype(np.uint8)
        mask = clean_mask(mask, close_radius=close_radius, open_radius=open_radius)

        # Skip tiny areas
        area = int(mask.sum())
        if area < min_area_px:
            continue

        # Find contours
        mode = cv2.RETR_CCOMP if draw_internal_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(mask*255, mode, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue

        color = tuple(int(x) for x in centers_bgr[ci])  # BGR

        # Draw each contour (optionally simplified)
        for c in contours:
            if cv2.contourArea(c) < min_area_px:
                continue

            if simplify_epsilon_px and simplify_epsilon_px > 0:
                approx = cv2.approxPolyDP(c, simplify_epsilon_px, True)
            else:
                approx = c

            cv2.drawContours(out, [approx], -1, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return out

def bgr_to_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"

def export_detected_colors(centers_bgr, out_path="detected_colors.json"):
    data = []
    for i, c in enumerate(centers_bgr):
        data.append({
            "cluster": i,
            "bgr": [int(x) for x in c],
            "hex": bgr_to_hex(c)
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved detected colors → {out_path}")

def extract_regions(labels, centers_bgr,
                    close_radius=3, open_radius=1,
                    min_area_px=200, simplify_epsilon_px=2.0):
    """
    Returns a list of region dicts:
    {
      region_id,
      cluster,
      color_bgr,
      color_hex,
      area_px,
      contour: [ [x,y], ... ]
    }
    """
    regions = []
    region_id = 0

    k = centers_bgr.shape[0]

    for ci in range(k):
        mask = (labels == ci).astype(np.uint8)
        mask = clean_mask(mask, close_radius, open_radius)

        contours, _ = cv2.findContours(
            mask * 255,
            cv2.RETR_CCOMP,
            cv2.CHAIN_APPROX_NONE
        )

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_px:
                continue

            if simplify_epsilon_px > 0:
                c = cv2.approxPolyDP(c, simplify_epsilon_px, True)

            regions.append({
                "region_id": region_id,
                "cluster": ci,
                "color_bgr": [int(x) for x in centers_bgr[ci]],
                "color_hex": bgr_to_hex(centers_bgr[ci]),
                "area_px": float(area),
                "contour": c.reshape(-1, 2).tolist()
            })

            region_id += 1

    return regions

def export_regions(regions, out_path="regions.json"):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(regions, f, indent=2)
    print(f"Saved regions → {out_path}")

def export_label_image(labels, out_path="labels.png"):
    # Normalize labels to 0–255
    lab = (labels.astype(np.float32) / labels.max() * 255).astype(np.uint8)
    cv2.imwrite(out_path, lab)



def main(
    input_path,
    output_path="classified_outlines.png",
    k=10,
    # If you want a fixed 10-color palette, put 10 BGR colors here and set use_palette=True
    use_palette=False,
    palette_bgr=None,
    # Cleanup & vector-ish simplification knobs
    close_radius=3,
    open_radius=1,
    min_area_px=200,
    simplify_epsilon_px=2.0,
    line_thickness=2,
    background="white",
    max_dim=1400
    
):
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read: {input_path}")

    # Optional resize for speed/consistency
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    labels, centers_bgr = quantize_kmeans_lab(bgr, k=k)

    if use_palette:
        if palette_bgr is None:
            raise ValueError("use_palette=True but palette_bgr is None")
        palette_bgr = np.array(palette_bgr, dtype=np.uint8).reshape((-1, 3))
        centers_bgr = map_centers_to_palette(centers_bgr, palette_bgr)

    out = draw_region_outlines(
        labels=labels,
        centers_bgr=centers_bgr,
        out_shape=bgr.shape,
        line_thickness=line_thickness,
        close_radius=close_radius,
        open_radius=open_radius,
        min_area_px=min_area_px,
        simplify_epsilon_px=simplify_epsilon_px,
        draw_internal_holes=True,
        background=background
    )
    labels, centers_bgr = quantize_kmeans_lab(bgr, k=k)

    export_detected_colors(centers_bgr, "detected_colors.json")
    regions = extract_regions(
        labels,
        centers_bgr,
        close_radius=close_radius,
        open_radius=open_radius,
        min_area_px=min_area_px,
        simplify_epsilon_px=simplify_epsilon_px
    )
    export_regions(regions, "regions.json")
    export_label_image(labels, "labels.png")
    cv2.imwrite(str(output_path), out)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    # ---- EDIT THESE ----
    INPUT_IMAGE = "input.jpg"  # <-- your crayon image
    OUTPUT_IMAGE = "classified_outlines.png"

    # Example fixed 10-color palette (BGR). Replace with your allowed colors.
    # palette_bgr_example = [
    #     (0,0,0), (255,255,255), (0,0,255), (0,255,0), (255,0,0),
    #     (0,255,255), (255,0,255), (255,255,0), (128,128,128), (0,128,255)
    # ]

    main(
        input_path=INPUT_IMAGE,
        output_path=OUTPUT_IMAGE,
        k=10,
        use_palette=False,          # set True if you want snapping to your restricted palette
        palette_bgr=None,           # set to palette_bgr_example if use_palette=True
        close_radius=3,
        open_radius=1,
        min_area_px=200,
        simplify_epsilon_px=2.0,
        line_thickness=2,
        background="white",
        max_dim=1400
    )
