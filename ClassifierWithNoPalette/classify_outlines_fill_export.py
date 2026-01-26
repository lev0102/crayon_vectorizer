import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def bgr_to_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def make_output_path(input_path, filename):
    p = Path(input_path)
    return str(p.parent / filename)


def clean_mask(mask01, close_radius=3, open_radius=1):
    """
    mask01: uint8 {0,1}
    Returns uint8 {0,1} cleaned mask.
    """
    m = (mask01.astype(np.uint8) * 255)

    if close_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if open_radius > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_radius * 2 + 1, open_radius * 2 + 1)
        )
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    return (m > 0).astype(np.uint8)


# -----------------------------
# Color quantization (non-ML)
# -----------------------------
def quantize_kmeans_lab(bgr, k=10, attempts=3, max_iter=30):
    """
    Quantize image into k colors using k-means in Lab space.
    Returns:
      labels: int32 HxW (0..k-1)
      centers_bgr: uint8 kx3
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    Z = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape((h, w)).astype(np.int32)
    centers = centers.astype(np.uint8)

    centers_lab = centers.reshape((k, 1, 3))
    centers_bgr = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR).reshape((k, 3))
    return labels, centers_bgr


def map_centers_to_palette(centers_bgr, palette_bgr):
    """
    Snap each cluster center to nearest palette color (in Lab space).
    palette_bgr: uint8 Px3
    """
    centers_lab = cv2.cvtColor(
        centers_bgr.reshape((-1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape((-1, 3)).astype(np.float32)

    palette_lab = cv2.cvtColor(
        palette_bgr.reshape((-1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape((-1, 3)).astype(np.float32)

    d = ((centers_lab[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    idx = np.argmin(d, axis=1)
    return palette_bgr[idx]


# -----------------------------
# Vector-ish filled reconstruction
# -----------------------------
def reconstruct_filled_from_clusters(labels, centers_bgr, out_shape,
                                     close_radius=3, open_radius=1,
                                     min_area_px=200, simplify_epsilon_px=2.0,
                                     background_bgr=(255, 255, 255),
                                     hole_mode="evenodd"):
    """
    Render a simplified filled image by extracting contours for each cluster and filling them.
    This "flattens" crayon texture into clean regions.

    hole_mode:
      - "evenodd": fill all contours (common good look; holes appear filled if not separated)
      - "hierarchy": parent filled, child subtracted (more correct holes, a bit more logic)
    """
    h, w = out_shape[:2]
    out = np.full((h, w, 3), background_bgr, dtype=np.uint8)

    k = int(centers_bgr.shape[0])

    for ci in range(k):
        mask = (labels == ci).astype(np.uint8)
        mask = clean_mask(mask, close_radius=close_radius, open_radius=open_radius)

        if int(mask.sum()) < min_area_px:
            continue

        contours, hierarchy = cv2.findContours(
            mask * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            continue

        color = tuple(int(x) for x in centers_bgr[ci])

        if hole_mode == "evenodd":
            for c in contours:
                if cv2.contourArea(c) < min_area_px:
                    continue
                if simplify_epsilon_px > 0:
                    c = cv2.approxPolyDP(c, simplify_epsilon_px, True)
                cv2.fillPoly(out, [c], color, lineType=cv2.LINE_AA)

        elif hole_mode == "hierarchy" and hierarchy is not None:
            # Fill parents with color, subtract children with background (simple hole support)
            hierarchy = hierarchy[0]  # shape: (num_contours, 4) [next, prev, first_child, parent]
            for idx, c in enumerate(contours):
                area = cv2.contourArea(c)
                if area < min_area_px:
                    continue

                parent = hierarchy[idx][3]
                if simplify_epsilon_px > 0:
                    c = cv2.approxPolyDP(c, simplify_epsilon_px, True)

                if parent == -1:
                    # outer contour
                    cv2.fillPoly(out, [c], color, lineType=cv2.LINE_AA)
                else:
                    # hole contour
                    cv2.fillPoly(out, [c], background_bgr, lineType=cv2.LINE_AA)
        else:
            raise ValueError("hole_mode must be 'evenodd' or 'hierarchy'")

    return out


def draw_cluster_outlines(labels, centers_bgr, out_shape,
                          close_radius=3, open_radius=1,
                          min_area_px=200, simplify_epsilon_px=2.0,
                          line_thickness=2, background_bgr=(255, 255, 255)):
    """
    Optional: draw region boundaries colored by cluster color.
    """
    h, w = out_shape[:2]
    out = np.full((h, w, 3), background_bgr, dtype=np.uint8)
    k = int(centers_bgr.shape[0])

    for ci in range(k):
        mask = (labels == ci).astype(np.uint8)
        mask = clean_mask(mask, close_radius=close_radius, open_radius=open_radius)

        if int(mask.sum()) < min_area_px:
            continue

        contours, _ = cv2.findContours(mask * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue

        color = tuple(int(x) for x in centers_bgr[ci])

        for c in contours:
            if cv2.contourArea(c) < min_area_px:
                continue
            if simplify_epsilon_px > 0:
                c = cv2.approxPolyDP(c, simplify_epsilon_px, True)
            cv2.drawContours(out, [c], -1, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    return out


def export_detected_colors(centers_bgr, out_path):
    data = []
    for i, c in enumerate(centers_bgr):
        data.append({
            "cluster": int(i),
            "bgr": [int(x) for x in c],
            "hex": bgr_to_hex(c)
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# Main / CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Non-ML crayon color quantization + vector-ish filled reconstruction.")
    ap.add_argument("input", help="Path to input image (jpg/png).")
    ap.add_argument("--k", type=int, default=10, help="Number of color clusters (default: 10).")
    ap.add_argument("--max-dim", type=int, default=1600, help="Resize so max(width,height)<=max_dim (default: 1600).")

    ap.add_argument("--close", type=int, default=3, help="Morph close radius (default: 3).")
    ap.add_argument("--open", dest="open_", type=int, default=1, help="Morph open radius (default: 1).")
    ap.add_argument("--min-area", type=int, default=200, help="Min region area in pixels (default: 200).")
    ap.add_argument("--epsilon", type=float, default=2.0, help="Contour simplification epsilon (default: 2.0).")

    ap.add_argument("--hole-mode", choices=["evenodd", "hierarchy"], default="hierarchy",
                    help="How to handle holes when filling (default: hierarchy).")

    ap.add_argument("--write-outlines", action="store_true", help="Also write output_outlines.png.")
    args = ap.parse_args()

    input_path = args.input
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Resize for speed/consistency
    h, w = bgr.shape[:2]
    if max(h, w) > args.max_dim:
        scale = args.max_dim / float(max(h, w))
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Quantize
    labels, centers_bgr = quantize_kmeans_lab(bgr, k=args.k)

    # Export detected colors JSON (same folder)
    colors_json_path = make_output_path(input_path, "output_colors.json")
    export_detected_colors(centers_bgr, colors_json_path)

    # Filled reconstruction (same folder)
    fill_path = make_output_path(input_path, "output_fill.png")
    filled = reconstruct_filled_from_clusters(
        labels=labels,
        centers_bgr=centers_bgr,
        out_shape=bgr.shape,
        close_radius=args.close,
        open_radius=args.open_,
        min_area_px=args.min_area,
        simplify_epsilon_px=args.epsilon,
        background_bgr=(255, 255, 255),
        hole_mode=args.hole_mode
    )
    cv2.imwrite(fill_path, filled)

    # Optional outlines
    if args.write_outlines:
        outlines_path = make_output_path(input_path, "output_outlines.png")
        outlines = draw_cluster_outlines(
            labels=labels,
            centers_bgr=centers_bgr,
            out_shape=bgr.shape,
            close_radius=args.close,
            open_radius=args.open_,
            min_area_px=args.min_area,
            simplify_epsilon_px=args.epsilon,
            line_thickness=2,
            background_bgr=(255, 255, 255)
        )
        cv2.imwrite(outlines_path, outlines)
        print(f"Wrote: {outlines_path}")

    print(f"Wrote: {fill_path}")
    print(f"Wrote: {colors_json_path}")


if __name__ == "__main__":
    main()
