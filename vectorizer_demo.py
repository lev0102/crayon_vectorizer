import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# -----------------------------
# Utils
# -----------------------------
def bgr_to_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def out_path_for(input_path: Path, suffix: str) -> Path:
    """
    output_<input_stem>_<suffix>
    """
    return input_path.parent / f"output_{input_path.stem}_{suffix}"


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


def resize_max_dim(bgr, max_dim):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr, 1.0
    scale = max_dim / float(max(h, w))
    resized = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


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
    Snap each kmeans center to nearest palette color (in Lab space).
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
# Rendering: Fill + Outlines
# -----------------------------
def reconstruct_filled_from_clusters(labels, centers_bgr, out_shape,
                                     close_radius=3, open_radius=1,
                                     min_area_px=200, simplify_epsilon_px=2.0,
                                     background_bgr=(255, 255, 255),
                                     hole_mode="hierarchy"):
    """
    Vector-ish filled reconstruction by filling simplified contours per cluster.
    hole_mode:
      - "evenodd": fill everything (fast; holes might fill)
      - "hierarchy": fill parents, subtract children (better holes)
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
            hierarchy = hierarchy[0]
            for idx, c in enumerate(contours):
                if cv2.contourArea(c) < min_area_px:
                    continue
                if simplify_epsilon_px > 0:
                    c = cv2.approxPolyDP(c, simplify_epsilon_px, True)

                parent = hierarchy[idx][3]
                if parent == -1:
                    cv2.fillPoly(out, [c], color, lineType=cv2.LINE_AA)
                else:
                    cv2.fillPoly(out, [c], background_bgr, lineType=cv2.LINE_AA)
        else:
            raise ValueError("hole_mode must be 'evenodd' or 'hierarchy'")

    return out


def draw_cluster_outlines(labels, centers_bgr, out_shape,
                          close_radius=3, open_radius=1,
                          min_area_px=200, simplify_epsilon_px=2.0,
                          line_thickness=2, background_bgr=(255, 255, 255)):
    """
    Draw boundaries (contours) colored by cluster color.
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

def bgr_to_xyz_dict(bgr):
    """
    Unreal-friendly color vector.
    Stored as x,y,z so BP can read it as Vector.
    """
    return {
        "x": int(bgr[0]),
        "y": int(bgr[1]),
        "z": int(bgr[2])
    }


# -----------------------------
# Vector JSON Export
# -----------------------------
def contour_to_points(c):
    """
    Unreal-friendly point format:
      [{ "x": 123, "y": 456 }, ...]
    This maps cleanly to Vector2D in Blueprints (x/y fields).
    """
    pts = c.reshape(-1, 2)
    return [{"x": int(x), "y": int(y)} for x, y in pts]



def export_vector_json(labels, centers_bgr, out_shape,
                       close_radius=3, open_radius=1,
                       min_area_px=200, simplify_epsilon_px=2.0,
                       include_holes=True):
    """
    Produces vector-ready region data:
    - regions: list of region objects with color + contour points
    - centers: list of detected cluster colors
    """
    h, w = out_shape[:2]
    regions = []
    region_id = 0
    k = int(centers_bgr.shape[0])

    for ci in range(k):
        mask = (labels == ci).astype(np.uint8)
        mask = clean_mask(mask, close_radius=close_radius, open_radius=open_radius)

        if int(mask.sum()) < min_area_px:
            continue

        mode = cv2.RETR_CCOMP if include_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(mask * 255, mode, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue

        color_bgr_vec = bgr_to_xyz_dict(centers_bgr[ci])
        color_hex = bgr_to_hex(centers_bgr[ci])


        # hierarchy: [next, prev, first_child, parent]
        if hierarchy is not None:
            hierarchy = hierarchy[0]

        for idx, c in enumerate(contours):
            area = float(cv2.contourArea(c))
            if area < float(min_area_px):
                continue

            if simplify_epsilon_px > 0:
                c = cv2.approxPolyDP(c, simplify_epsilon_px, True)

            x, y, bw, bh = cv2.boundingRect(c)
            M = cv2.moments(c)
            if abs(M["m00"]) > 1e-6:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
            else:
                cx = float(x + bw * 0.5)
                cy = float(y + bh * 0.5)

            parent = int(hierarchy[idx][3]) if hierarchy is not None else -1
            is_hole = (parent != -1)

            regions.append({
                "region_id": int(region_id),
                "cluster_id": int(ci),
                "is_hole": bool(is_hole),
                "color_bgr": color_bgr_vec,
                "color_hex": color_hex,
                "area_px": area,
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "centroid": {"x": cx, "y": cy},
                "closed": True,  # contours are closed
                "points": contour_to_points(c)
            })
            region_id += 1

    centers = [{
        "cluster_id": int(i),
        "color_bgr": bgr_to_xyz_dict(c),
        "color_hex": bgr_to_hex(c)
    } for i, c in enumerate(centers_bgr)]

    return {
        "image": {"width": int(w), "height": int(h)},
        "clusters": centers,
        "regions": regions
    }


# -----------------------------
# CLI
# -----------------------------
def parse_palette_bgr(palette_str):
    """
    Parse palette string like:
      "0,0,0;255,255,255;0,0,255"
    into uint8 Nx3 BGR array.
    """
    items = palette_str.split(";")
    cols = []
    for it in items:
        parts = it.strip().split(",")
        if len(parts) != 3:
            raise ValueError(f"Bad palette entry: {it}")
        b, g, r = [int(x) for x in parts]
        cols.append([b, g, r])
    return np.array(cols, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description="Non-ML: color quantization + outlines/fill + Unreal-friendly vector JSON"
    )

    ap.add_argument("--input", required=True, help="Path to input image (jpg/png).")

    # Outputs
    ap.add_argument("--outlines", action="store_true", help="Write outline image.")
    ap.add_argument("--fill", action="store_true", help="Write filled reconstruction image.")
    ap.add_argument("--json", action="store_true", help="Write vector JSON export.")
    ap.add_argument("--all", action="store_true", help="Write outlines + fill + json.")

    # Quantization options
    ap.add_argument("--k", type=int, default=10, help="Number of clusters (default: 10).")
    ap.add_argument("--max-dim", type=int, default=1600, help="Resize so max dimension <= this (default: 1600).")

    # Cleanup & simplification
    ap.add_argument("--close", type=int, default=3, help="Morph close radius (default: 3).")
    ap.add_argument("--open", dest="open_", type=int, default=1, help="Morph open radius (default: 1).")
    ap.add_argument("--min-area", type=int, default=200, help="Min contour area to keep (default: 200).")
    ap.add_argument("--epsilon", type=float, default=2.0, help="Contour simplify epsilon (default: 2.0).")

    # Fill hole handling
    ap.add_argument("--hole-mode", choices=["evenodd", "hierarchy"], default="hierarchy",
                    help="Fill hole handling (default: hierarchy).")

    # Optional palette snapping
    ap.add_argument("--palette",
                    help='Optional fixed palette in BGR: "b,g,r;b,g,r;..." '
                         '(example: "0,0,0;255,255,255;0,0,255")')

    args = ap.parse_args()

    input_path = Path(args.input)
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Decide outputs
    do_outlines = args.outlines or args.all
    do_fill = args.fill or args.all
    do_json = args.json or args.all

    if not (do_outlines or do_fill or do_json):
        raise SystemExit("No outputs selected. Use --outlines, --fill, --json, or --all.")

    # Resize for speed/consistency
    bgr, scale = resize_max_dim(bgr, args.max_dim)

    # Quantize
    labels, centers_bgr = quantize_kmeans_lab(bgr, k=args.k)

    # Optional: snap cluster centers to fixed palette
    if args.palette:
        palette_bgr = parse_palette_bgr(args.palette)
        centers_bgr = map_centers_to_palette(centers_bgr, palette_bgr)

    # Outputs use input stem
    if do_outlines:
        outlines_img = draw_cluster_outlines(
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
        outlines_path = out_path_for(input_path, "outlines.png")
        cv2.imwrite(str(outlines_path), outlines_img)
        print(f"Wrote: {outlines_path}")

    if do_fill:
        fill_img = reconstruct_filled_from_clusters(
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
        fill_path = out_path_for(input_path, "fill.png")
        cv2.imwrite(str(fill_path), fill_img)
        print(f"Wrote: {fill_path}")

    if do_json:
        data = export_vector_json(
            labels=labels,
            centers_bgr=centers_bgr,
            out_shape=bgr.shape,
            close_radius=args.close,
            open_radius=args.open_,
            min_area_px=args.min_area,
            simplify_epsilon_px=args.epsilon,
            include_holes=True
        )
        data["meta"] = {
            "input": str(input_path.name),
            "scaled": bool(scale != 1.0),
            "scale_factor": float(scale),
            "k": int(args.k),
            "close_radius": int(args.close),
            "open_radius": int(args.open_),
            "min_area_px": int(args.min_area),
            "epsilon": float(args.epsilon),
            "hole_mode": str(args.hole_mode),
            "palette_snapped": bool(args.palette is not None)
        }

        json_path = out_path_for(input_path, "vector.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()


# python vectorizer_demo.py --input "input2.jpg" --all
# python vectorizer_demo.py --input "./smiley.png" --outlines --json
# python vectorizer_demo.py --input "./smiley.png" --fill
# Force to a fixed palette (example)
# python vectorizer_demo.py --input "./smiley.png" --all \
#   --palette "0,0,0;255,255,255;0,0,255;0,255,0;255,0,0;0,255,255;255,0,255;255,255,0;128,128,128;0,128,255"
# python crayon_vector_export.py --input "./img.png" --all --k 14 \
#   --palette "255,0,0;0,165,255;0,255,0;0,0,255;128,128,128;0,0,0;25,80,140"



