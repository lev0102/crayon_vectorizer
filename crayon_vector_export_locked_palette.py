import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1) LOCKED PALETTE (edit here once; no CLI needed)
#    Format is BGR because OpenCV uses BGR.
# ============================================================
DEFAULT_PALETTE = [
    {"palette_id": 0, "name": "BLUE",   "bgr": (255,   0,   0)},
    {"palette_id": 1, "name": "ORANGE", "bgr": (  0, 165, 255)},
    {"palette_id": 2, "name": "GREEN",  "bgr": (  0, 255,   0)},
    {"palette_id": 3, "name": "RED",    "bgr": (  0,   0, 255)},
    {"palette_id": 4, "name": "GREY",   "bgr": (128, 128, 128)},
    {"palette_id": 5, "name": "BLACK",  "bgr": (  0,   0,   0)},
    {"palette_id": 6, "name": "BROWN",  "bgr": ( 25,  80, 140)},  # tweak to match your brown crayon
]


# -----------------------------
# Utils
# -----------------------------
def bgr_to_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def bgr_to_xyz_dict(bgr):
    # Unreal-friendly (Vector): x=b, y=g, z=r
    return {"x": int(bgr[0]), "y": int(bgr[1]), "z": int(bgr[2])}


def out_path_for(input_path: Path, suffix: str) -> Path:
    return input_path.parent / f"output_{input_path.stem}_{suffix}"


def clean_mask(mask01, close_radius=3, open_radius=1):
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


def parse_palette_bgr(palette_str):
    """
    Optional override via CLI:
      "b,g,r;b,g,r;..."
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


def build_locked_palette_arrays():
    """
    Returns:
      palette_bgr: uint8 Px3
      palette_names: list[str]
    """
    palette_bgr = np.array([list(p["bgr"]) for p in DEFAULT_PALETTE], dtype=np.uint8)
    palette_names = [p["name"] for p in DEFAULT_PALETTE]
    return palette_bgr, palette_names


def snap_centers_to_palette_with_ids(centers_bgr, palette_bgr, palette_names):
    """
    For each k-means center:
      - find nearest palette color in Lab
      - return snapped centers + palette_id per cluster + palette_name per cluster
    """
    centers_lab = cv2.cvtColor(
        centers_bgr.reshape((-1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape((-1, 3)).astype(np.float32)

    palette_lab = cv2.cvtColor(
        palette_bgr.reshape((-1, 1, 3)).astype(np.uint8), cv2.COLOR_BGR2LAB
    ).reshape((-1, 3)).astype(np.float32)

    d = ((centers_lab[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)  # k x P
    palette_id = np.argmin(d, axis=1).astype(np.int32)

    snapped = palette_bgr[palette_id]  # k x 3
    snapped_names = [palette_names[i] for i in palette_id.tolist()]
    return snapped, palette_id, snapped_names


# -----------------------------
# Rendering: Fill + Outlines
# -----------------------------
def reconstruct_filled_from_clusters(labels, centers_bgr, out_shape,
                                     close_radius=3, open_radius=1,
                                     min_area_px=200, simplify_epsilon_px=2.0,
                                     background_bgr=(255, 255, 255),
                                     hole_mode="hierarchy"):
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


# -----------------------------
# Vector JSON Export (minimal structure changes)
# -----------------------------
def contour_to_points_obj(c):
    pts = c.reshape(-1, 2)
    return [{"x": int(x), "y": int(y)} for x, y in pts]


def contour_to_points_raw(c):
    pts = c.reshape(-1, 2)
    return [[int(x), int(y)] for x, y in pts]


def export_vector_json(labels, centers_bgr, palette_id_per_cluster, palette_name_per_cluster,
                       out_shape,
                       close_radius=3, open_radius=1,
                       min_area_px=200, simplify_epsilon_px=2.0,
                       include_holes=True):
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

        # snapped-to-palette color for this kmeans cluster
        snapped_bgr = centers_bgr[ci]
        color_hex = bgr_to_hex(snapped_bgr)

        # hierarchy: [next, prev, first_child, parent]
        if hierarchy is not None:
            hierarchy = hierarchy[0]

        pal_id = int(palette_id_per_cluster[ci])
        pal_name = str(palette_name_per_cluster[ci])

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

                # keep your existing meaning:
                "cluster_id": int(ci),  # still k-means index

                # NEW (what you want for Unreal spawning):
                "palette_id": pal_id,
                "palette_name": pal_name,

                "is_hole": bool(is_hole),

                # keep same keys, but make them BP-friendly while keeping legacy too
                "color_bgr": bgr_to_xyz_dict(snapped_bgr),
                "color_bgr_raw": [int(x) for x in snapped_bgr],
                "color_hex": color_hex,

                "area_px": area,
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "centroid": {"x": cx, "y": cy},
                "closed": True,

                # BP-friendly points + legacy points:
                "points": contour_to_points_obj(c),
                "points_raw": contour_to_points_raw(c),
            })
            region_id += 1

    # clusters list: keep it, but add palette_id/name for each kmeans cluster
    clusters = []
    for i, c in enumerate(centers_bgr):
        clusters.append({
            "cluster_id": int(i),
            "palette_id": int(palette_id_per_cluster[i]),
            "palette_name": str(palette_name_per_cluster[i]),
            "color_bgr": bgr_to_xyz_dict(c),
            "color_bgr_raw": [int(x) for x in c],
            "color_hex": bgr_to_hex(c)
        })

    # palette list: small addition (optional but useful in UE)
    palette_list = [{
        "palette_id": int(p["palette_id"]),
        "palette_name": str(p["name"]),
        "color_bgr": bgr_to_xyz_dict(p["bgr"]),
        "color_bgr_raw": [int(x) for x in p["bgr"]],
        "color_hex": bgr_to_hex(p["bgr"])
    } for p in DEFAULT_PALETTE]

    return {
        "image": {"width": int(w), "height": int(h)},
        "clusters": clusters,   # still present
        "palette": palette_list,  # small addition; can ignore if you want
        "regions": regions
    }


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Non-ML: kmeans + LOCKED semantic palette snapping + outlines/fill + UE-friendly JSON"
    )

    ap.add_argument("--input", required=True, help="Path to input image (jpg/png).")

    # Outputs
    ap.add_argument("--outlines", action="store_true", help="Write outline image.")
    ap.add_argument("--fill", action="store_true", help="Write filled reconstruction image.")
    ap.add_argument("--json", action="store_true", help="Write vector JSON export.")
    ap.add_argument("--all", action="store_true", help="Write outlines + fill + json.")

    # Quantization options
    ap.add_argument("--k", type=int, default=14, help="Number of kmeans clusters BEFORE snapping (default: 14).")
    ap.add_argument("--max-dim", type=int, default=1600, help="Resize so max dimension <= this (default: 1600).")

    # Cleanup & simplification
    ap.add_argument("--close", type=int, default=3, help="Morph close radius (default: 3).")
    ap.add_argument("--open", dest="open_", type=int, default=1, help="Morph open radius (default: 1).")
    ap.add_argument("--min-area", type=int, default=300, help="Min contour area to keep (default: 300).")
    ap.add_argument("--epsilon", type=float, default=3.0, help="Contour simplify epsilon (default: 3.0).")

    # Fill hole handling
    ap.add_argument("--hole-mode", choices=["evenodd", "hierarchy"], default="hierarchy",
                    help="Fill hole handling (default: hierarchy).")

    # Optional palette override (normally you ignore this)
    ap.add_argument("--palette",
                    help='Override palette in BGR: "b,g,r;b,g,r;..." (optional)')

    args = ap.parse_args()

    input_path = Path(args.input)
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    do_outlines = args.outlines or args.all
    do_fill = args.fill or args.all
    do_json = args.json or args.all
    if not (do_outlines or do_fill or do_json):
        raise SystemExit("No outputs selected. Use --outlines, --fill, --json, or --all.")

    # Resize for speed/consistency
    bgr, scale = resize_max_dim(bgr, args.max_dim)

    # Quantize (kmeans)
    labels, centers_bgr = quantize_kmeans_lab(bgr, k=args.k)

    # Build locked palette arrays
    if args.palette:
        # If override is provided, use it (names become "P0..")
        palette_bgr = parse_palette_bgr(args.palette)
        palette_names = [f"P{i}" for i in range(len(palette_bgr))]
    else:
        palette_bgr, palette_names = build_locked_palette_arrays()

    # Snap + also compute palette_id/name per kmeans cluster
    snapped_centers, palette_id_per_cluster, palette_name_per_cluster = snap_centers_to_palette_with_ids(
        centers_bgr, palette_bgr, palette_names
    )
    centers_bgr = snapped_centers  # important: everything downstream uses snapped colors

    # Outputs
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
            palette_id_per_cluster=palette_id_per_cluster,
            palette_name_per_cluster=palette_name_per_cluster,
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
            "palette_locked": bool(args.palette is None),
            "palette_override": bool(args.palette is not None)
        }

        json_path = out_path_for(input_path, "vector.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()

# Quick run command for testing
# python crayon_vector_export_locked_palette.py --input "./test.png" --all


# If you want fewer spline points for runtime:

# increase --epsilon (e.g. 4.5)

# or increase --min-area to reduce tiny fragments