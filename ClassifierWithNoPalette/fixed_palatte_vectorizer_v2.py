import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# LOCKED PALETTE (edit here once)
# Format is BGR (OpenCV).
# Added WHITE.
# ============================================================
DEFAULT_PALETTE = [
    {"palette_id": 0, "name": "BLUE",   "bgr": (255,   0,   0)},
    {"palette_id": 1, "name": "ORANGE", "bgr": (  0, 165, 255)},
    {"palette_id": 2, "name": "GREEN",  "bgr": (  0, 255,   0)},
    {"palette_id": 3, "name": "RED",    "bgr": (  0,   0, 255)},
    {"palette_id": 4, "name": "BROWN",  "bgr": ( 25,  80, 140)},  # tweak to match your brown crayon
    {"palette_id": 5, "name": "GREY",   "bgr": (128, 128, 128)},
    {"palette_id": 6, "name": "BLACK",  "bgr": (  0,   0,   0)},
    {"palette_id": 7, "name": "WHITE",  "bgr": (255, 255, 255)},
]


# -----------------------------
# Utils
# -----------------------------
def bgr_to_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def bgr_to_xyz_dict(bgr):
    # Unreal-friendly Vector storage: x=b, y=g, z=r
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
def quantize_kmeans_lab(bgr, k=14, attempts=3, max_iter=30):
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


def build_locked_palette():
    palette_bgr = np.array([list(p["bgr"]) for p in DEFAULT_PALETTE], dtype=np.uint8)
    palette_names = [p["name"] for p in DEFAULT_PALETTE]
    palette_ids = np.array([p["palette_id"] for p in DEFAULT_PALETTE], dtype=np.int32)
    name_to_id = {p["name"]: p["palette_id"] for p in DEFAULT_PALETTE}
    return palette_bgr, palette_names, palette_ids, name_to_id


def _bgr_to_hsv01(bgr_u8):
    """Return HSV components in OpenCV ranges: H[0..179], S[0..255], V[0..255]."""
    bgr_u8 = np.array(bgr_u8, dtype=np.uint8).reshape((1, 1, 3))
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def snap_centers_to_palette_semantic(
    centers_bgr,
    palette_bgr,
    palette_names,
    name_to_id,
    s_grey=40,
    v_black=35,
    s_white=25,
    v_white=235
):
    """
    Key upgrade:
    - Use HSV "neutral gate" on each cluster center:
        if V <= v_black -> BLACK
        elif S <= s_white and V >= v_white -> WHITE
        elif S <= s_grey -> GREY
        else -> snap among chromatic palette colors via Lab distance

    This prevents:
    - light blue -> GREY (by lowering neutral gating for chroma-like colors)
    - white background -> ORANGE (WHITE category exists & is gated)
    """

    # Precompute Lab for palette
    palette_lab = cv2.cvtColor(
        palette_bgr.reshape((-1, 1, 3)).astype(np.uint8),
        cv2.COLOR_BGR2LAB
    ).reshape((-1, 3)).astype(np.float32)

    # Identify which palette indices are chromatic candidates
    # (exclude GREY/BLACK/WHITE from "color snapping")
    excluded = {"GREY", "BLACK", "WHITE"}
    chroma_idx = [i for i, n in enumerate(palette_names) if n not in excluded]
    chroma_idx = np.array(chroma_idx, dtype=np.int32)

    snapped = np.zeros_like(centers_bgr, dtype=np.uint8)
    palette_id_per_cluster = np.zeros((centers_bgr.shape[0],), dtype=np.int32)
    palette_name_per_cluster = [""] * centers_bgr.shape[0]

    for ci, c in enumerate(centers_bgr):
        h, s, v = _bgr_to_hsv01(c)

        # Neutral gate
        if v <= v_black:
            pname = "BLACK"
            pid = name_to_id[pname]
            pbgr = next(p["bgr"] for p in DEFAULT_PALETTE if p["name"] == pname)
            snapped[ci] = np.array(pbgr, dtype=np.uint8)
            palette_id_per_cluster[ci] = pid
            palette_name_per_cluster[ci] = pname
            continue

        if (s <= s_white) and (v >= v_white):
            pname = "WHITE"
            pid = name_to_id[pname]
            pbgr = next(p["bgr"] for p in DEFAULT_PALETTE if p["name"] == pname)
            snapped[ci] = np.array(pbgr, dtype=np.uint8)
            palette_id_per_cluster[ci] = pid
            palette_name_per_cluster[ci] = pname
            continue

        if s <= s_grey:
            pname = "GREY"
            pid = name_to_id[pname]
            pbgr = next(p["bgr"] for p in DEFAULT_PALETTE if p["name"] == pname)
            snapped[ci] = np.array(pbgr, dtype=np.uint8)
            palette_id_per_cluster[ci] = pid
            palette_name_per_cluster[ci] = pname
            continue

        # Chromatic snap using Lab distance to chroma palette entries
        c_lab = cv2.cvtColor(
            np.array(c, dtype=np.uint8).reshape((1, 1, 3)),
            cv2.COLOR_BGR2LAB
        ).reshape((3,)).astype(np.float32)

        d = ((palette_lab[chroma_idx] - c_lab[None, :]) ** 2).sum(axis=1)
        best_local = int(np.argmin(d))
        best_palette_index = int(chroma_idx[best_local])

        snapped[ci] = palette_bgr[best_palette_index]
        pname = palette_names[best_palette_index]
        pid = name_to_id[pname]
        palette_id_per_cluster[ci] = pid
        palette_name_per_cluster[ci] = pname

    return snapped, palette_id_per_cluster, palette_name_per_cluster


# -----------------------------
# Rendering
# -----------------------------
def reconstruct_filled_from_clusters(labels, centers_bgr, out_shape,
                                     close_radius=3, open_radius=1,
                                     min_area_px=300, simplify_epsilon_px=3.0,
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
                          min_area_px=300, simplify_epsilon_px=3.0,
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
# Vector JSON Export (minimal structure change)
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
                       min_area_px=300, simplify_epsilon_px=3.0,
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

        snapped_bgr = centers_bgr[ci]
        color_hex = bgr_to_hex(snapped_bgr)

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

            # Mark background in JSON if it's WHITE and huge (optional but very useful)
            is_background = False
            if pal_name == "WHITE":
                # heuristic: large area likely background
                if area >= 0.25 * (h * w):
                    is_background = True

            regions.append({
                "region_id": int(region_id),
                "cluster_id": int(ci),

                "palette_id": pal_id,
                "palette_name": pal_name,

                "is_hole": bool(is_hole),
                "is_background": bool(is_background),

                "color_bgr": bgr_to_xyz_dict(snapped_bgr),
                "color_bgr_raw": [int(x) for x in snapped_bgr],
                "color_hex": color_hex,

                "area_px": area,
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "centroid": {"x": cx, "y": cy},
                "closed": True,

                "points": contour_to_points_obj(c),
                "points_raw": contour_to_points_raw(c),
            })
            region_id += 1

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

    palette_list = [{
        "palette_id": int(p["palette_id"]),
        "palette_name": str(p["name"]),
        "color_bgr": bgr_to_xyz_dict(p["bgr"]),
        "color_bgr_raw": [int(x) for x in p["bgr"]],
        "color_hex": bgr_to_hex(p["bgr"])
    } for p in DEFAULT_PALETTE]

    return {
        "image": {"width": int(w), "height": int(h)},
        "clusters": clusters,
        "palette": palette_list,
        "regions": regions
    }


# -----------------------------
# Main / CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Non-ML: kmeans + locked semantic palette + HSV neutral gate + outlines/fill + UE-friendly JSON"
    )
    ap.add_argument("--input", required=True, help="Path to input image (jpg/png).")

    ap.add_argument("--outlines", action="store_true")
    ap.add_argument("--fill", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true")

    ap.add_argument("--k", type=int, default=16, help="kmeans clusters before snapping (default: 16).")
    ap.add_argument("--max-dim", type=int, default=1600)

    ap.add_argument("--close", type=int, default=3)
    ap.add_argument("--open", dest="open_", type=int, default=1)
    ap.add_argument("--min-area", type=int, default=300)
    ap.add_argument("--epsilon", type=float, default=3.5)

    ap.add_argument("--hole-mode", choices=["evenodd", "hierarchy"], default="hierarchy")

    # Neutral gate thresholds (tune if needed)
    ap.add_argument("--s-grey", type=int, default=40, help="If S <= this -> GREY (unless WHITE).")
    ap.add_argument("--v-black", type=int, default=35, help="If V <= this -> BLACK.")
    ap.add_argument("--s-white", type=int, default=25, help="If S <= this and V >= v-white -> WHITE.")
    ap.add_argument("--v-white", type=int, default=235, help="If V >= this and S <= s-white -> WHITE.")

    args = ap.parse_args()

    input_path = Path(args.input)
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    do_outlines = args.outlines or args.all
    do_fill = args.fill or args.all
    do_json = args.json or args.all
    if not (do_outlines or do_fill or do_json):
        raise SystemExit("Select outputs: --outlines, --fill, --json, or --all")

    bgr, scale = resize_max_dim(bgr, args.max_dim)

    labels, centers_bgr = quantize_kmeans_lab(bgr, k=args.k)

    palette_bgr, palette_names, _, name_to_id = build_locked_palette()

    # Snap with semantic gating
    snapped_centers, palette_id_per_cluster, palette_name_per_cluster = snap_centers_to_palette_semantic(
        centers_bgr=centers_bgr,
        palette_bgr=palette_bgr,
        palette_names=palette_names,
        name_to_id=name_to_id,
        s_grey=args.s_grey,
        v_black=args.v_black,
        s_white=args.s_white,
        v_white=args.v_white,
    )
    centers_bgr = snapped_centers  # downstream uses snapped colors

    if do_outlines:
        img = draw_cluster_outlines(
            labels=labels, centers_bgr=centers_bgr, out_shape=bgr.shape,
            close_radius=args.close, open_radius=args.open_,
            min_area_px=args.min_area, simplify_epsilon_px=args.epsilon,
            line_thickness=2
        )
        p = out_path_for(input_path, "outlines.png")
        cv2.imwrite(str(p), img)
        print(f"Wrote: {p}")

    if do_fill:
        img = reconstruct_filled_from_clusters(
            labels=labels, centers_bgr=centers_bgr, out_shape=bgr.shape,
            close_radius=args.close, open_radius=args.open_,
            min_area_px=args.min_area, simplify_epsilon_px=args.epsilon,
            hole_mode=args.hole_mode
        )
        p = out_path_for(input_path, "fill.png")
        cv2.imwrite(str(p), img)
        print(f"Wrote: {p}")

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
            "neutral_gate": {
                "s_grey": int(args.s_grey),
                "v_black": int(args.v_black),
                "s_white": int(args.s_white),
                "v_white": int(args.v_white)
            }
        }

        p = out_path_for(input_path, "vector.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()


# python fixed_palatte_vectorizer_v2.py --input "input/input.jpg" --all


# What to tune if light blue still becomes grey

# Lower the GREY saturation threshold so “pale but still blue” stays chromatic:

# default --s-grey 40

# try --s-grey 25 or --s-grey 20

# Example:

# python crayon_vector_export_locked_palette_v2.py --input "./test.png" --all --s-grey 20

# What to tune if white background still isn’t white

# Make WHITE gate a bit more permissive:

# default --v-white 235, --s-white 25

# try --v-white 220 (more forgiving)

# or --s-white 35 (more forgiving)

# Example:

# python crayon_vector_export_locked_palette_v2.py --input "./test.png" --all --v-white 220 --s-white 35
