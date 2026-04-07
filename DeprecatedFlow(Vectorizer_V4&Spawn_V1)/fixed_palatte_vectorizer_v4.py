#!/usr/bin/env python3

# python fixed_palatte_vectorizer_v4.py --input input/Drawing0.jpg --palette-json output/output_palette_Drawing0.json --all --s-low 7 --v-black 70 --v-brown-max 130

"""
fixed_palatte_vectorizer_v4_palette_json.py

v4 (JSON-palette) unified vectorizer:
- Requires --input (matches your current CLI usage)
- Loads palette entries from picker JSON via --palette-json
- Supports duplicate palette names (e.g., multiple RED shades all named "RED")
- Adds BACKGROUND/paper gate: very bright + low saturation => "BACKGROUND" and skipped (no splines)
- Produces:
  - outlines PNG (optional)
  - fill PNG (optional)
  - vector JSON (UE-friendly points + palette name)

Notes:
- Palette entries are expected to contain: {"name": "...", "bgr": [b,g,r]}
  (The updated picker exports this as "palette_entries".)
- If your picker uses "color_bgr_raw" instead of "bgr", this loader supports that too.
"""

"""
To tweak grey snapping behavior, adjust these parameters:
A) Reduce grey snapping (make GREY gate harder to hit)

Lower --s-low so light-but-colored strokes don’t get treated as neutral:

python fixed_palatte_vectorizer_v4_palette_json.py --input input/Drawing1.jpg --palette-json output/output_palette_Drawing0.json --all --s-low 18


If it still snaps too much to GREY, go lower:

--s-low 12

B) Stop black → brown (make BLACK gate easier to hit)

Increase --v-black (so “dark-ish” low-sat strokes become BLACK sooner):

--v-black 70


A good combo to try:

python fixed_palatte_vectorizer_v4_palette_json.py --input input/Drawing1.jpg --palette-json output/output_palette_Drawing0.json --all --s-low 18 --v-black 70

C) Reduce brown stealing neutrals

Lower --v-brown-max so low-sat pixels don’t auto-become brown unless truly dark:

--v-brown-max 130
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Utils
# ============================================================

def bgr_to_hex(bgr):
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def bgr_to_xyz_dict(bgr):
    # Unreal-friendly Vector: x=b, y=g, z=r
    return {"x": int(bgr[0]), "y": int(bgr[1]), "z": int(bgr[2])}


def resize_max_dim(bgr, max_dim):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr, 1.0
    scale = max_dim / float(max(h, w))
    resized = cv2.resize(
        bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )
    return resized, scale


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


def out_path_for(input_path: Path, suffix: str, out_dir: Path | None = None) -> Path:
    base = out_dir if out_dir is not None else input_path.parent
    base.mkdir(parents=True, exist_ok=True)
    return base / f"output_{input_path.stem}_{suffix}"


# ============================================================
# Palette loading (from picker JSON)
# ============================================================

def _extract_entries(data: dict):
    # Prefer picker format
    if isinstance(data, dict):
        if isinstance(data.get("palette_entries"), list):
            return data["palette_entries"]
        if isinstance(data.get("palette"), list):
            return data["palette"]
        if isinstance(data.get("entries"), list):
            return data["entries"]
    return []


def load_palette_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = _extract_entries(data)
    if not entries:
        raise ValueError(
            f"Palette JSON '{path}' does not contain 'palette_entries' (or 'palette'/'entries')."
        )

    palette_bgr = []
    palette_names = []
    palette_ids = []

    for i, p in enumerate(entries):
        if not isinstance(p, dict):
            continue

        name = p.get("name") or p.get("palette_name") or p.get("label")
        bgr = p.get("bgr") or p.get("color_bgr_raw") or p.get("color_bgr")

        if name is None or bgr is None:
            continue

        # bgr could be list, tuple, or dict {"x","y","z"}
        if isinstance(bgr, dict):
            bgr_list = [int(bgr.get("x", 0)), int(bgr.get("y", 0)), int(bgr.get("z", 0))]
        else:
            bgr_list = [int(bgr[0]), int(bgr[1]), int(bgr[2])]

        palette_names.append(str(name))
        palette_bgr.append(bgr_list)
        palette_ids.append(i)

    if not palette_bgr:
        raise ValueError(f"Palette JSON '{path}' had no valid entries with name+bgr.")

    palette_bgr = np.array(palette_bgr, dtype=np.uint8)
    palette_ids = np.array(palette_ids, dtype=np.int32)

    name_to_indices: dict[str, list[int]] = {}
    for i, n in enumerate(palette_names):
        name_to_indices.setdefault(n, []).append(i)

    return palette_bgr, palette_names, palette_ids, name_to_indices, entries


# ============================================================
# Quantization (non-ML)
# ============================================================

def quantize_kmeans_lab(bgr, k=16, attempts=3, max_iter=30):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    Z = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    labels = labels.reshape((h, w)).astype(np.int32)
    centers = centers.astype(np.uint8)

    centers_lab = centers.reshape((k, 1, 3))
    centers_bgr = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR).reshape((k, 3))
    return labels, centers_bgr


# ============================================================
# Semantic snapping with BACKGROUND gate
# ============================================================

def snap_centers_to_palette_semantic(
    centers_bgr,
    palette_bgr,
    palette_names,
    palette_ids,
    name_to_indices,
    v_black=35,
    v_white=235,
    s_white=25,
    s_low=35,
    v_brown_max=170,
    v_bg=225,
    s_bg=45,
    bg_name="BACKGROUND",
):
    """
    Hue-gated semantic snapping:
    - BACKGROUND: (V >= v_bg and S <= s_bg) => BG (pid=-1), skipped later
    - BLACK / WHITE gates
    - low-sat: GREY vs BROWN by V
    - chromatic: restrict candidates by Hue bands, then snap in Lab
    """

    palette_lab = cv2.cvtColor(
        palette_bgr.reshape((-1, 1, 3)).astype(np.uint8),
        cv2.COLOR_BGR2LAB
    ).reshape((-1, 3)).astype(np.float32)

    def pick_named_first(name: str):
        # Use first entry of a name for gates (BLACK/WHITE/GREY/BROWN)
        idx = name_to_indices.get(name, [0])[0]
        return palette_bgr[idx], palette_names[idx], int(palette_ids[idx])

    def hue_band_candidates(h):
        # OpenCV HSV: H 0..179
        if 90 <= h <= 140:
            return ["BLUE"]
        if 35 <= h <= 85:
            return ["GREEN"]
        if 10 <= h < 35:
            return ["ORANGE", "BROWN", "YELLOW", "RED", "SKIN"]
        return ["RED", "ORANGE", "BROWN", "YELLOW", "SKIN"]

    snapped = np.zeros_like(centers_bgr, dtype=np.uint8)
    palette_id_per_cluster = np.zeros((centers_bgr.shape[0],), dtype=np.int32)
    palette_name_per_cluster = [""] * centers_bgr.shape[0]

    for ci, c in enumerate(centers_bgr):
        hsv = cv2.cvtColor(np.array(c, np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # ---- BACKGROUND / paper gate ----
        if (v >= v_bg) and (s <= s_bg):
            snapped[ci] = np.array([255, 255, 255], dtype=np.uint8)
            palette_id_per_cluster[ci] = -1
            palette_name_per_cluster[ci] = bg_name
            continue

        # ---- Neutrals ----
        if v <= v_black:
            pbgr, pname, pid = pick_named_first("BLACK")
            snapped[ci] = pbgr
            palette_id_per_cluster[ci] = pid
            palette_name_per_cluster[ci] = pname
            continue

        if (v >= v_white) and (s <= s_white):
            pbgr, pname, pid = pick_named_first("WHITE")
            snapped[ci] = pbgr
            palette_id_per_cluster[ci] = pid
            palette_name_per_cluster[ci] = pname
            continue

        if s <= s_low:
            if v <= v_brown_max:
                pbgr, pname, pid = pick_named_first("BROWN")
            else:
                pbgr, pname, pid = pick_named_first("GREY")
            snapped[ci] = pbgr
            palette_id_per_cluster[ci] = pid
            palette_name_per_cluster[ci] = pname
            continue

        # ---- Chromatic hue-gated Lab snap ----
        candidates = hue_band_candidates(h)
        cand_idx: list[int] = []
        for nm in candidates:
            cand_idx.extend(name_to_indices.get(nm, []))  # ALL indices for that name
        if not cand_idx:
            # fallback: allow everything except neutrals if possible
            cand_idx = [i for i, n in enumerate(palette_names) if n not in ("GREY", "BLACK", "WHITE")]

        cand_idx = np.array(cand_idx, dtype=np.int32)
        c_lab = cv2.cvtColor(np.array(c, np.uint8).reshape(1, 1, 3), cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        d = ((palette_lab[cand_idx] - c_lab[None, :]) ** 2).sum(axis=1)
        best = int(cand_idx[int(np.argmin(d))])

        snapped[ci] = palette_bgr[best]
        palette_id_per_cluster[ci] = int(palette_ids[best])
        palette_name_per_cluster[ci] = str(palette_names[best])

    return snapped, palette_id_per_cluster, palette_name_per_cluster


# ============================================================
# Rendering
# ============================================================

def draw_cluster_outlines(
    labels,
    centers_bgr,
    out_shape,
    palette_name_per_cluster=None,
    skip_names=("BACKGROUND",),
    close_radius=3,
    open_radius=1,
    min_area_px=300,
    simplify_epsilon_px=3.5,
    line_thickness=2,
    background_bgr=(255, 255, 255),
):
    h, w = out_shape[:2]
    out = np.full((h, w, 3), background_bgr, dtype=np.uint8)

    for ci in range(int(centers_bgr.shape[0])):
        if palette_name_per_cluster is not None:
            if str(palette_name_per_cluster[ci]) in set(skip_names):
                continue

        mask = clean_mask((labels == ci).astype(np.uint8), close_radius, open_radius)
        if int(mask.sum()) < min_area_px:
            continue

        contours, _ = cv2.findContours(mask * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        color = tuple(int(x) for x in centers_bgr[ci])
        for c in contours:
            if cv2.contourArea(c) < min_area_px:
                continue
            if simplify_epsilon_px > 0:
                c = cv2.approxPolyDP(c, simplify_epsilon_px, True)
            cv2.drawContours(out, [c], -1, color, thickness=line_thickness, lineType=cv2.LINE_AA)
    return out


def reconstruct_filled_from_clusters(
    labels,
    centers_bgr,
    out_shape,
    palette_name_per_cluster=None,
    skip_names=("BACKGROUND",),
    close_radius=3,
    open_radius=1,
    min_area_px=300,
    simplify_epsilon_px=3.5,
    background_bgr=(255, 255, 255),
    hole_mode="hierarchy",
):
    h, w = out_shape[:2]
    out = np.full((h, w, 3), background_bgr, dtype=np.uint8)

    for ci in range(int(centers_bgr.shape[0])):
        if palette_name_per_cluster is not None:
            if str(palette_name_per_cluster[ci]) in set(skip_names):
                continue

        mask = clean_mask((labels == ci).astype(np.uint8), close_radius, open_radius)
        if int(mask.sum()) < min_area_px:
            continue

        contours, hierarchy = cv2.findContours(mask * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not contours:
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


# ============================================================
# Vector JSON Export
# ============================================================

def contour_to_points_obj(c):
    pts = c.reshape(-1, 2)
    return [{"x": int(x), "y": int(y)} for x, y in pts]


def export_vector_json(
    labels,
    centers_bgr,
    palette_id_per_cluster,
    palette_name_per_cluster,
    out_shape,
    palette_entries_original,
    close_radius=3,
    open_radius=1,
    min_area_px=300,
    simplify_epsilon_px=3.5,
    include_holes=True,
    skip_names=("BACKGROUND",),
):
    h, w = out_shape[:2]
    regions = []
    region_id = 0
    k = int(centers_bgr.shape[0])

    for ci in range(k):
        name = str(palette_name_per_cluster[ci])
        if name in set(skip_names):
            continue

        mask = clean_mask((labels == ci).astype(np.uint8), close_radius, open_radius)
        if int(mask.sum()) < min_area_px:
            continue

        mode = cv2.RETR_CCOMP if include_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(mask * 255, mode, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        if hierarchy is not None:
            hierarchy = hierarchy[0]

        for idx, c in enumerate(contours):
            area = float(cv2.contourArea(c))
            if area < float(min_area_px):
                continue

            if simplify_epsilon_px > 0:
                c = cv2.approxPolyDP(c, simplify_epsilon_px, True)

            x, y, bw, bh = cv2.boundingRect(c)

            is_hole = False
            if hierarchy is not None:
                parent = int(hierarchy[idx][3])
                is_hole = (parent != -1)

            regions.append({
                "region_id": int(region_id),
                "cluster_id": int(ci),
                "palette_id": int(palette_id_per_cluster[ci]),
                "palette_name": name,
                "is_hole": bool(is_hole),
                "color_bgr": bgr_to_xyz_dict(centers_bgr[ci]),
                "color_hex": bgr_to_hex(centers_bgr[ci]),
                "area_px": area,
                "bbox": {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)},
                "closed": True,
                "points": contour_to_points_obj(c),
            })
            region_id += 1

    # echo palette used for snapping
    palette_out = []
    for i, p in enumerate(palette_entries_original):
        name = p.get("name") or p.get("palette_name") or p.get("label") or f"ENTRY_{i}"
        bgr = p.get("bgr") or p.get("color_bgr_raw") or p.get("color_bgr")
        if isinstance(bgr, dict):
            bgr_list = [int(bgr.get("x", 0)), int(bgr.get("y", 0)), int(bgr.get("z", 0))]
        else:
            bgr_list = [int(bgr[0]), int(bgr[1]), int(bgr[2])]
        palette_out.append({
            "entry_index": int(i),
            "name": str(name),
            "bgr": bgr_to_xyz_dict(bgr_list),
            "bgr_raw": [int(x) for x in bgr_list],
            "hex": bgr_to_hex(bgr_list),
        })

    return {
        "image": {"width": int(w), "height": int(h)},
        "palette_entries": palette_out,
        "regions": regions,
    }


# ============================================================
# Main / CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Crayon vectorizer v4: kmeans (Lab) + palette JSON snapping + BACKGROUND gate + UE-friendly JSON"
    )
    ap.add_argument("--input", required=True, help="Path to input image (jpg/png).")
    ap.add_argument("--palette-json", required=True, help="Palette JSON exported by the picker.")

    ap.add_argument("--outdir", default="", help="Optional output directory. Default: next to input image.")
    ap.add_argument("--outlines", action="store_true", help="Write outlines PNG.")
    ap.add_argument("--fill", action="store_true", help="Write filled PNG.")
    ap.add_argument("--json", action="store_true", help="Write vector JSON.")
    ap.add_argument("--all", action="store_true", help="Write outlines+fill+json.")

    ap.add_argument("--k", type=int, default=16, help="kmeans clusters before snapping (default: 16).")
    ap.add_argument("--max-dim", type=int, default=1600, help="Resize max dimension for speed.")

    ap.add_argument("--close", type=int, default=3, help="Morph close radius.")
    ap.add_argument("--open", dest="open_", type=int, default=1, help="Morph open radius.")
    ap.add_argument("--min-area", type=int, default=300, help="Minimum area in px to keep.")
    ap.add_argument("--epsilon", type=float, default=3.5, help="Contour simplify epsilon in px.")
    ap.add_argument("--hole-mode", choices=["evenodd", "hierarchy"], default="hierarchy")

    # Neutral gate thresholds
    ap.add_argument("--s-low", type=int, default=35)
    ap.add_argument("--v-brown-max", type=int, default=170)
    ap.add_argument("--v-black", type=int, default=35)
    ap.add_argument("--s-white", type=int, default=25)
    ap.add_argument("--v-white", type=int, default=235)

    # Background/paper gate: very bright + low saturation => BACKGROUND
    ap.add_argument("--s-bg", type=int, default=45)
    ap.add_argument("--v-bg", type=int, default=225)

    args = ap.parse_args()

    input_path = Path(args.input)
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    out_dir = Path(args.outdir) if args.outdir else None

    do_outlines = args.outlines or args.all
    do_fill = args.fill or args.all
    do_json = args.json or args.all
    if not (do_outlines or do_fill or do_json):
        raise SystemExit("Select outputs: --outlines, --fill, --json, or --all")

    bgr, scale = resize_max_dim(bgr, args.max_dim)

    labels, centers_bgr = quantize_kmeans_lab(bgr, k=args.k)

    palette_bgr, palette_names, palette_ids, name_to_indices, entries_original = load_palette_json(Path(args.palette_json))

    snapped_centers, palette_id_per_cluster, palette_name_per_cluster = snap_centers_to_palette_semantic(
        centers_bgr=centers_bgr,
        palette_bgr=palette_bgr,
        palette_names=palette_names,
        palette_ids=palette_ids,
        name_to_indices=name_to_indices,
        v_black=args.v_black,
        v_white=args.v_white,
        s_white=args.s_white,
        s_low=args.s_low,
        v_brown_max=args.v_brown_max,
        v_bg=args.v_bg,
        s_bg=args.s_bg,
        bg_name="BACKGROUND",
    )

    centers_bgr = snapped_centers

    if do_outlines:
        img = draw_cluster_outlines(
            labels=labels,
            centers_bgr=centers_bgr,
            out_shape=bgr.shape,
            palette_name_per_cluster=palette_name_per_cluster,
            close_radius=args.close,
            open_radius=args.open_,
            min_area_px=args.min_area,
            simplify_epsilon_px=args.epsilon,
            line_thickness=2,
        )
        p = out_path_for(input_path, "outlines.png", out_dir)
        cv2.imwrite(str(p), img)
        print(f"Wrote: {p}")

    if do_fill:
        img = reconstruct_filled_from_clusters(
            labels=labels,
            centers_bgr=centers_bgr,
            out_shape=bgr.shape,
            palette_name_per_cluster=palette_name_per_cluster,
            close_radius=args.close,
            open_radius=args.open_,
            min_area_px=args.min_area,
            simplify_epsilon_px=args.epsilon,
            hole_mode=args.hole_mode,
        )
        p = out_path_for(input_path, "fill.png", out_dir)
        cv2.imwrite(str(p), img)
        print(f"Wrote: {p}")

    if do_json:
        data = export_vector_json(
            labels=labels,
            centers_bgr=centers_bgr,
            palette_id_per_cluster=palette_id_per_cluster,
            palette_name_per_cluster=palette_name_per_cluster,
            out_shape=bgr.shape,
            palette_entries_original=entries_original,
            close_radius=args.close,
            open_radius=args.open_,
            min_area_px=args.min_area,
            simplify_epsilon_px=args.epsilon,
            include_holes=True,
            skip_names=("BACKGROUND",),
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
            "gates": {
                "v_black": int(args.v_black),
                "s_white": int(args.s_white),
                "v_white": int(args.v_white),
                "s_low": int(args.s_low),
                "v_brown_max": int(args.v_brown_max),
                "v_bg": int(args.v_bg),
                "s_bg": int(args.s_bg),
            },
            "palette_json": str(Path(args.palette_json).name),
        }

        p = out_path_for(input_path, "vector.json", out_dir)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
