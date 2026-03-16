#!/usr/bin/env python3
"""
Second-stage generator for Unreal scatter spawning.

Reads the vector JSON exported by fixed_palatte_vectorizer_v4.py and generates
precomputed spawn points so Unreal does not need to solve region fill again.

Generation modes:
- scatter : random points inside polygon, with simple minimum-distance filtering
- grid    : jittered grid points inside polygon
- spline  : points sampled along contour/polyline

Mode assignment is configured near the top of this script per palette color/name,
so you do not need to pass mode selection via CLI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ============================================================
# USER CONFIG
# ============================================================

DEFAULT_METHOD = "scatter"

METHOD_BY_COLOR = {
    # Example defaults; edit freely
    "RED": "scatter",
    "ORANGE": "scatter",
    "YELLOW": "scatter",
    "GREEN": "scatter",
    "BLUE": "grid",
    "BROWN": "grid",
    "SKIN": "grid",
    "GREY": "spline",
    "GRAY": "spline",
    "BLACK": "spline",
    "WHITE": "spline",
}

# Global fallback settings by method
METHOD_DEFAULTS = {
    "scatter": {
        "density_scale": 1.0,            # multiplier on estimated point count
        "target_spacing_px": 22.0,       # rough distance / density control
        "min_distance_px": 12.0,         # simple anti-clump distance
        "max_points": 5000,
        "max_attempt_factor": 20,        # attempts ~= target_count * factor
    },
    "grid": {
        "density_scale": 1.0,
        "grid_spacing_px": 24.0,
        "jitter_ratio": 0.35,            # fraction of spacing
        "max_points": 5000,
    },
    "spline": {
        "density_scale": 1.0,
        "step_px": 16.0,
        "closed_loop": True,
        "max_points": 5000,
    },
}

# Optional per-color overrides.
# You only need to list colors you want to customize.
COLOR_SETTINGS = {
    # "GREEN": {"density_scale": 1.25, "target_spacing_px": 18.0, "min_distance_px": 10.0},
    # "BLACK": {"step_px": 10.0},
}

# Hole handling for fill-based methods
SKIP_HOLE_REGIONS = True

# Deterministic base seed
BASE_SEED = 1337

# ============================================================
# Utils
# ============================================================


def out_path_for(input_path: Path, suffix: str, out_dir: Path | None = None) -> Path:
    base = out_dir if out_dir is not None else input_path.parent
    base.mkdir(parents=True, exist_ok=True)
    return base / f"output_{input_path.stem}_{suffix}"


def normalize_color_name(name: str) -> str:
    return str(name).strip().upper()


def merge_settings(color_name: str, method: str) -> Dict:
    settings = dict(METHOD_DEFAULTS[method])
    settings.update(COLOR_SETTINGS.get(normalize_color_name(color_name), {}))
    return settings


def polygon_to_np(points: List[Dict]) -> np.ndarray:
    arr = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
    if arr.ndim != 2 or arr.shape[0] < 3:
        raise ValueError("Region must contain at least 3 contour points.")
    return arr


def bbox_from_points(poly: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(poly.reshape(-1, 1, 2))
    return int(x), int(y), int(w), int(h)


def polygon_centroid(poly: np.ndarray) -> Tuple[float, float]:
    m = cv2.moments(poly.reshape(-1, 1, 2))
    if abs(m["m00"]) > 1e-8:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    pts = poly.astype(np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def build_region_mask(poly: np.ndarray, holes: List[np.ndarray], bbox_pad: int = 1):
    x, y, w, h = bbox_from_points(poly)
    x0 = max(0, x - bbox_pad)
    y0 = max(0, y - bbox_pad)
    w2 = w + bbox_pad * 2
    h2 = h + bbox_pad * 2

    mask = np.zeros((h2, w2), dtype=np.uint8)
    outer_local = poly - np.array([[x0, y0]], dtype=np.int32)
    cv2.fillPoly(mask, [outer_local.reshape(-1, 1, 2)], 255)

    if holes:
        hole_locals = [(hpoly - np.array([[x0, y0]], dtype=np.int32)).reshape(-1, 1, 2) for hpoly in holes]
        cv2.fillPoly(mask, hole_locals, 0)

    return mask, x0, y0


def make_rng(region_id: int, color_name: str) -> random.Random:
    seed = BASE_SEED + int(region_id) * 1000003 + sum(ord(c) for c in color_name)
    return random.Random(seed)


def estimate_target_count(area_px: float, spacing_px: float, density_scale: float, max_points: int) -> int:
    spacing_px = max(1.0, float(spacing_px))
    approx = area_px / (spacing_px * spacing_px)
    count = int(round(approx * float(density_scale)))
    return max(0, min(int(max_points), count))


def distance_sq(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def contour_arc_lengths(points: np.ndarray, closed_loop: bool = True):
    pts = points.astype(np.float32)
    seg_lengths = []
    total = 0.0
    n = len(pts)
    count = n if closed_loop else max(0, n - 1)
    for i in range(count):
        j = (i + 1) % n
        if not closed_loop and i == n - 1:
            break
        d = float(np.linalg.norm(pts[j] - pts[i]))
        seg_lengths.append(d)
        total += d
    return seg_lengths, total


def point_along_contour(points: np.ndarray, distance: float, closed_loop: bool = True):
    pts = points.astype(np.float32)
    seg_lengths, total = contour_arc_lengths(pts, closed_loop=closed_loop)
    if total <= 1e-8:
        return float(pts[0, 0]), float(pts[0, 1])

    d = distance % total if closed_loop else max(0.0, min(distance, total))
    n = len(pts)
    count = n if closed_loop else n - 1
    acc = 0.0
    for i in range(count):
        seg = seg_lengths[i]
        j = (i + 1) % n
        if acc + seg >= d or i == count - 1:
            t = 0.0 if seg <= 1e-8 else (d - acc) / seg
            p = pts[i] * (1.0 - t) + pts[j] * t
            return float(p[0]), float(p[1])
        acc += seg
    return float(pts[-1, 0]), float(pts[-1, 1])


# ============================================================
# Point generation methods
# ============================================================


def generate_scatter_points(poly: np.ndarray, holes: List[np.ndarray], area_px: float, settings: Dict, rng: random.Random):
    spacing = float(settings.get("target_spacing_px", 22.0))
    min_distance = float(settings.get("min_distance_px", 12.0))
    density_scale = float(settings.get("density_scale", 1.0))
    max_points = int(settings.get("max_points", 5000))
    max_attempt_factor = int(settings.get("max_attempt_factor", 20))

    target_count = estimate_target_count(area_px, spacing, density_scale, max_points)
    if target_count <= 0:
        return []

    mask, x0, y0 = build_region_mask(poly, holes)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    min_dist_sq = min_distance * min_distance
    points: List[Tuple[float, float]] = []
    attempts = 0
    max_attempts = max(target_count * max_attempt_factor, target_count)

    while len(points) < target_count and attempts < max_attempts:
        idx = rng.randrange(len(xs))
        x = float(xs[idx] + x0 + rng.random())
        y = float(ys[idx] + y0 + rng.random())

        keep = True
        for p in points:
            if distance_sq((x, y), p) < min_dist_sq:
                keep = False
                break
        if keep:
            points.append((x, y))
        attempts += 1

    return points


def generate_grid_points(poly: np.ndarray, holes: List[np.ndarray], settings: Dict, rng: random.Random):
    spacing = float(settings.get("grid_spacing_px", 24.0))
    jitter_ratio = float(settings.get("jitter_ratio", 0.35))
    max_points = int(settings.get("max_points", 5000))

    mask, x0, y0 = build_region_mask(poly, holes)
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return []

    jitter = spacing * jitter_ratio
    points: List[Tuple[float, float]] = []

    y = spacing * 0.5
    while y < h and len(points) < max_points:
        x = spacing * 0.5
        while x < w and len(points) < max_points:
            sx = x + rng.uniform(-jitter, jitter)
            sy = y + rng.uniform(-jitter, jitter)
            ix = int(round(sx))
            iy = int(round(sy))
            if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
                points.append((float(ix + x0), float(iy + y0)))
            x += spacing
        y += spacing

    return points


def generate_spline_points(poly: np.ndarray, settings: Dict, rng: random.Random):
    step_px = float(settings.get("step_px", 16.0))
    closed_loop = bool(settings.get("closed_loop", True))
    max_points = int(settings.get("max_points", 5000))
    density_scale = float(settings.get("density_scale", 1.0))

    seg_lengths, total_len = contour_arc_lengths(poly, closed_loop=closed_loop)
    if total_len <= 1e-8:
        return []

    step_px = max(1.0, step_px / max(1e-8, density_scale))
    count = max(1, min(max_points, int(math.floor(total_len / step_px))))

    phase = rng.random() * step_px
    points = [point_along_contour(poly, phase + i * step_px, closed_loop=closed_loop) for i in range(count)]
    return points


# ============================================================
# Region organization
# ============================================================


def assign_holes_to_outers(regions: List[Dict]) -> Dict[int, List[np.ndarray]]:
    outers = []
    holes = []

    for r in regions:
        poly = polygon_to_np(r["points"])
        entry = {"region": r, "poly": poly}
        if bool(r.get("is_hole", False)):
            holes.append(entry)
        else:
            outers.append(entry)

    hole_map: Dict[int, List[np.ndarray]] = {int(o["region"]["region_id"]): [] for o in outers}

    for hole in holes:
        test_pt = tuple(map(float, hole["poly"][0]))
        best_outer_id = None
        best_area = None
        for outer in outers:
            inside = cv2.pointPolygonTest(outer["poly"].reshape(-1, 1, 2), test_pt, False)
            if inside >= 0:
                area = abs(float(cv2.contourArea(outer["poly"].reshape(-1, 1, 2))))
                if best_area is None or area < best_area:
                    best_area = area
                    best_outer_id = int(outer["region"]["region_id"])
        if best_outer_id is not None:
            hole_map.setdefault(best_outer_id, []).append(hole["poly"])

    return hole_map


# ============================================================
# Export
# ============================================================


def make_spawn_point_dict(x: float, y: float, region: Dict, method: str, idx: int, rng: random.Random):
    return {
        "spawn_id": int(idx),
        "region_id": int(region["region_id"]),
        "palette_name": str(region["palette_name"]),
        "method": str(method),
        "x": float(round(x, 3)),
        "y": float(round(y, 3)),
        "rand_keep": float(round(rng.random(), 6)),
        "seed": int(rng.randrange(0, 2**31 - 1)),
    }


def generate_spawn_data(data: Dict):
    regions = data.get("regions", [])
    image = data.get("image", {})
    hole_map = assign_holes_to_outers(regions)

    spawn_points: List[Dict] = []
    region_summaries: List[Dict] = []
    spawn_id = 0

    for region in regions:
        if bool(region.get("is_hole", False)) and SKIP_HOLE_REGIONS:
            continue

        color_name = normalize_color_name(region.get("palette_name", ""))
        method = METHOD_BY_COLOR.get(color_name, DEFAULT_METHOD)
        if method not in METHOD_DEFAULTS:
            raise ValueError(f"Unsupported method '{method}' for color '{color_name}'.")

        poly = polygon_to_np(region["points"])
        holes = hole_map.get(int(region["region_id"]), [])
        settings = merge_settings(color_name, method)
        rng = make_rng(int(region["region_id"]), color_name)

        area_px = float(region.get("area_px", abs(cv2.contourArea(poly.reshape(-1, 1, 2)))))
        centroid_x, centroid_y = polygon_centroid(poly)

        if method == "scatter":
            pts = generate_scatter_points(poly, holes, area_px, settings, rng)
        elif method == "grid":
            pts = generate_grid_points(poly, holes, settings, rng)
        elif method == "spline":
            pts = generate_spline_points(poly, settings, rng)
        else:
            pts = []

        for x, y in pts:
            spawn_points.append(make_spawn_point_dict(x, y, region, method, spawn_id, rng))
            spawn_id += 1

        region_summaries.append({
            "region_id": int(region["region_id"]),
            "palette_name": str(region.get("palette_name", "")),
            "method": str(method),
            "is_hole": bool(region.get("is_hole", False)),
            "area_px": float(round(area_px, 3)),
            "centroid": {"x": float(round(centroid_x, 3)), "y": float(round(centroid_y, 3))},
            "bbox": region.get("bbox", {}),
            "generated_point_count": int(len(pts)),
        })

    out = {
        "generator": "generate_spawn_points_from_vector_json.py",
        "version": 1,
        "image": {
            "width": int(image.get("width", 0)),
            "height": int(image.get("height", 0)),
        },
        "settings": {
            "default_method": DEFAULT_METHOD,
            "method_by_color": METHOD_BY_COLOR,
            "method_defaults": METHOD_DEFAULTS,
            "color_settings": COLOR_SETTINGS,
            "skip_hole_regions": SKIP_HOLE_REGIONS,
            "base_seed": BASE_SEED,
        },
        "region_summaries": region_summaries,
        "spawn_points": spawn_points,
    }
    return out


# ============================================================
# Main
# ============================================================


def main():
    ap = argparse.ArgumentParser(description="Generate Unreal-friendly spawn points from vector JSON.")
    ap.add_argument("--input", required=True, help="Path to output_*_vector.json from the vectorizer.")
    ap.add_argument("--outdir", default="", help="Optional output directory. Default: next to input JSON.")
    ap.add_argument("--suffix", default="spawn_points.json", help="Output suffix. Default: spawn_points.json")
    args = ap.parse_args()

    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = generate_spawn_data(data)

    out_dir = Path(args.outdir) if args.outdir else None
    out_path = out_path_for(input_path, args.suffix, out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Total spawn points: {len(out['spawn_points'])}")


if __name__ == "__main__":
    main()
