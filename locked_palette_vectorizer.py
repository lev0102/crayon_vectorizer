#!/usr/bin/env python3
"""
locked_palette_vectorizer.py

Drop-in locked-palette vectorizer for the newer pipeline.

Purpose
- Reads an already masked / standardized scanner image.
- Snaps pixels to a fixed locked palette with nearest-color + tolerance.
- Ignores configured palette names such as WHITE.
- Extracts regions/contours for each palette color.
- Writes:
    - optional locked fill PNG
    - optional outlines PNG
    - vector JSON
- Can optionally auto-call the spawn-point generator after vector JSON is written.

Recommended pipeline
scanner/preprocess -> locked_palette_vectorizer.py -> generate_spawn_points_from_vector_json.py -> Unreal

Example
python locked_palette_vectorizer.py --input input/Drawing0.png --config locked_palette_vectorizer_config.json --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Utilities
# ============================================================

def bgr_to_hex(bgr: list[int] | tuple[int, int, int] | np.ndarray) -> str:
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def bgr_to_xyz_dict(bgr: list[int] | tuple[int, int, int] | np.ndarray) -> dict[str, int]:
    # Unreal-friendly Vector style: x=b, y=g, z=r
    return {"x": int(bgr[0]), "y": int(bgr[1]), "z": int(bgr[2])}


def out_path_for(input_path: Path, suffix: str, out_dir: Path | None = None) -> Path:
    base = out_dir if out_dir is not None else input_path.parent
    base.mkdir(parents=True, exist_ok=True)
    return base / f"output_{input_path.stem}_{suffix}"


def clean_mask(mask01: np.ndarray, close_radius: int = 3, open_radius: int = 1) -> np.ndarray:
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


def approx_contour(cnt: np.ndarray, epsilon_px: float) -> np.ndarray:
    if epsilon_px <= 0:
        return cnt
    return cv2.approxPolyDP(cnt, epsilon_px, True)


def signed_area_of_polygon(points_xy: list[dict[str, int]]) -> float:
    if len(points_xy) < 3:
        return 0.0
    pts = np.array([[p["x"], p["y"]] for p in points_xy], dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


# ============================================================
# Config
# ============================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "locked_palette": {
        # BGR order, not RGB
        "RED":    [74, 68, 217],
        "ORANGE": [47, 110, 255],
        "YELLOW": [65, 190, 241],
        "GREEN":  [119, 181, 64],
        "BLUE":   [212, 108, 58],
        "BROWN":  [77, 107, 204],
        "BLACK":  [82, 86, 88],
        "WHITE":  [255, 255, 255],
    },
    "palette_matching": {
        "mode": "nearest_with_tolerance",   # exact | nearest | nearest_with_tolerance
        "tolerance": 18.0,
        "ignore_palette_names": ["WHITE"],
        "unknown_behavior": "ignore",       # ignore | nearest_any
    },
    "vectorizer": {
        "close_radius": 3,
        "open_radius": 1,
        "min_area_px": 300,
        "epsilon_px": 3.5,
        "include_holes": True,
        "write_fill_png": True,
        "write_outlines_png": True,
        "write_vector_json": True,
    },
    "automation": {
        "auto_run_spawn_generator": False,
        "spawn_generator_script": "generate_spawn_points_from_vector_json.py",
        "spawn_generator_extra_args": []
    }
}


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None) -> dict[str, Any]:
    cfg = DEFAULT_CONFIG
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)
    return deep_merge(cfg, user_cfg)


def palette_from_config(cfg: dict[str, Any]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    palette_dict = cfg["locked_palette"]
    names: list[str] = []
    bgrs: list[list[int]] = []
    entries: list[dict[str, Any]] = []

    for i, (name, bgr) in enumerate(palette_dict.items()):
        bgr_list = [int(bgr[0]), int(bgr[1]), int(bgr[2])]
        names.append(str(name))
        bgrs.append(bgr_list)
        entries.append({
            "entry_index": int(i),
            "name": str(name),
            "bgr": bgr_to_xyz_dict(bgr_list),
            "bgr_raw": bgr_list,
            "hex": bgr_to_hex(bgr_list),
        })

    return np.array(bgrs, dtype=np.uint8), names, entries


# ============================================================
# Palette matching
# ============================================================

def classify_pixels_to_locked_palette(
    bgr: np.ndarray,
    palette_bgr: np.ndarray,
    palette_names: list[str],
    match_mode: str,
    tolerance: float,
    ignore_palette_names: set[str],
    unknown_behavior: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        locked_bgr: image snapped to canonical palette colors (ignored/unknown become white if ignored)
        palette_index_map: HxW int32, -1 for ignored pixels
    """
    h, w = bgr.shape[:2]
    pixels = bgr.reshape((-1, 3)).astype(np.float32)
    pal = palette_bgr.astype(np.float32)

    diff = pixels[:, None, :] - pal[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest_idx = np.argmin(dist2, axis=1)
    nearest_dist = np.sqrt(np.min(dist2, axis=1).astype(np.float32))

    exact_mask = (nearest_dist == 0.0)

    if match_mode == "exact":
        accept_mask = exact_mask
    elif match_mode == "nearest":
        accept_mask = np.ones_like(nearest_dist, dtype=bool)
    elif match_mode == "nearest_with_tolerance":
        accept_mask = nearest_dist <= float(tolerance)
    else:
        raise ValueError(f"Unsupported palette matching mode: {match_mode}")

    palette_index_map = nearest_idx.astype(np.int32)

    if unknown_behavior == "ignore":
        palette_index_map[~accept_mask] = -1
    elif unknown_behavior == "nearest_any":
        pass
    else:
        raise ValueError(f"Unsupported unknown_behavior: {unknown_behavior}")

    # Ignore configured palette names after classification
    for i, name in enumerate(palette_names):
        if name in ignore_palette_names:
            palette_index_map[palette_index_map == i] = -1

    locked_flat = np.full((pixels.shape[0], 3), 255, dtype=np.uint8)  # default white background for ignored pixels
    valid = palette_index_map >= 0
    if np.any(valid):
        locked_flat[valid] = palette_bgr[palette_index_map[valid]]

    locked_bgr = locked_flat.reshape((h, w, 3))
    palette_index_map = palette_index_map.reshape((h, w))
    return locked_bgr, palette_index_map


# ============================================================
# Region extraction
# ============================================================

def contours_to_regions_for_palette(
    palette_index_map: np.ndarray,
    palette_bgr: np.ndarray,
    palette_names: list[str],
    close_radius: int,
    open_radius: int,
    min_area_px: int,
    epsilon_px: float,
    include_holes: bool,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    region_id = 0

    for palette_id, palette_name in enumerate(palette_names):
        mask01 = (palette_index_map == palette_id).astype(np.uint8)
        if np.count_nonzero(mask01) == 0:
            continue

        clean = clean_mask(mask01, close_radius=close_radius, open_radius=open_radius)
        if np.count_nonzero(clean) == 0:
            continue

        mode = cv2.RETR_CCOMP if include_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(clean, mode, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        hierarchy = hierarchy[0] if hierarchy is not None else None

        for i, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))
            if area < float(min_area_px):
                continue

            approx = approx_contour(cnt, epsilon_px)
            if approx.shape[0] < 3:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            pts = [{"x": int(p[0][0]), "y": int(p[0][1])} for p in approx]

            # RETR_CCOMP: parent < 0 means outer contour, else hole
            is_hole = False
            if hierarchy is not None:
                parent = int(hierarchy[i][3])
                is_hole = parent >= 0

            if is_hole and not include_holes:
                continue

            color_raw = [int(v) for v in palette_bgr[palette_id]]

            regions.append({
                "region_id": int(region_id),
                "palette_id": int(palette_id),
                "palette_name": str(palette_name),
                "is_hole": bool(is_hole),
                "color_bgr": bgr_to_xyz_dict(color_raw),
                "color_bgr_raw": color_raw,
                "color_hex": bgr_to_hex(color_raw),
                "area_px": float(area),
                "bbox": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                },
                "closed": True,
                "points": pts,
                "signed_area": float(signed_area_of_polygon(pts)),
            })
            region_id += 1

    return regions


def build_outlines_image(shape_hw: tuple[int, int], regions: list[dict[str, Any]]) -> np.ndarray:
    h, w = shape_hw
    outlines = np.full((h, w, 3), 255, dtype=np.uint8)
    for region in regions:
        contour = np.array([[p["x"], p["y"]] for p in region["points"]], dtype=np.int32).reshape(-1, 1, 2)
        color = tuple(int(v) for v in region["color_bgr_raw"])
        cv2.polylines(outlines, [contour], isClosed=True, color=color, thickness=1)
    return outlines


def build_vector_data(
    input_path: Path,
    locked_bgr: np.ndarray,
    palette_entries: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    pm = cfg["palette_matching"]

    data = {
        "generator": "locked_palette_vectorizer.py",
        "version": 2,
        "source": {
            "mode": "locked_palette_exact_or_near_exact",
            "input_file": str(input_path.name),
            "palette_matching_mode": str(pm["mode"]),
            "color_tolerance": float(pm["tolerance"]),
            "ignored_palette_names": list(pm.get("ignore_palette_names", [])),
            "unknown_behavior": str(pm.get("unknown_behavior", "ignore")),
        },
        "image": {
            "width": int(locked_bgr.shape[1]),
            "height": int(locked_bgr.shape[0]),
            "pixel_count": int(locked_bgr.shape[0] * locked_bgr.shape[1]),
            "coordinate_space": "pixel",
            "origin": "top_left",
            "y_axis_direction": "down",
            "file_name": str(input_path.name),
        },
        "palette_entries": palette_entries,
        "regions": regions,
    }
    return data


# ============================================================
# Main process
# ============================================================

def vectorize_locked_palette_image(
    input_path: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    palette_bgr, palette_names, palette_entries = palette_from_config(cfg)
    pm = cfg["palette_matching"]
    vec_cfg = cfg["vectorizer"]

    locked_bgr, palette_index_map = classify_pixels_to_locked_palette(
        bgr=bgr,
        palette_bgr=palette_bgr,
        palette_names=palette_names,
        match_mode=str(pm["mode"]),
        tolerance=float(pm["tolerance"]),
        ignore_palette_names=set(pm.get("ignore_palette_names", [])),
        unknown_behavior=str(pm.get("unknown_behavior", "ignore")),
    )

    regions = contours_to_regions_for_palette(
        palette_index_map=palette_index_map,
        palette_bgr=palette_bgr,
        palette_names=palette_names,
        close_radius=int(vec_cfg["close_radius"]),
        open_radius=int(vec_cfg["open_radius"]),
        min_area_px=int(vec_cfg["min_area_px"]),
        epsilon_px=float(vec_cfg["epsilon_px"]),
        include_holes=bool(vec_cfg["include_holes"]),
    )

    outlines = build_outlines_image((locked_bgr.shape[0], locked_bgr.shape[1]), regions)
    data = build_vector_data(
        input_path=input_path,
        locked_bgr=locked_bgr,
        palette_entries=palette_entries,
        regions=regions,
        cfg=cfg,
    )
    return data, locked_bgr, outlines


def maybe_run_spawn_generator(
    vector_json_path: Path,
    config_path: Path | None,
    cfg: dict[str, Any],
) -> None:
    auto_cfg = cfg.get("automation", {})
    if not bool(auto_cfg.get("auto_run_spawn_generator", False)):
        return

    script_name = str(auto_cfg.get("spawn_generator_script", "generate_spawn_points_from_vector_json.py"))
    extra_args = [str(x) for x in auto_cfg.get("spawn_generator_extra_args", [])]

    script_path = Path(script_name)
    if not script_path.is_absolute():
        script_path = Path(__file__).resolve().parent / script_path

    cmd = [sys.executable, str(script_path), "--input", str(vector_json_path)]
    if config_path is not None:
        # Optional: your spawn generator can choose to ignore this if not supported yet
        cmd.extend(["--config", str(config_path)])
    cmd.extend(extra_args)

    print("Auto-running spawn generator:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Locked-palette vectorizer for 7-color scanner pipeline.")
    ap.add_argument("--input", required=True, help="Path to locked-palette image.")
    ap.add_argument("--config", default="", help="Optional JSON config file.")
    ap.add_argument("--outdir", default="", help="Optional output directory. Default: next to input image.")
    ap.add_argument("--fill", action="store_true", help="Write locked fill PNG.")
    ap.add_argument("--outlines", action="store_true", help="Write outlines PNG.")
    ap.add_argument("--json", action="store_true", help="Write vector JSON.")
    ap.add_argument("--all", action="store_true", help="Write fill + outlines + json.")
    args = ap.parse_args()

    input_path = Path(args.input)
    config_path = Path(args.config) if args.config else None
    cfg = load_config(config_path)
    out_dir = Path(args.outdir) if args.outdir else None

    do_fill = bool(args.all or args.fill or cfg["vectorizer"].get("write_fill_png", False))
    do_outlines = bool(args.all or args.outlines or cfg["vectorizer"].get("write_outlines_png", False))
    do_json = bool(args.all or args.json or cfg["vectorizer"].get("write_vector_json", True))

    data, locked_bgr, outlines = vectorize_locked_palette_image(input_path, cfg)

    vector_json_path: Path | None = None

    if do_fill:
        p = out_path_for(input_path, "fill.png", out_dir)
        cv2.imwrite(str(p), locked_bgr)
        print(f"Wrote: {p}")

    if do_outlines:
        p = out_path_for(input_path, "outlines.png", out_dir)
        cv2.imwrite(str(p), outlines)
        print(f"Wrote: {p}")

    if do_json:
        p = out_path_for(input_path, "vector.json", out_dir)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        vector_json_path = p
        print(f"Wrote: {p}")

    if vector_json_path is not None:
        maybe_run_spawn_generator(vector_json_path, config_path, cfg)


if __name__ == "__main__":
    main()
