#!/usr/bin/env python3
"""
Locked-palette vectorizer for pre-processed scanner images.

Expected input:
- already masked / cropped / standardized image
- colors already reduced to a small fixed palette by an upstream process

This script does NOT run k-means. It reads exact/near-exact locked colors from a config
file, extracts regions per palette name, and writes:
- locked fill PNG (optional)
- outlines PNG (optional)
- vector JSON

Optionally, it can automatically call the spawn-point generator after the vector JSON is written.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ------------------------------
# helpers
# ------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bgr_to_xyz_dict(bgr: List[int] | Tuple[int, int, int]) -> dict:
    return {"x": int(bgr[0]), "y": int(bgr[1]), "z": int(bgr[2])}


def bgr_to_hex(bgr: List[int] | Tuple[int, int, int]) -> str:
    b, g, r = [int(x) for x in bgr]
    return f"#{r:02X}{g:02X}{b:02X}"


def out_path_for(input_path: Path, suffix: str, out_dir: Path | None = None) -> Path:
    base = out_dir if out_dir is not None else input_path.parent
    base.mkdir(parents=True, exist_ok=True)
    return base / f"output_{input_path.stem}_{suffix}"


def resize_max_dim(bgr: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr, 1.0
    scale = max_dim / float(max(h, w))
    resized = cv2.resize(
        bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


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


def contour_to_points_obj(contour: np.ndarray) -> list[dict]:
    pts = contour.reshape(-1, 2)
    return [{"x": int(p[0]), "y": int(p[1])} for p in pts]


def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(contour)
    if abs(m["m00"]) > 1e-8:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    pts = contour.reshape(-1, 2).astype(np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def normalize_name(name: str) -> str:
    return str(name).strip().upper()


# ------------------------------
# locked palette matching
# ------------------------------

def palette_from_config(cfg: dict) -> tuple[list[str], np.ndarray]:
    palette_dict = cfg.get("palette", {})
    if not isinstance(palette_dict, dict) or not palette_dict:
        raise ValueError("Config must contain a non-empty 'palette' object.")

    names: list[str] = []
    bgr_values: list[list[int]] = []
    for name, bgr in palette_dict.items():
        if not isinstance(bgr, list) or len(bgr) != 3:
            raise ValueError(f"Palette color '{name}' must be [b, g, r].")
        names.append(normalize_name(name))
        bgr_values.append([int(bgr[0]), int(bgr[1]), int(bgr[2])])
    return names, np.array(bgr_values, dtype=np.uint8)


def assign_pixels_to_locked_palette(
    image_bgr: np.ndarray,
    palette_names: list[str],
    palette_bgr: np.ndarray,
    tolerance: int,
    background_names: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - label_map: int32 image of palette index, or -1 for background/unmatched
    - locked_image: BGR image where matched pixels are replaced by exact palette colors,
      unmatched are black.
    """
    h, w = image_bgr.shape[:2]
    pixels = image_bgr.reshape((-1, 3)).astype(np.int16)
    palette = palette_bgr.astype(np.int16)

    diffs = pixels[:, None, :] - palette[None, :, :]
    dist_sq = np.sum(diffs * diffs, axis=2)
    nearest = np.argmin(dist_sq, axis=1).astype(np.int32)
    nearest_dist = np.sqrt(np.min(dist_sq, axis=1))

    label_map = nearest.reshape((h, w))
    matched = nearest_dist.reshape((h, w)) <= float(tolerance)

    # force unmatched to background label -1
    label_map = np.where(matched, label_map, -1).astype(np.int32)

    # also skip explicitly named background entries if they exist in the palette
    bg_indices = {i for i, n in enumerate(palette_names) if n in background_names}
    if bg_indices:
        for idx in bg_indices:
            label_map[label_map == idx] = -1

    locked = np.zeros_like(image_bgr)
    for idx, bgr in enumerate(palette_bgr):
        locked[label_map == idx] = bgr

    return label_map, locked


# ------------------------------
# vector extraction
# ------------------------------

def build_regions_from_label_map(
    label_map: np.ndarray,
    palette_names: list[str],
    palette_bgr: np.ndarray,
    close_radius: int,
    open_radius: int,
    min_area_px: int,
    epsilon_px: float,
    include_holes: bool,
) -> list[dict]:
    regions: list[dict] = []
    region_id = 0

    for palette_id, palette_name in enumerate(palette_names):
        raw_mask = (label_map == palette_id).astype(np.uint8)
        if int(raw_mask.sum()) == 0:
            continue

        mask = clean_mask(raw_mask, close_radius=close_radius, open_radius=open_radius)
        if int(mask.sum()) < min_area_px:
            continue

        mode = cv2.RETR_CCOMP if include_holes else cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(mask * 255, mode, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        hierarchy = hierarchy[0] if hierarchy is not None else None

        for idx, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area < float(min_area_px):
                continue

            if epsilon_px > 0:
                contour = cv2.approxPolyDP(contour, epsilon_px, True)

            x, y, w, h = cv2.boundingRect(contour)
            centroid_x, centroid_y = contour_centroid(contour)
            is_hole = False if hierarchy is None else int(hierarchy[idx][3]) != -1
            canonical_bgr = [int(v) for v in palette_bgr[palette_id].tolist()]

            regions.append(
                {
                    "region_id": int(region_id),
                    "cluster_id": int(palette_id),
                    "palette_id": int(palette_id),
                    "palette_name": palette_name,
                    "is_hole": bool(is_hole),
                    "color_bgr": bgr_to_xyz_dict(canonical_bgr),
                    "color_bgr_raw": canonical_bgr,
                    "color_hex": bgr_to_hex(canonical_bgr),
                    "area_px": float(area),
                    "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "centroid": {"x": float(round(centroid_x, 3)), "y": float(round(centroid_y, 3))},
                    "closed": True,
                    "points": contour_to_points_obj(contour),
                }
            )
            region_id += 1

    return regions


# ------------------------------
# output
# ------------------------------

def build_vector_json(image_bgr: np.ndarray, cfg: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    palette_names, palette_bgr = palette_from_config(cfg)
    background_names = {normalize_name(x) for x in cfg.get("background_names", [])}
    vcfg = cfg.get("vectorizer", {})

    max_dim = int(vcfg.get("max_dim", 1600))
    tolerance = int(vcfg.get("color_tolerance", 6))
    close_radius = int(vcfg.get("close_radius", 3))
    open_radius = int(vcfg.get("open_radius", 1))
    min_area_px = int(vcfg.get("min_area_px", 300))
    epsilon_px = float(vcfg.get("epsilon_px", 3.5))
    include_holes = bool(vcfg.get("include_holes", True))

    processed, scale = resize_max_dim(image_bgr, max_dim)
    label_map, locked_bgr = assign_pixels_to_locked_palette(
        processed,
        palette_names,
        palette_bgr,
        tolerance,
        background_names,
    )

    regions = build_regions_from_label_map(
        label_map,
        palette_names,
        palette_bgr,
        close_radius,
        open_radius,
        min_area_px,
        epsilon_px,
        include_holes,
    )

    palette_entries = []
    for idx, (name, bgr) in enumerate(zip(palette_names, palette_bgr.tolist())):
        palette_entries.append(
            {
                "entry_index": int(idx),
                "name": name,
                "bgr": bgr_to_xyz_dict(bgr),
                "bgr_raw": [int(x) for x in bgr],
                "hex": bgr_to_hex(bgr),
            }
        )

    outlines = np.zeros_like(locked_bgr)
    for region in regions:
        contour = np.array([[p["x"], p["y"]] for p in region["points"]], dtype=np.int32).reshape(-1, 1, 2)
        color = tuple(int(v) for v in region["color_bgr_raw"])
        cv2.polylines(outlines, [contour], isClosed=True, color=color, thickness=1)

    data = {
        "generator": "locked_palette_vectorizer.py",
        "version": 1,
        "source": {
            "mode": "locked_palette_exact_or_near_exact",
            "resize_scale": float(scale),
            "color_tolerance": tolerance,
        },
        "image": {"width": int(locked_bgr.shape[1]), "height": int(locked_bgr.shape[0])},
        "palette_entries": palette_entries,
        "regions": regions,
    }
    return data, locked_bgr, outlines


# ------------------------------
# main
# ------------------------------

def maybe_run_spawn_generation(
    vector_json_path: Path,
    config_path: Path,
    cfg: dict,
    out_dir: Path | None,
) -> None:
    vcfg = cfg.get("vectorizer", {})
    if not bool(vcfg.get("auto_run_spawn_generation", False)):
        return

    spawn_script_value = str(vcfg.get("spawn_script", "generate_spawn_points_from_vector_json_v2.py"))
    spawn_script = Path(spawn_script_value)
    if not spawn_script.is_absolute():
        spawn_script = config_path.parent / spawn_script
        if not spawn_script.exists():
            spawn_script = vector_json_path.parent / Path(spawn_script_value)

    if not spawn_script.exists():
        raise FileNotFoundError(
            f"Auto-run spawn generation is enabled, but spawn script was not found: {spawn_script}"
        )

    cmd = [
        sys.executable,
        str(spawn_script),
        "--input",
        str(vector_json_path),
        "--config",
        str(config_path),
    ]
    if out_dir is not None:
        cmd.extend(["--outdir", str(out_dir)])

    print("Running spawn generation:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Vectorize a locked-palette scanner image.")
    ap.add_argument("--input", required=True, help="Path to preprocessed locked-palette image.")
    ap.add_argument("--config", required=True, help="Path to shared pipeline config JSON.")
    ap.add_argument("--outdir", default="", help="Optional output directory.")
    ap.add_argument("--skip-spawn", action="store_true", help="Do not auto-run spawn generation.")
    args = ap.parse_args()

    input_path = Path(args.input)
    config_path = Path(args.config)
    out_dir = Path(args.outdir) if args.outdir else None

    cfg = load_json(config_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    vector_data, locked_fill, outlines = build_vector_json(image, cfg)
    vcfg = cfg.get("vectorizer", {})

    if bool(vcfg.get("write_fill_png", True)):
        fill_path = out_path_for(input_path, "locked_fill.png", out_dir)
        cv2.imwrite(str(fill_path), locked_fill)
        print(f"Wrote: {fill_path}")

    if bool(vcfg.get("write_outlines_png", True)):
        outlines_path = out_path_for(input_path, "vector_outlines.png", out_dir)
        cv2.imwrite(str(outlines_path), outlines)
        print(f"Wrote: {outlines_path}")

    vector_json_path = out_path_for(input_path, "vector.json", out_dir)
    with open(vector_json_path, "w", encoding="utf-8") as f:
        json.dump(vector_data, f, indent=2)
    print(f"Wrote: {vector_json_path}")
    print(f"Vector regions: {len(vector_data['regions'])}")

    if not args.skip_spawn:
        maybe_run_spawn_generation(vector_json_path, config_path, cfg, out_dir)


if __name__ == "__main__":
    main()
