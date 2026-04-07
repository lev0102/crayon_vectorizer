#!/usr/bin/env python3
"""
Convenience wrapper: run vectorization, then spawn generation using one config file.

This is mostly useful if you want one command for testing, while still keeping the
vectorizer and spawn generator as separate scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the locked palette vectorize -> spawn pipeline.")
    ap.add_argument("--input", required=True, help="Path to the preprocessed locked-palette image.")
    ap.add_argument("--config", required=True, help="Path to shared config JSON.")
    ap.add_argument("--outdir", default="", help="Optional output directory.")
    args = ap.parse_args()

    this_dir = Path(__file__).resolve().parent
    vectorizer = this_dir / "locked_palette_vectorizer.py"

    cmd = [
        sys.executable,
        str(vectorizer),
        "--input",
        args.input,
        "--config",
        args.config,
    ]
    if args.outdir:
        cmd.extend(["--outdir", args.outdir])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
