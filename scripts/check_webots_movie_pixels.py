"""Pixel-level QA for Webots supplementary movies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def portable_path_text(path: Path) -> str:
    return path.as_posix()


def check_movie(
    path: Path,
    *,
    y_min: int = 100,
    min_nonzero_pixels: int = 1000,
    min_mean_rgb: float = 1.0,
    min_std_rgb: float = 20.0,
) -> dict:
    frames = 0
    max_nonzero = 0
    mean_values: list[float] = []
    std_values: list[float] = []
    shape = None
    for frame in iio.imiter(path):
        arr = np.asarray(frame)
        shape = list(arr.shape)
        roi = arr[y_min:, :, :3]
        max_nonzero = max(max_nonzero, int(np.count_nonzero(roi)))
        mean_values.append(float(roi.mean()))
        std_values.append(float(roi.std()))
        frames += 1
    mean_roi = float(np.mean(mean_values)) if mean_values else 0.0
    max_std_roi = float(np.max(std_values)) if std_values else 0.0
    return {
        "path": portable_path_text(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "frames": frames,
        "shape": shape,
        "y_min": y_min,
        "max_nonzero_pixels_below_y_min": max_nonzero,
        "mean_rgb_below_y_min": mean_roi,
        "max_std_rgb_below_y_min": max_std_roi,
        "min_nonzero_pixels": min_nonzero_pixels,
        "min_mean_rgb": min_mean_rgb,
        "min_std_rgb": min_std_rgb,
        "passed": bool(
            frames > 0
            and max_nonzero >= min_nonzero_pixels
            and mean_roi >= min_mean_rgb
            and max_std_roi >= min_std_rgb
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that Webots movies are not black below the label area.")
    parser.add_argument("movies", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--y-min", type=int, default=100)
    parser.add_argument("--min-nonzero-pixels", type=int, default=1000)
    parser.add_argument("--min-mean-rgb", type=float, default=1.0)
    parser.add_argument("--min-std-rgb", type=float, default=20.0)
    args = parser.parse_args()

    checks = [
        check_movie(
            movie,
            y_min=args.y_min,
            min_nonzero_pixels=args.min_nonzero_pixels,
            min_mean_rgb=args.min_mean_rgb,
            min_std_rgb=args.min_std_rgb,
        )
        for movie in args.movies
    ]
    result = {
        "protocol": "webots_movie_pixel_qa",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
