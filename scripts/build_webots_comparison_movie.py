from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "revision_outputs" / "webots_movie_selection.json"
DEFAULT_OUTPUT = ROOT / "revision_outputs" / "webots_movie_side_by_side_tradeoff.mp4"


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 25.0))
    frames = [frame for frame in reader]
    reader.close()
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    return frames, fps


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(frame).resize(size, Image.Resampling.LANCZOS)


def get_trial(selection: dict, label: str) -> dict:
    for trial in selection["trials"]:
        trial_label = trial.get("label") or trial.get("trial", {}).get("label")
        if trial_label == label:
            return trial
    raise KeyError(label)


def portable_path(value: object) -> Path:
    return Path(str(value).replace("\\", "/"))


def row_value(trial: dict, key: str) -> str:
    return str(trial["archive_row"][key])


def build_frame(
    left: np.ndarray,
    right: np.ndarray,
    left_trial: dict,
    right_trial: dict,
    *,
    canvas_size: tuple[int, int] = (1920, 720),
) -> np.ndarray:
    canvas = Image.new("RGB", canvas_size, (18, 22, 26))
    left_img = resize_frame(left, (960, 540))
    right_img = resize_frame(right, (960, 540))
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (960, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(32)
    body_font = load_font(26)
    small_font = load_font(22)

    draw.rectangle([(0, 0), (960, 44)], fill=(0, 0, 0))
    draw.rectangle([(960, 0), (1920, 44)], fill=(0, 0, 0))
    draw.text((24, 8), "No-relay allocation", fill=(255, 255, 255), font=title_font)
    draw.text((984, 8), "Relay-rich allocation", fill=(255, 255, 255), font=title_font)

    draw.rectangle([(0, 540), (1920, 720)], fill=(18, 22, 26))
    draw.text(
        (40, 552),
        "Role colors: blue=standard, green=scout, yellow=relay, red=payload. Green band marks the payload goal zone.",
        fill=(230, 235, 240),
        font=small_font,
    )
    draw.text(
        (40, 582),
        "Same Webots stress/command setting: severe, autonomous.",
        fill=(230, 235, 240),
        font=body_font,
    )
    left_summary = (
        f"No-relay: delay={row_value(left_trial, 'mean_effective_map_delay')}, "
        f"safe={row_value(left_trial, 'safe_delivery_success')}, "
        f"attrition={float(row_value(left_trial, 'attrition_rate')):.3f}, "
        f"score={float(row_value(left_trial, 'operational_score')):.1f}"
    )
    right_summary = (
        f"Relay-rich: delay={row_value(right_trial, 'mean_effective_map_delay')}, "
        f"safe={row_value(right_trial, 'safe_delivery_success')}, "
        f"attrition={float(row_value(right_trial, 'attrition_rate')):.3f}, "
        f"score={float(row_value(right_trial, 'operational_score')):.1f}"
    )
    draw.text((40, 622), left_summary, fill=(190, 220, 255), font=small_font)
    draw.text((40, 652), right_summary, fill=(255, 220, 120), font=small_font)
    draw.text(
        (40, 684),
        "Illustrated trade-off: relay-rich reduces effective delay but does not rescue safe delivery in this sealed pair.",
        fill=(255, 255, 255),
        font=small_font,
    )
    return np.asarray(canvas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a side-by-side Webots supplementary comparison movie.")
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    left_trial = get_trial(selection, "no_relay_hetero_autonomous_severe_rep3")
    right_trial = get_trial(selection, "relay_rich_autonomous_severe_rep3")
    left_path = ROOT / portable_path(left_trial["movie_path"])
    right_path = ROOT / portable_path(right_trial["movie_path"])

    left_frames, left_fps = read_video(left_path)
    right_frames, right_fps = read_video(right_path)
    fps = min(left_fps, right_fps)
    n_frames = max(len(left_frames), len(right_frames))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.output, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for idx in range(n_frames):
            left = left_frames[min(idx, len(left_frames) - 1)]
            right = right_frames[min(idx, len(right_frames) - 1)]
            writer.append_data(build_frame(left, right, left_trial, right_trial))
    finally:
        writer.close()

    print(json.dumps({"output": str(args.output), "frames": n_frames, "fps": fps}, indent=2))


if __name__ == "__main__":
    main()
