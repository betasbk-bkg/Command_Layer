"""Record the sealed Webots replicate-3 movie trials and verify their metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revision_outputs"
THEORY_OUT = ROOT / "theory_outputs"
WORLD = ROOT / "webots_validation" / "worlds" / "swarm_validation.wbt"
DEFAULT_WEBOTS = Path(os.environ.get("WEBOTS_EXECUTABLE", "webotsw.exe"))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_webots_movie_pixels import check_movie  # noqa: E402

TRIALS = [
    {
        "label": "no_relay_hetero_autonomous_severe_rep3",
        "team": "no_relay_hetero",
        "command_mode": "autonomous",
        "stress": "severe",
        "seed": 967915163,
        "replicate": 3,
    },
    {
        "label": "relay_rich_autonomous_severe_rep3",
        "team": "relay_rich",
        "command_mode": "autonomous",
        "stress": "severe",
        "seed": 678182620,
        "replicate": 3,
    },
]


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def build_full_trials(seeds: int = 8) -> list[dict[str, object]]:
    teams = ["no_relay_hetero", "relay_sparse", "balanced_hetero", "relay_rich"]
    commands = ["autonomous", "crowd_vector", "consensus_gated"]
    stresses = ["degraded", "severe"]
    trials = []
    for team in teams:
        for command_mode in commands:
            for stress in stresses:
                for rep in range(seeds):
                    trials.append(
                        {
                            "team": team,
                            "command_mode": command_mode,
                            "stress": stress,
                            "seed": stable_seed("webots", team, command_mode, stress, rep),
                        }
                    )
    return trials


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def norm(value: object) -> str:
    text = str(value)
    try:
        number = float(text)
        if abs(number - round(number)) < 1e-12:
            return str(int(round(number)))
        return format(number, ".12g")
    except Exception:
        return text


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def match_archive(row: dict[str, str], archive_rows: list[dict[str, str]]) -> dict[str, str]:
    for archive in archive_rows:
        if (
            archive["team"] == row["team"]
            and archive["command_mode"] == row["command_mode"]
            and archive["stress"] == row["stress"]
            and archive["seed"] == row["seed"]
        ):
            return archive
    raise RuntimeError(f"archive Webots row not found for {row}")


def compare_rows(recorded: dict[str, str], archive: dict[str, str]) -> list[dict[str, str]]:
    mismatches = []
    for column in archive:
        if column not in recorded:
            mismatches.append({"column": column, "archive": archive[column], "recorded": "<missing>"})
        elif norm(archive[column]) != norm(recorded[column]):
            mismatches.append({"column": column, "archive": archive[column], "recorded": recorded[column]})
    return mismatches


def run_trial(
    webots: Path,
    trial: dict[str, object],
    timeout: int,
    *,
    record_movie: bool,
    no_rendering: bool,
    prefix_replay: bool,
) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    THEORY_OUT.mkdir(parents=True, exist_ok=True)
    prefix = "webots_movie" if record_movie else "webots_replay_verify"
    output_path = OUT / f"{prefix}_{trial['label']}_runs.csv"
    config_path = OUT / f"{prefix}_{trial['label']}_config.json"
    log_path = OUT / f"{prefix}_{trial['label']}_console.log"
    movie_path = OUT / f"{prefix}_{trial['label']}.mp4"

    target = {
        "team": trial["team"],
        "command_mode": trial["command_mode"],
        "stress": trial["stress"],
        "seed": trial["seed"],
    }
    if prefix_replay:
        full_trials = build_full_trials(seeds=8)
        target_index = None
        for idx, candidate in enumerate(full_trials):
            if all(str(candidate[key]) == str(value) for key, value in target.items()):
                target_index = idx
                break
        if target_index is None:
            raise RuntimeError(f"target not found in full Webots order: {target}")
        configured_trials = full_trials[: target_index + 1]
    else:
        target_index = 0
        configured_trials = [target]
    config = {
        "output_path": str(output_path),
        "trials": configured_trials,
        "movie_match": target if record_movie else None,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["WEBOTS_SWARM_CONFIG"] = str(config_path)
    if record_movie:
        env["WEBOTS_SWARM_MOVIE_PATH"] = str(movie_path)
        env["WEBOTS_SWARM_MOVIE_WIDTH"] = "1280"
        env["WEBOTS_SWARM_MOVIE_HEIGHT"] = "720"
        env["WEBOTS_SWARM_MOVIE_CODEC"] = "2"
        env["WEBOTS_SWARM_MOVIE_QUALITY"] = "60"
        env["WEBOTS_SWARM_MOVIE_ACCELERATION"] = "4"
        env["WEBOTS_SWARM_MOVIE_CAPTION"] = "0"
    env["WEBOTS_HOME"] = str(webots.parents[3])
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        str(webots),
        "--batch",
        "--mode=fast",
    ]
    if no_rendering:
        cmd.append("--no-rendering")
    cmd.extend(["--stdout", "--stderr", str(WORLD)])
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    wall_clock_s = time.perf_counter() - start
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Webots returned {proc.returncode}; see {log_path}")
    if not output_path.exists():
        raise RuntimeError(f"Webots movie trial did not produce output CSV: {output_path}")
    rows = read_rows(output_path)
    target_rows = [
        row
        for row in rows
        if row["team"] == str(target["team"])
        and row["command_mode"] == str(target["command_mode"])
        and row["stress"] == str(target["stress"])
        and row["seed"] == str(target["seed"])
    ]
    if len(target_rows) != 1:
        raise RuntimeError(f"expected 1 target row, got {len(target_rows)} in {output_path}")
    movie_exists = movie_path.exists()
    movie_size = movie_path.stat().st_size if movie_exists else 0
    pixel_qa = None
    if record_movie and movie_exists and movie_size > 0:
        pixel_qa = check_movie(movie_path)
        pixel_qa["path"] = rel(movie_path)
    return {
        "trial": trial,
        "record_movie": record_movie,
        "no_rendering": no_rendering,
        "prefix_replay": prefix_replay,
        "configured_trials": len(configured_trials),
        "target_index_zero_based": target_index,
        "output_path": rel(output_path),
        "config_path": rel(config_path),
        "log_path": rel(log_path),
        "movie_path": rel(movie_path),
        "movie_exists": movie_exists,
        "movie_size_bytes": movie_size,
        "movie_size_mb": movie_size / (1024 * 1024),
        "wall_clock_s": wall_clock_s,
        "row": target_rows[0],
        "movie_ready_in_log": "WEBOTS_MOVIE_READY" in proc.stdout or "INFO: Video creation finished." in proc.stdout,
        "movie_failed_in_log": "WEBOTS_MOVIE_FAILED" in proc.stdout
        or "WEBOTS_MOVIE_NOT_READY_AFTER_WAIT" in proc.stdout
        or "Video creation canceled" in proc.stdout,
        "pixel_qa": pixel_qa,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record sealed Webots movie trials.")
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--verify-only", action="store_true", help="Replay sealed trials without movie recording.")
    parser.add_argument("--no-rendering", action="store_true", help="Use Webots --no-rendering for replay verification.")
    parser.add_argument("--prefix-replay", action="store_true", help="Replay the original 192-run order prefix up to each target trial.")
    parser.add_argument("--label", default="all", help="Run one trial label or all.")
    args = parser.parse_args()

    if not args.webots.exists():
        raise FileNotFoundError(f"Webots executable not found: {args.webots}")
    archive_rows = read_rows(ROOT / "data" / "webots_runs.csv")
    trial_reports = []
    selected_trials = TRIALS if args.label == "all" else [trial for trial in TRIALS if trial["label"] == args.label]
    if not selected_trials:
        raise RuntimeError(f"unknown label {args.label}; valid labels: {[trial['label'] for trial in TRIALS]}")
    for trial in selected_trials:
        report = run_trial(
            args.webots,
            trial,
            args.timeout,
            record_movie=not args.verify_only,
            no_rendering=args.no_rendering,
            prefix_replay=args.prefix_replay,
        )
        archive = match_archive(report["row"], archive_rows)
        mismatches = compare_rows(report["row"], archive)
        report["archive_row"] = archive
        report["metric_mismatches"] = mismatches
        report["metrics_match_archive"] = len(mismatches) == 0
        if args.verify_only:
            report["usable"] = report["metrics_match_archive"]
        else:
            report["usable"] = (
                report["movie_exists"]
                and report["movie_size_bytes"] > 0
                and report["movie_size_mb"] <= 100.0
                and report["movie_ready_in_log"]
                and not report["movie_failed_in_log"]
                and bool(report.get("pixel_qa", {}).get("passed"))
                and report["metrics_match_archive"]
            )
        trial_reports.append(report)

    selection = {
        "protocol": "webots_movie_selection_and_recording"
        if not args.verify_only
        else "webots_replay_verification_without_movie",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "replicate 3 selected by sealed operational-score contrast ranking; safe delivery and attrition were not used for selection",
        "webots_executable": str(args.webots),
        "world": str(WORLD),
        "trials": trial_reports,
        "all_movies_usable": all(report["usable"] for report in trial_reports),
    }
    out = OUT / ("webots_movie_selection.json" if not args.verify_only else "webots_replay_verification.json")
    out.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "all_movies_usable": selection["all_movies_usable"],
                "movies": [
                    {
                        "label": report["trial"]["label"],
                        "exists": report["movie_exists"],
                        "size_mb": report["movie_size_mb"],
                        "metrics_match_archive": report["metrics_match_archive"],
                        "usable": report["usable"],
                    }
                    for report in trial_reports
                ],
            },
            indent=2,
        )
    )
    if not selection["all_movies_usable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
