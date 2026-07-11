"""Run the Webots validation batch and analyze the resulting CSV."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "theory_outputs"
WORLD = ROOT / "webots_validation" / "worlds" / "swarm_validation.wbt"
DEFAULT_WEBOTS = Path(r"D:\Webots\msys64\mingw64\bin\webots.exe")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def build_trials(seeds: int, smoke: bool) -> list[dict[str, object]]:
    teams = ["no_relay_hetero", "relay_sparse", "balanced_hetero", "relay_rich"]
    commands = ["autonomous", "crowd_vector", "consensus_gated"]
    stresses = ["degraded", "severe"]
    if smoke:
        teams = ["no_relay_hetero", "relay_rich"]
        commands = ["autonomous", "crowd_vector"]
        stresses = ["degraded"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Webots multi-robot validation.")
    parser.add_argument("--webots", type=Path, default=DEFAULT_WEBOTS)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    if not args.webots.exists():
        raise FileNotFoundError(f"Webots executable not found: {args.webots}")
    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / ("webots_smoke_runs.csv" if args.smoke else "webots_runs.csv")
    config_path = OUT / ("webots_smoke_config.json" if args.smoke else "webots_batch_config.json")
    log_path = OUT / ("webots_smoke_console.log" if args.smoke else "webots_console.log")
    config = {
        "output_path": str(output_path),
        "trials": build_trials(args.seeds, args.smoke),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["WEBOTS_SWARM_CONFIG"] = str(config_path)
    env["WEBOTS_HOME"] = str(args.webots.parents[3])
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        str(args.webots),
        "--batch",
        "--mode=fast",
        "--no-rendering",
        "--stdout",
        "--stderr",
        str(WORLD),
    ]
    print(f"Running {len(config['trials'])} Webots trials...", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
    )
    log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        raise SystemExit(proc.returncode)

    analyze_cmd = [sys.executable, str(ROOT / "webots_validation" / "analyze_webots_results.py"), "--input", str(output_path)]
    subprocess.run(analyze_cmd, cwd=str(ROOT), check=True)
    print(f"Wrote {output_path}", flush=True)
    print(f"Console log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
