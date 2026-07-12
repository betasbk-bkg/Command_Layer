from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Check:
    name: str
    passed: bool
    observed: Any
    expected: Any
    tolerance: float | None = None
    detail: str = ""


def close(observed: float, expected: float, tolerance: float = 5e-4) -> bool:
    return abs(float(observed) - float(expected)) <= tolerance


def add_file_check(checks: list[Check], root: Path, rel: str) -> None:
    path = root / rel
    checks.append(Check(f"file_exists:{rel}", path.exists(), str(path), "exists"))


def add_count_check(checks: list[Check], frame: pd.DataFrame, name: str, expected: int) -> None:
    observed = len(frame)
    checks.append(Check(f"row_count:{name}", observed == expected, observed, expected))


def add_value_check(checks: list[Check], name: str, observed: float, expected: float, tolerance: float = 5e-4) -> None:
    checks.append(Check(name, close(observed, expected, tolerance), float(observed), expected, tolerance))


def find_q2(df: pd.DataFrame, contrast: str, metric: str) -> pd.Series:
    row = df[(df["contrast"] == contrast) & (df["metric"] == metric)]
    if row.empty:
        raise KeyError(f"missing q2 row: {contrast} / {metric}")
    return row.iloc[0]


def find_relay(df: pd.DataFrame, stress: str, contrast: str, metric: str) -> pd.Series:
    row = df[(df["stress"] == stress) & (df["contrast"] == contrast) & (df["metric"] == metric)]
    if row.empty:
        raise KeyError(f"missing relay row: {stress} / {contrast} / {metric}")
    return row.iloc[0]


def find_general(df: pd.DataFrame, stress: str, contrast: str, metric: str) -> pd.Series:
    row = df[(df["stress"] == stress) & (df["contrast"] == contrast) & (df["metric"] == metric)]
    if row.empty:
        raise KeyError(f"missing contrast row: {stress} / {contrast} / {metric}")
    return row.iloc[0]


def mean_delta(frame: pd.DataFrame, stress: str, contrast: str, metric: str) -> float:
    sub = frame[frame["stress"] == stress]
    if contrast == "relay_rich_minus_no_relay":
        left = sub[sub["team"] == "relay_rich"][metric].astype(float)
        right = sub[sub["team"] == "no_relay_hetero"][metric].astype(float)
    elif contrast == "crowd_vector_minus_auto":
        left = sub[sub["command_mode"] == "crowd_vector"][metric].astype(float)
        right = sub[sub["command_mode"] == "autonomous"][metric].astype(float)
    elif contrast == "consensus_gated_minus_auto":
        left = sub[sub["command_mode"] == "consensus_gated"][metric].astype(float)
        right = sub[sub["command_mode"] == "autonomous"][metric].astype(float)
    else:
        raise ValueError(f"unsupported contrast: {contrast}")
    return float(left.mean() - right.mean())


def run_checks(root: Path) -> dict[str, Any]:
    checks: list[Check] = []
    required_files = [
        "README.md",
        "REPRODUCE.md",
        "requirements.txt",
        "repository_metadata.json",
        "src/swarm_pilot_experiment.py",
        "src/robotics_extension_experiments.py",
        "webots_validation/worlds/swarm_validation.wbt",
        "webots_validation/controllers/swarm_supervisor/swarm_supervisor.py",
        "webots_validation/controllers/swarm_bot/swarm_bot.py",
        "data/q2_runs.csv",
        "data/q2_bootstrap_contrasts.csv",
        "data/relay_sweep_runs.csv",
        "data/relay_sweep_bootstrap.csv",
        "data/robotics_grid_runs.csv",
        "data/robotics_grid_contrasts.csv",
        "data/webots_runs.csv",
        "data/webots_contrasts.csv",
        "reports/validation_summary.md",
    ]
    for rel in required_files:
        add_file_check(checks, root, rel)

    data = root / "data"
    q2_runs = pd.read_csv(data / "q2_runs.csv")
    relay_runs = pd.read_csv(data / "relay_sweep_runs.csv")
    grid_runs = pd.read_csv(data / "robotics_grid_runs.csv")
    webots_runs = pd.read_csv(data / "webots_runs.csv")
    q2 = pd.read_csv(data / "q2_bootstrap_contrasts.csv")
    relay = pd.read_csv(data / "relay_sweep_bootstrap.csv")
    grid = pd.read_csv(data / "robotics_grid_contrasts.csv")
    webots = pd.read_csv(data / "webots_contrasts.csv")
    diagnostic = pd.read_csv(data / "model_diagnostic_auc_confirmatory.csv")

    add_count_check(checks, q2_runs, "q2_runs", 9024)
    add_count_check(checks, relay_runs, "relay_sweep_runs", 4800)
    add_count_check(checks, grid_runs, "robotics_grid_runs", 2304)
    add_count_check(checks, webots_runs, "webots_runs", 192)
    add_count_check(checks, q2, "q2_bootstrap_contrasts", 48)
    add_count_check(checks, relay, "relay_sweep_bootstrap", 72)
    add_count_check(checks, grid, "robotics_grid_contrasts", 40)
    add_count_check(checks, webots, "webots_contrasts", 38)

    q2_expected = [
        ("A crowd vs autonomous, degraded", "safe_delivery_success", "delta_left_minus_right", 0.087500),
        ("A crowd vs autonomous, degraded", "operational_score", "delta_left_minus_right", 2.327639),
        ("A crowd vs autonomous, degraded", "degraded_outcome", "delta_left_minus_right", -0.147917),
        ("A crowd vs autonomous, degraded", "attrition_rate", "delta_left_minus_right", -0.009549),
        ("A crowd vs autonomous, severe", "safe_delivery_success", "delta_left_minus_right", 0.058333),
        ("A crowd vs autonomous, severe", "operational_score", "delta_left_minus_right", 1.798556),
        ("A crowd vs autonomous, severe", "degraded_outcome", "delta_left_minus_right", -0.108333),
        ("E relay_rich vs no_relay_hetero", "safe_delivery_success", "delta_left_minus_right", -0.166667),
        ("E relay_rich vs no_relay_hetero", "operational_score", "delta_left_minus_right", -1.535333),
        ("E relay_rich vs no_relay_hetero", "attrition_rate", "delta_left_minus_right", 0.025000),
        ("E relay_rich vs no_relay_hetero", "mean_effective_map_delay", "delta_left_minus_right", -11.261589),
    ]
    for contrast, metric, column, expected in q2_expected:
        row = find_q2(q2, contrast, metric)
        add_value_check(checks, f"q2:{contrast}:{metric}:{column}", row[column], expected)

    relay_expected = [
        ("degraded", "relay_4_minus_0", "safe_delivery_success", -0.127083),
        ("degraded", "relay_4_minus_0", "operational_score", -1.158813),
        ("degraded", "relay_4_minus_0", "attrition_rate", 0.017361),
        ("degraded", "relay_4_minus_0", "mean_effective_map_delay", -7.689147),
        ("severe", "relay_4_minus_0", "safe_delivery_success", -0.156250),
        ("severe", "relay_4_minus_0", "operational_score", -1.522693),
        ("severe", "relay_4_minus_0", "attrition_rate", 0.031424),
        ("severe", "relay_4_minus_0", "mean_effective_map_delay", -14.769210),
    ]
    for stress, contrast, metric, expected in relay_expected:
        row = find_relay(relay, stress, contrast, metric)
        add_value_check(checks, f"relay_sweep:{stress}:{contrast}:{metric}", row["delta"], expected)

    grid_expected = [
        ("degraded", "crowd_vector_minus_auto", "safe_delivery_success", -0.076389),
        ("degraded", "crowd_vector_minus_auto", "operational_score", -2.792077),
        ("severe", "relay_rich_minus_no_relay", "safe_delivery_success", -0.038194),
        ("severe", "relay_rich_minus_no_relay", "attrition_rate", 0.026563),
        ("severe", "relay_rich_minus_no_relay", "mean_effective_map_delay", -7.000000),
    ]
    for stress, contrast, metric, expected in grid_expected:
        row = find_general(grid, stress, contrast, metric)
        add_value_check(checks, f"grid_table:{stress}:{contrast}:{metric}", row["delta"], expected)
        add_value_check(checks, f"grid_raw:{stress}:{contrast}:{metric}", mean_delta(grid_runs, stress, contrast, metric), expected)

    webots_expected = [
        ("degraded", "relay_rich_minus_no_relay", "safe_delivery_success", -0.333333),
        ("degraded", "relay_rich_minus_no_relay", "degraded_outcome", 0.333333),
        ("degraded", "relay_rich_minus_no_relay", "attrition_rate", 0.114583),
        ("degraded", "relay_rich_minus_no_relay", "mean_effective_map_delay", -12.000000),
        ("severe", "relay_rich_minus_no_relay", "safe_delivery_success", -0.625000),
        ("severe", "relay_rich_minus_no_relay", "degraded_outcome", 0.625000),
        ("severe", "relay_rich_minus_no_relay", "attrition_rate", 0.149306),
        ("severe", "relay_rich_minus_no_relay", "mean_effective_map_delay", -22.000000),
        ("degraded", "crowd_vector_minus_auto", "safe_delivery_success", 0.156250),
        ("degraded", "consensus_gated_minus_auto", "operational_score", 7.058089),
    ]
    for stress, contrast, metric, expected in webots_expected:
        row = find_general(webots, stress, contrast, metric)
        add_value_check(checks, f"webots_table:{stress}:{contrast}:{metric}", row["delta"], expected)
        add_value_check(checks, f"webots_raw:{stress}:{contrast}:{metric}", mean_delta(webots_runs, stress, contrast, metric), expected)

    diag_expected = {
        "command_geometry": 0.582322,
        "combined_no_attrition": 0.654073,
        "combined_with_attrition": 0.766337,
    }
    for predictor, expected in diag_expected.items():
        row = diagnostic[diagnostic["predictor"] == predictor].iloc[0]
        add_value_check(checks, f"diagnostic_auc:{predictor}", row["auroc"], expected)

    passed = sum(1 for check in checks if check.passed)
    total = len(checks)
    rate = passed / total if total else 0.0
    return {
        "passed": passed,
        "total": total,
        "reproducibility_rate": rate,
        "threshold": 0.95,
        "status": "PASS" if rate >= 0.95 else "FAIL",
        "checks": [check.__dict__ for check in checks],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify claim-level reproducibility for the repository package.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository package root.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()
    result = run_checks(args.root.resolve())
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
