from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_relay_frontier_v15 import compute_freshness, compute_layer1, compute_track_e
from check_webots_movie_pixels import check_movie


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


def portable_path(value: Any) -> Path:
    return Path(str(value).replace("\\", "/"))


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
        "scripts/lambda2_utils.py",
        "tests/test_lambda2_utils.py",
        "data/relay_sweep_reinstrumented_v15.csv",
        "data/s1_lambda2_runs.csv",
        "data/s1_lambda2_condition_summary.csv",
        "data/s1_lambda2_team_summary.csv",
        "reports/layer1_slope_cellpreserving.json",
        "reports/trackE_selection_frequency_final.json",
        "reports/freshness_return_cellpreserving_v15.json",
        "reports/s2_greedy_allocation.json",
        "reports/s1_g1g2_identity.json",
        "reports/s1_directional_derivative_gate.json",
        "reports/s1_smoke_implementation_gates.json",
        "reports/s1_production_merge_report.json",
        "reports/s1_lambda2_contrasts.csv",
        "reports/s1_lambda2_summary.json",
        "reports/revision_result_brief_v15.md",
        "reports/webots_movie_pixel_qa.json",
        "revision_outputs/webots_timing_log.csv",
        "revision_outputs/webots_movie_selection.json",
        "revision_outputs/webots_movie_no_relay_hetero_autonomous_severe_rep3.mp4",
        "revision_outputs/webots_movie_relay_rich_autonomous_severe_rep3.mp4",
        "revision_outputs/webots_movie_side_by_side_tradeoff.mp4",
        "scripts/analyze_relay_frontier_v15.py",
        "scripts/build_webots_comparison_movie.py",
        "scripts/check_webots_movie_pixels.py",
        "webots_validation/run_webots_movie_trial.py",
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
    relay_reinstrumented = pd.read_csv(data / "relay_sweep_reinstrumented_v15.csv")
    s1_runs = pd.read_csv(data / "s1_lambda2_runs.csv")
    s1_condition_summary = pd.read_csv(data / "s1_lambda2_condition_summary.csv")
    s1_team_summary = pd.read_csv(data / "s1_lambda2_team_summary.csv")
    s1_contrasts = pd.read_csv(root / "reports" / "s1_lambda2_contrasts.csv")
    s1_summary = json.loads((root / "reports" / "s1_lambda2_summary.json").read_text(encoding="utf-8"))
    g2_identity = json.loads((root / "reports" / "s1_g1g2_identity.json").read_text(encoding="utf-8"))
    i2_gate = json.loads((root / "reports" / "s1_directional_derivative_gate.json").read_text(encoding="utf-8"))
    smoke_gates = json.loads((root / "reports" / "s1_smoke_implementation_gates.json").read_text(encoding="utf-8"))
    production_merge = json.loads((root / "reports" / "s1_production_merge_report.json").read_text(encoding="utf-8"))
    s2 = json.loads((root / "reports" / "s2_greedy_allocation.json").read_text(encoding="utf-8"))
    freshness_v15 = json.loads((root / "reports" / "freshness_return_cellpreserving_v15.json").read_text(encoding="utf-8"))
    track_e = json.loads((root / "reports" / "trackE_selection_frequency_final.json").read_text(encoding="utf-8"))
    layer1 = json.loads((root / "reports" / "layer1_slope_cellpreserving.json").read_text(encoding="utf-8"))
    movie_selection = json.loads((root / "revision_outputs" / "webots_movie_selection.json").read_text(encoding="utf-8"))
    movie_pixel_report = json.loads((root / "reports" / "webots_movie_pixel_qa.json").read_text(encoding="utf-8"))

    add_count_check(checks, q2_runs, "q2_runs", 9024)
    add_count_check(checks, relay_runs, "relay_sweep_runs", 4800)
    add_count_check(checks, grid_runs, "robotics_grid_runs", 2304)
    add_count_check(checks, webots_runs, "webots_runs", 192)
    add_count_check(checks, q2, "q2_bootstrap_contrasts", 48)
    add_count_check(checks, relay, "relay_sweep_bootstrap", 72)
    add_count_check(checks, grid, "robotics_grid_contrasts", 40)
    add_count_check(checks, webots, "webots_contrasts", 38)
    add_count_check(checks, relay_reinstrumented, "relay_sweep_reinstrumented_v15", 4800)
    add_count_check(checks, s1_runs, "s1_lambda2_runs", 3840)
    add_count_check(checks, s1_condition_summary, "s1_lambda2_condition_summary", 96)
    add_count_check(checks, s1_team_summary, "s1_lambda2_team_summary", 8)
    add_count_check(checks, s1_contrasts, "s1_lambda2_contrasts", 32)

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

    checks.append(
        Check(
            "s1:g2_identity_passed",
            bool(g2_identity.get("passed")),
            g2_identity.get("passed"),
            True,
            detail="no-relay S1 identity, excluding the panel-relative degraded_outcome relabeling noted in the report",
        )
    )
    checks.append(Check("s1:i2_directional_derivative_passed", bool(i2_gate.get("passed")), i2_gate.get("passed"), True))
    checks.append(
        Check(
            "s1:smoke_all_gates_passed",
            bool(smoke_gates.get("all_gates_passed")),
            smoke_gates.get("all_gates_passed"),
            True,
        )
    )
    checks.append(
        Check(
            "s1:production_merge_passed",
            bool(production_merge.get("passed")),
            production_merge.get("passed"),
            True,
        )
    )
    checks.append(
        Check(
            "s1:branch_by_stress",
            s1_summary.get("branch_by_stress") == {"degraded": "B", "severe": "B"},
            s1_summary.get("branch_by_stress"),
            {"degraded": "B", "severe": "B"},
        )
    )
    checks.append(
        Check(
            "s1:pairing_missing",
            int(s1_summary.get("pairing_missing", -1)) == 0,
            s1_summary.get("pairing_missing"),
            0,
        )
    )
    add_value_check(checks, "s1:k4_degraded_safe_mean_diff", s1_summary["k4_primary"]["degraded"]["mean_diff"], 0.0104167)
    add_value_check(checks, "s1:k4_severe_safe_mean_diff", s1_summary["k4_primary"]["severe"]["mean_diff"], 0.0125000)
    checks.append(Check("s2:selected_k", int(s2.get("selected_k", -1)) == 4, s2.get("selected_k"), 4))
    add_value_check(checks, "freshness_v15:degraded_R", freshness_v15["results"]["degraded"]["R"], 0.363017)
    add_value_check(checks, "freshness_v15:severe_R", freshness_v15["results"]["severe"]["R"], 0.504420)
    checks.append(
        Check(
            "trackE:delay_selects_k4_degraded",
            float(track_e["results"]["degraded|mean_effective_map_delay"]["freq"][4]) == 1.0,
            track_e["results"]["degraded|mean_effective_map_delay"]["freq"][4],
            1.0,
        )
    )

    generated_layer1 = compute_layer1(data)
    generated_track_e = compute_track_e(data)
    generated_freshness = compute_freshness(data)
    checks.append(
        Check(
            "frontier_regeneration:layer1_summary",
            generated_layer1["summary"] == layer1["summary"],
            generated_layer1["summary"],
            layer1["summary"],
        )
    )
    for key, expected in layer1["strata"].items():
        observed = generated_layer1["strata"][key]
        for metric in ["c", "ci_low", "ci_high", "boot_frac_positive"]:
            add_value_check(checks, f"frontier_regeneration:layer1:{key}:{metric}", observed[metric], expected[metric])
        checks.append(
            Check(
                f"frontier_regeneration:layer1:{key}:excludes_zero",
                observed["excludes_zero"] == expected["excludes_zero"],
                observed["excludes_zero"],
                expected["excludes_zero"],
            )
        )
    for key, expected in track_e["results"].items():
        observed = generated_track_e["results"][key]
        checks.append(Check(f"frontier_regeneration:trackE:{key}:rule", observed["rule"] == expected["rule"], observed["rule"], expected["rule"]))
        add_value_check(checks, f"frontier_regeneration:trackE:{key}:k_ge2", observed["k_ge2"], expected["k_ge2"])
        for idx, (obs_value, exp_value) in enumerate(zip(observed["freq"], expected["freq"])):
            add_value_check(checks, f"frontier_regeneration:trackE:{key}:freq_k{idx}", obs_value, exp_value)
    for stress, expected in freshness_v15["results"].items():
        observed = generated_freshness["results"][stress]
        for metric in ["g", "c", "beta", "R", "ci_low", "ci_high", "beta_crit_over_beta"]:
            add_value_check(checks, f"frontier_regeneration:freshness:{stress}:{metric}", observed[metric], expected[metric])
        for metric in ["n_resamples_valid", "n_invalid_denominator", "n_resamples_R_below_1"]:
            checks.append(
                Check(
                    f"frontier_regeneration:freshness:{stress}:{metric}",
                    int(observed[metric]) == int(expected[metric]),
                    observed[metric],
                    expected[metric],
                )
            )
    checks.append(
        Check(
            "webots_movie:all_usable",
            bool(movie_selection.get("all_movies_usable")),
            movie_selection.get("all_movies_usable"),
            True,
        )
    )
    for movie in movie_selection.get("trials", []):
        movie_label = movie.get("label") or movie.get("trial", {}).get("label")
        checks.append(
            Check(
                f"webots_movie:{movie_label}:metrics_match_archive",
                bool(movie.get("metrics_match_archive")),
                movie.get("metrics_match_archive"),
                True,
            )
        )
        checks.append(
            Check(
                f"webots_movie:{movie_label}:size_under_100mb",
                0 < float(movie.get("movie_size_mb", 0.0)) <= 100.0,
                movie.get("movie_size_mb"),
                "<=100 MB and >0",
            )
        )
        movie_path = portable_path(movie.get("movie_path", ""))
        if not movie_path.is_absolute():
            movie_path = root / movie_path
        pixel_qa = check_movie(movie_path)
        checks.append(
            Check(
                f"webots_movie:{movie_label}:pixel_qa",
                bool(pixel_qa["passed"]),
                {
                    "frames": pixel_qa["frames"],
                    "max_nonzero_pixels_below_y_min": pixel_qa["max_nonzero_pixels_below_y_min"],
                    "mean_rgb_below_y_min": pixel_qa["mean_rgb_below_y_min"],
                    "max_std_rgb_below_y_min": pixel_qa["max_std_rgb_below_y_min"],
                },
                "nonblack high-contrast scene below y=100",
            )
        )
    checks.append(
        Check(
            "webots_movie_pixel_report:passed",
            bool(movie_pixel_report.get("passed")),
            movie_pixel_report.get("passed"),
            True,
        )
    )
    checks.append(
        Check(
            "webots_movie_pixel_report:three_movies",
            len(movie_pixel_report.get("checks", [])) >= 3,
            len(movie_pixel_report.get("checks", [])),
            ">=3",
        )
    )
    side_by_side_path = root / "revision_outputs" / "webots_movie_side_by_side_tradeoff.mp4"
    side_by_side_qa = check_movie(side_by_side_path)
    checks.append(
        Check(
            "webots_movie:side_by_side_tradeoff:pixel_qa",
            bool(side_by_side_qa["passed"]),
            {
                "frames": side_by_side_qa["frames"],
                "max_nonzero_pixels_below_y_min": side_by_side_qa["max_nonzero_pixels_below_y_min"],
                "mean_rgb_below_y_min": side_by_side_qa["mean_rgb_below_y_min"],
                "max_std_rgb_below_y_min": side_by_side_qa["max_std_rgb_below_y_min"],
            },
            "nonblack high-contrast side-by-side scene below y=100",
        )
    )

    passed = sum(1 for check in checks if check.passed)
    total = len(checks)
    rate = passed / total if total else 0.0
    return {
        "passed": passed,
        "total": total,
        "reproducibility_rate": rate,
        "threshold": 1.0,
        "status": "PASS" if passed == total else "FAIL",
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
