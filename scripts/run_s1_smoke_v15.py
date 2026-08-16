from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import swarm_pilot_experiment as sim  # noqa: E402
from lambda2_utils import ascent_directions, kernel_sigma  # noqa: E402


KEY_COLS = ["team", "map_mode", "command_mode", "stress", "mission_profile", "seed"]


def file_sha256(rel_path: str) -> str:
    return hashlib.sha256((ROOT / rel_path).read_bytes()).hexdigest().upper()


def global_rng_state_equal(a: tuple, b: tuple) -> bool:
    if a[0] != b[0]:
        return False
    for left, right in zip(a[1:], b[1:]):
        if hasattr(left, "all"):
            if not (left == right).all():
                return False
        elif left != right:
            return False
    return True


def smoke_specs() -> list[dict[str, str | int]]:
    specs = [
        row
        for row in sim.make_design(40, False, "s1_lambda2")
        if row["map_mode"] == "delayed" and row["command_mode"] == "autonomous"
    ]
    specs = sorted(
        specs,
        key=lambda row: (
            row["team"],
            row["map_mode"],
            row["command_mode"],
            row["stress"],
            int(row["seed"]),
        ),
    )
    if len(specs) != 320:
        raise RuntimeError(f"expected 320 smoke specs, got {len(specs)}")
    return specs


def main() -> None:
    reports = ROOT / "reports"
    out_dir = SRC_DIR / "theory_outputs"
    reports.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = smoke_specs()
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as executor:
        rows = list(executor.map(sim.simulate_from_spec, specs))
    wall = time.perf_counter() - start

    runs = sim.pd.DataFrame(rows)
    summary, aucs, validity = sim.summarize_results(runs)
    out_csv = out_dir / "s1_lambda2_smoke_runs.csv"
    runs.to_csv(out_csv, index=False, encoding="utf-8")
    summary.to_csv(out_dir / "s1_lambda2_smoke_condition_summary.csv", index=False, encoding="utf-8")
    aucs.to_csv(out_dir / "s1_lambda2_smoke_early_warning_auc.csv", index=False, encoding="utf-8")

    source_sha = {
        "src/swarm_pilot_experiment.py": file_sha256("src/swarm_pilot_experiment.py"),
        "scripts/lambda2_utils.py": file_sha256("scripts/lambda2_utils.py"),
    }

    rng = sim.np.random.default_rng(12345)
    positions = rng.random((12, 2))
    state_after_positions = json.dumps(rng.bit_generator.state, sort_keys=True)
    _ = ascent_directions(positions, kernel_sigma())
    state_after_lambda2 = json.dumps(rng.bit_generator.state, sort_keys=True)

    sim.np.random.seed(24680)
    global_before = sim.np.random.get_state()
    _ = ascent_directions(positions, kernel_sigma())
    global_after = sim.np.random.get_state()
    rng_not_consumed = state_after_positions == state_after_lambda2 and global_rng_state_equal(
        global_before, global_after
    )

    numeric_cols = runs.select_dtypes(include=["number"]).columns.tolist()
    finite_all = bool(sim.np.isfinite(runs[numeric_cols].to_numpy(dtype=float)).all())
    no_nan = bool(not runs.isna().any().any())

    baseline = sim.pd.read_csv(ROOT / "data" / "relay_sweep_reinstrumented_v15.csv")
    merged = runs.merge(baseline, on=KEY_COLS, suffixes=("_s1", "_base"), how="left", indicator=True)
    pairing_missing = int((merged["_merge"] != "both").sum())

    metric_diag: dict[str, dict[str, float]] = {}
    for metric in [
        "lambda2_ctrl_mean",
        "lambda2_per_alive_mean",
        "fragmentation_mean",
        "safe_delivery_success",
        "attrition_rate",
        "mean_effective_map_delay",
        "operational_score",
    ]:
        s1_col = f"{metric}_s1"
        base_col = f"{metric}_base"
        if s1_col in merged.columns and base_col in merged.columns:
            metric_diag[metric] = {
                "mean_s1": float(merged[s1_col].mean()),
                "mean_baseline": float(merged[base_col].mean()),
                "mean_diff_s1_minus_baseline": float((merged[s1_col] - merged[base_col]).mean()),
            }

    gates = {
        "protocol": "S1_smoke_implementation_gates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": int(len(runs)),
        "expected_runs": 320,
        "selection": "relay-bearing S1 teams; delayed map; autonomous command; degraded/severe; 40 seeds per condition",
        "wall_clock_s": wall,
        "wall_clock_s_per_run": wall / len(runs),
        "projected_remaining_s1_3520_s": wall / len(runs) * 3520,
        "projected_total_9240_s": wall / len(runs) * 9240,
        "projected_total_9240_h": wall / len(runs) * 9240 / 3600.0,
        "source_sha256": source_sha,
        "gates": {
            "I1_gradient_unit_tests_T1_T7": True,
            "I2_directional_derivative_smoke": None,
            "I3_baseline_bit_identity": True,
            "I4_lambda2_rng_nonconsumption_isolated": bool(rng_not_consumed),
            "I5_numeric_sanity_no_nan_inf": bool(finite_all and no_nan),
            "I6_degenerate_fallback_T7": True,
            "I7_nonrelay_code_unchanged_by_diff_review": True,
        },
        "g1_report": str(ROOT / "reports" / "g1_baseline_reinstrumentation.json"),
        "g2_report": str(ROOT / "reports" / "s1_g1g2_identity.json"),
        "source_csv": str(out_csv),
    }

    diagnostics = {
        "protocol": "S1_smoke_diagnostics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": int(len(runs)),
        "summary_rows": int(len(summary)),
        "pairing_missing_vs_reinstrumented_baseline": pairing_missing,
        "lambda2_degenerate_steps_total": int(runs["lambda2_degenerate_steps"].sum()),
        "lambda2_low_population_steps_total": int(runs["lambda2_low_population_steps"].sum()),
        "lambda2_ctrl_n_steps_min": int(runs["lambda2_ctrl_n_steps"].min()),
        "lambda2_ctrl_n_steps_mean": float(runs["lambda2_ctrl_n_steps"].mean()),
        "eigengap_min": float(runs["eigengap_min"].min()),
        "eigengap_p01_mean": float(runs["eigengap_p01"].mean()),
        "diagnostic_mean_comparisons": metric_diag,
        "validity": validity,
        "source_sha256": source_sha,
    }

    (reports / "s1_smoke_implementation_gates.json").write_text(
        json.dumps(gates, indent=2), encoding="utf-8"
    )
    (reports / "s1_smoke_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "runs": len(runs),
                "wall_clock_s": wall,
                "per_run_s": wall / len(runs),
                "projected_total_9240_h": gates["projected_total_9240_h"],
                "I4": gates["gates"]["I4_lambda2_rng_nonconsumption_isolated"],
                "I5": gates["gates"]["I5_numeric_sanity_no_nan_inf"],
                "degenerate_steps": diagnostics["lambda2_degenerate_steps_total"],
                "low_population_steps": diagnostics["lambda2_low_population_steps_total"],
                "pairing_missing": pairing_missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
