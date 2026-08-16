from __future__ import annotations

import hashlib
import json
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


KEY_COLS = ["team", "map_mode", "command_mode", "stress", "mission_profile", "seed"]


def file_sha256(rel_path: str) -> str:
    return hashlib.sha256((ROOT / rel_path).read_bytes()).hexdigest().upper()


def row_key(row: dict[str, str | int]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in KEY_COLS)


def is_smoke_spec(row: dict[str, str | int]) -> bool:
    return row["map_mode"] == "delayed" and row["command_mode"] == "autonomous"


def main() -> None:
    reports = ROOT / "reports"
    data = ROOT / "data"
    out_dir = SRC_DIR / "theory_outputs"
    reports.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_path = reports / "s1_smoke_implementation_gates.json"
    gates = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gates.get("all_gates_passed"):
        raise RuntimeError("S1 smoke gates have not all passed")
    current_sha = {
        "src/swarm_pilot_experiment.py": file_sha256("src/swarm_pilot_experiment.py"),
        "scripts/lambda2_utils.py": file_sha256("scripts/lambda2_utils.py"),
    }
    if current_sha != gates["source_sha256"]:
        raise RuntimeError(f"M0 hash mismatch: {current_sha} != {gates['source_sha256']}")

    full_specs = sim.make_design(40, False, "s1_lambda2")
    smoke_keys = {row_key(row) for row in full_specs if is_smoke_spec(row)}
    remaining_specs = [row for row in full_specs if row_key(row) not in smoke_keys]
    remaining_specs = sorted(
        remaining_specs,
        key=lambda row: (
            row["team"],
            row["map_mode"],
            row["command_mode"],
            row["stress"],
            int(row["seed"]),
        ),
    )
    if len(smoke_keys) != 320:
        raise RuntimeError(f"expected 320 smoke keys, got {len(smoke_keys)}")
    if len(remaining_specs) != 3520:
        raise RuntimeError(f"expected 3520 remaining specs, got {len(remaining_specs)}")

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as executor:
        remaining_rows = list(executor.map(sim.simulate_from_spec, remaining_specs))
    wall = time.perf_counter() - start

    remaining = sim.pd.DataFrame(remaining_rows)
    remaining_csv = out_dir / "s1_lambda2_remaining_runs.csv"
    remaining.to_csv(remaining_csv, index=False, encoding="utf-8")

    smoke = sim.pd.read_csv(out_dir / "s1_lambda2_smoke_runs.csv")
    combined = sim.pd.concat([smoke, remaining], ignore_index=True, sort=False)
    duplicate_rows = int(combined.duplicated(subset=KEY_COLS).sum())
    expected_keys = {row_key(row) for row in full_specs}
    observed_keys = {
        tuple(str(row[column]) for column in KEY_COLS)
        for row in combined[KEY_COLS].to_dict(orient="records")
    }
    missing_keys = len(expected_keys - observed_keys)
    extra_keys = len(observed_keys - expected_keys)
    if len(combined) != 3840 or duplicate_rows or missing_keys or extra_keys:
        raise RuntimeError(
            {
                "rows": len(combined),
                "duplicate_rows": duplicate_rows,
                "missing_keys": missing_keys,
                "extra_keys": extra_keys,
            }
        )

    summary, aucs, validity = sim.summarize_results(combined)
    combined.to_csv(data / "s1_lambda2_runs.csv", index=False, encoding="utf-8")
    summary.to_csv(data / "s1_lambda2_condition_summary.csv", index=False, encoding="utf-8")
    aucs.to_csv(out_dir / "s1_lambda2_early_warning_auc.csv", index=False, encoding="utf-8")

    team_summary = (
        combined.groupby(["team", "stress"], dropna=False)
        .agg(
            n=("safe_delivery_success", "size"),
            safe_delivery_rate=("safe_delivery_success", "mean"),
            attrition_rate=("attrition_rate", "mean"),
            mean_effective_map_delay=("mean_effective_map_delay", "mean"),
            operational_score=("operational_score", "mean"),
            lambda2_ctrl_mean=("lambda2_ctrl_mean", "mean"),
            lambda2_per_alive_mean=("lambda2_per_alive_mean", "mean"),
            fragmentation_mean=("fragmentation_mean", "mean"),
            lambda2_degenerate_steps=("lambda2_degenerate_steps", "sum"),
            lambda2_low_population_steps=("lambda2_low_population_steps", "sum"),
        )
        .reset_index()
    )
    team_summary.to_csv(data / "s1_lambda2_team_summary.csv", index=False, encoding="utf-8")

    report = {
        "protocol": "S1_production_merge",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_rows_reused": int(len(smoke)),
        "remaining_rows_run": int(len(remaining)),
        "combined_rows": int(len(combined)),
        "expected_combined_rows": 3840,
        "duplicate_key_rows": duplicate_rows,
        "missing_expected_keys": missing_keys,
        "extra_keys": extra_keys,
        "remaining_wall_clock_s": wall,
        "remaining_wall_clock_s_per_run": wall / len(remaining),
        "source_sha256": current_sha,
        "validity": validity,
        "passed": len(combined) == 3840 and duplicate_rows == 0 and missing_keys == 0 and extra_keys == 0,
        "remaining_source_csv": str(remaining_csv),
        "combined_csv": str(data / "s1_lambda2_runs.csv"),
    }
    (reports / "s1_production_merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "remaining_rows": len(remaining),
                "combined_rows": len(combined),
                "duplicate_key_rows": duplicate_rows,
                "missing_expected_keys": missing_keys,
                "remaining_wall_clock_s": wall,
                "passed": report["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
