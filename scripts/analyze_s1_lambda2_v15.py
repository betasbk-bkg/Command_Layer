from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
BOOTSTRAP_SEED = 20260808
B = 2000
KEY_COLS = ["team", "map_mode", "command_mode", "stress", "mission_profile", "seed"]
CELL_COLS = ["map_mode", "command_mode"]
TEAM_TO_K = {
    "no_relay_hetero": 0,
    "relay_sparse": 1,
    "balanced_hetero": 2,
    "relay_mid": 3,
    "relay_rich": 4,
}
K_TO_TEAM = {value: key for key, value in TEAM_TO_K.items()}
METRIC_DIRECTIONS = {
    "safe_delivery_success": 1,
    "attrition_rate": -1,
    "mean_effective_map_delay": -1,
    "operational_score": 1,
}


def paired_cell_bootstrap(
    paired: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    b: int = B,
) -> dict[str, float | int]:
    diff = paired[f"{metric}_s1"].to_numpy(dtype=float) - paired[f"{metric}_base"].to_numpy(dtype=float)
    paired = paired.copy()
    paired["_diff"] = diff
    cells = [cell for _, cell in paired.groupby(CELL_COLS, dropna=False)]
    boot = np.empty(b, dtype=float)
    for i in range(b):
        cell_means = []
        for cell in cells:
            vals = cell["_diff"].to_numpy(dtype=float)
            sample = rng.choice(vals, size=len(vals), replace=True)
            cell_means.append(float(np.mean(sample)))
        boot[i] = float(np.mean(cell_means))
    direction = METRIC_DIRECTIONS[metric]
    return {
        "n_pairs": int(len(paired)),
        "n_cells": int(len(cells)),
        "mean_diff": float(np.mean(diff)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "improvement_rate": float(np.mean(direction * diff > 0)),
        "positive_diff_rate": float(np.mean(diff > 0)),
        "boot_frac_positive": float(np.mean(boot > 0)),
        "boot_frac_improvement": float(np.mean(direction * boot > 0)),
    }


def independent_cell_bootstrap(
    s1: pd.DataFrame,
    base: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    b: int = B,
) -> dict[str, float | int]:
    cell_keys = sorted(set(map(tuple, s1[CELL_COLS].itertuples(index=False, name=None))))
    boot = np.empty(b, dtype=float)
    for i in range(b):
        cell_diffs = []
        for key in cell_keys:
            s1_vals = s1[(s1[CELL_COLS[0]] == key[0]) & (s1[CELL_COLS[1]] == key[1])][metric].to_numpy(dtype=float)
            base_vals = base[(base[CELL_COLS[0]] == key[0]) & (base[CELL_COLS[1]] == key[1])][metric].to_numpy(dtype=float)
            s1_sample = rng.choice(s1_vals, size=len(s1_vals), replace=True)
            base_sample = rng.choice(base_vals, size=len(base_vals), replace=True)
            cell_diffs.append(float(np.mean(s1_sample) - np.mean(base_sample)))
        boot[i] = float(np.mean(cell_diffs))
    observed = []
    for key in cell_keys:
        s1_vals = s1[(s1[CELL_COLS[0]] == key[0]) & (s1[CELL_COLS[1]] == key[1])][metric].to_numpy(dtype=float)
        base_vals = base[(base[CELL_COLS[0]] == key[0]) & (base[CELL_COLS[1]] == key[1])][metric].to_numpy(dtype=float)
        observed.append(float(np.mean(s1_vals) - np.mean(base_vals)))
    direction = METRIC_DIRECTIONS[metric]
    return {
        "n_s1": int(len(s1)),
        "n_base": int(len(base)),
        "n_cells": int(len(cell_keys)),
        "mean_diff": float(np.mean(observed)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "boot_frac_positive": float(np.mean(boot > 0)),
        "boot_frac_improvement": float(np.mean(direction * boot > 0)),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(DATA / "relay_sweep_reinstrumented_v15.csv")
    s1 = pd.read_csv(DATA / "s1_lambda2_runs.csv")
    baseline["k"] = baseline["team"].map(TEAM_TO_K)
    s1["k"] = s1["team"].map(TEAM_TO_K)

    paired = s1.merge(baseline, on=KEY_COLS, suffixes=("_s1", "_base"), how="left", indicator=True)
    pairing_missing = int((paired["_merge"] != "both").sum())
    if pairing_missing:
        raise RuntimeError(f"paired join missing rows: {pairing_missing}")

    rows: list[dict[str, float | int | str]] = []
    summary: dict[str, object] = {
        "protocol": "S1_lambda2_cell_preserving_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": B,
        "scheme": "cell-preserving paired bootstrap over map_mode x command_mode for S1-vs-baseline within the same team/stress/seed; independent cell-preserving bootstrap for k=4 S1 vs k=0 baseline rescue tests",
        "s1_rows": int(len(s1)),
        "baseline_rows": int(len(baseline)),
        "paired_rows": int(len(paired)),
        "pairing_missing": pairing_missing,
        "excluded_metrics": ["degraded_outcome"],
        "branch_by_stress": {},
        "d_test": {},
        "k4_primary": {},
        "diagnostics": {},
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    for stress in ["degraded", "severe"]:
        for k in [1, 2, 3, 4]:
            team = K_TO_TEAM[k]
            frame = paired[(paired["stress"] == stress) & (paired["team"] == team)].copy()
            for metric in METRIC_DIRECTIONS:
                stats = paired_cell_bootstrap(frame, metric, rng)
                rows.append(
                    {
                        "level": "stress_k",
                        "k": k,
                        "team": team,
                        "stress": stress,
                        "command_mode": "ALL",
                        "metric": metric,
                        **stats,
                    }
                )
                if k == 4 and metric == "safe_delivery_success":
                    summary["k4_primary"][stress] = stats

        s1_k4 = s1[(s1["stress"] == stress) & (s1["team"] == K_TO_TEAM[4])].copy()
        base_k0 = baseline[(baseline["stress"] == stress) & (baseline["team"] == K_TO_TEAM[0])].copy()
        safe_rescue = independent_cell_bootstrap(s1_k4, base_k0, "safe_delivery_success", rng)
        delay_rescue = independent_cell_bootstrap(s1_k4, base_k0, "mean_effective_map_delay", rng)
        pareto = bool(safe_rescue["ci_low"] > 0.0 and delay_rescue["ci_high"] < 0.0)
        summary["d_test"][stress] = {
            "safe_s1_k4_minus_baseline_k0": safe_rescue,
            "delay_s1_k4_minus_baseline_k0": delay_rescue,
            "pareto": pareto,
        }
        k4_safe = summary["k4_primary"][stress]
        if pareto:
            branch = "D"
        elif k4_safe["ci_high"] < 0.0:
            branch = "A"
        elif k4_safe["ci_low"] <= 0.0 <= k4_safe["ci_high"]:
            branch = "B"
        elif k4_safe["ci_low"] > 0.0:
            branch = "C"
        else:
            raise RuntimeError(f"unclassifiable branch for {stress}: {k4_safe}")
        summary["branch_by_stress"][stress] = branch

        for metric in ["lambda2_ctrl_mean", "lambda2_per_alive_mean", "fragmentation_mean"]:
            frame = paired[(paired["stress"] == stress) & (paired["team"] == K_TO_TEAM[4])].copy()
            diff = frame[f"{metric}_s1"].to_numpy(dtype=float) - frame[f"{metric}_base"].to_numpy(dtype=float)
            summary["diagnostics"][f"{stress}_k4_{metric}"] = {
                "mean_s1": float(frame[f"{metric}_s1"].mean()),
                "mean_baseline": float(frame[f"{metric}_base"].mean()),
                "mean_diff_s1_minus_baseline": float(np.mean(diff)),
            }

    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(REPORTS / "s1_lambda2_contrasts.csv", index=False, encoding="utf-8")
    (REPORTS / "s1_lambda2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "s1_rows": summary["s1_rows"],
                "paired_rows": summary["paired_rows"],
                "pairing_missing": summary["pairing_missing"],
                "branch_by_stress": summary["branch_by_stress"],
                "d_test_pareto": {
                    stress: summary["d_test"][stress]["pareto"] for stress in ["degraded", "severe"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
