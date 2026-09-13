from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260808
DEFAULT_B = 2000
TEAM_TO_K = {
    "no_relay_hetero": 0,
    "relay_sparse": 1,
    "balanced_hetero": 2,
    "relay_mid": 3,
    "relay_rich": 4,
}
EXPOSURE_GRID = [0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22]
RECOVERY_GRID = [4, 6, 8, 10, 12, 14, 16]


def round_float(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_relay_runs(data_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / "relay_sweep_runs.csv")
    if "relay_count" not in frame.columns:
        frame["relay_count"] = frame["team"].map(TEAM_TO_K)
    missing = frame["relay_count"].isna()
    if missing.any():
        teams = sorted(frame.loc[missing, "team"].astype(str).unique())
        raise ValueError(f"cannot infer relay_count for teams: {teams}")
    frame["relay_count"] = frame["relay_count"].astype(int)
    reference = safe_delivery(frame, exposure_cut=0.150, recovery_cut=10)
    mismatches = int((reference != frame["safe_delivery_success"].astype(int)).sum())
    if mismatches:
        raise ValueError(
            "stored safe_delivery_success does not match the integer-loss "
            f"definition at the reference thresholds: {mismatches} mismatches"
        )
    return frame


def safe_delivery(frame: pd.DataFrame, *, exposure_cut: float, recovery_cut: int) -> pd.Series:
    # Use alive_final == n_agents rather than attrition_rate < 1/12 to avoid
    # floating-point ambiguity at exactly one lost robot.
    return (
        (frame["strict_success"].astype(int) == 1)
        & (frame["alive_final"].astype(float) == frame["n_agents"].astype(float))
        & (frame["exposure_per_agent_step"].astype(float) <= exposure_cut)
        & (frame["recovery_time"].astype(float) <= recovery_cut)
    ).astype(int)


def selection_frequency(
    frame: pd.DataFrame,
    *,
    exposure_cut: float,
    recovery_cut: int,
    b: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = safe_delivery(frame, exposure_cut=exposure_cut, recovery_cut=recovery_cut)
    work = frame[["stress", "relay_count", "map_mode", "command_mode"]].copy()
    work["safe_delivery_thresholded"] = values.to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []

    for stress, stress_frame in work.groupby("stress", sort=True):
        cell_specs: list[tuple[int, int, float]] = []
        for (k, map_mode, command_mode), group in stress_frame.groupby(
            ["relay_count", "map_mode", "command_mode"], sort=True
        ):
            cell_values = group["safe_delivery_thresholded"].to_numpy(dtype=int)
            cell_specs.append((int(k), int(len(cell_values)), float(cell_values.mean())))

        k_boot = np.zeros((5, b), dtype=float)
        k_cell_counts = np.zeros(5, dtype=int)
        for k, n_rows, p_hat in cell_specs:
            k_boot[k] += rng.binomial(n_rows, p_hat, size=b) / n_rows
            k_cell_counts[k] += 1
        k_boot = k_boot / k_cell_counts[:, None]
        maxima = k_boot.max(axis=0)
        winners = np.isclose(k_boot, maxima, rtol=0.0, atol=1e-12)
        winner_weights = winners / winners.sum(axis=0)
        freq = winner_weights.mean(axis=1)

        point_means = (
            work.loc[work["stress"] == stress]
            .assign(safe_delivery_thresholded=values.loc[work["stress"] == stress].to_numpy(dtype=int))
            .groupby(["relay_count", "map_mode", "command_mode"], sort=True)["safe_delivery_thresholded"]
            .mean()
            .groupby(level=0)
            .mean()
        )
        rows.append(
            {
                "stress": str(stress),
                "exposure_cut": round_float(exposure_cut, 3),
                "recovery_cut": int(recovery_cut),
                "point_means_by_k": [round_float(point_means.loc[k], 6) for k in range(5)],
                "selection_frequency_by_k": [round_float(x, 4) for x in freq],
                "weight_k_le_1": round_float(float(freq[0] + freq[1]), 4),
                "weight_k_ge_2": round_float(float(freq[2:].sum()), 4),
                "weight_k_ge_3": round_float(float(freq[3:].sum()), 4),
                "weight_k2": round_float(float(freq[2]), 4),
            }
        )
    return {"rows": rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stress in ["degraded", "severe"]:
        stress_rows = [row for row in rows if row["stress"] == stress]
        reference = [
            row
            for row in stress_rows
            if row["exposure_cut"] == 0.15 and row["recovery_cut"] == 10
        ][0]
        plausible = [
            row
            for row in stress_rows
            if 0.10 <= row["exposure_cut"] <= 0.20 and 10 <= row["recovery_cut"] <= 16
        ]
        boundary = [row for row in stress_rows if row["recovery_cut"] <= 8]

        def min_row(metric: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
            return min(candidates, key=lambda row: float(row[metric]))

        def max_row(metric: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
            return max(candidates, key=lambda row: float(row[metric]))

        out[stress] = {
            "reference_E_0p150_Trec_10": reference,
            "plausible_region": {
                "definition": "E in [0.10,0.20] and recovery cut in [10,16]",
                "min_weight_k_le_1": min_row("weight_k_le_1", plausible),
            },
            "entire_grid": {
                "definition": "8 exposure thresholds x 7 recovery thresholds",
                "min_weight_k_le_1": min_row("weight_k_le_1", stress_rows),
                "max_weight_k_ge_3": max_row("weight_k_ge_3", stress_rows),
            },
            "boundary_region_Trec_le_8": {
                "max_weight_k2": max_row("weight_k2", boundary),
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate v1.6 safe-delivery threshold sensitivity analysis."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--b", type=int, default=DEFAULT_B)
    args = parser.parse_args()

    frame = load_relay_runs(args.data)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for exposure_cut in EXPOSURE_GRID:
        for recovery_cut in RECOVERY_GRID:
            rows.extend(
                selection_frequency(
                    frame,
                    exposure_cut=exposure_cut,
                    recovery_cut=recovery_cut,
                    b=args.b,
                    rng=rng,
                )["rows"]
            )

    report = {
        "protocol": "threshold_sensitivity_v16",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": int(args.b),
        "safe_delivery_definition": (
            "strict_success == 1, alive_final == n_agents, exposure_per_agent_step <= E, "
            "and recovery_time <= T_rec"
        ),
        "exposure_grid": EXPOSURE_GRID,
        "recovery_grid": RECOVERY_GRID,
        "rows": rows,
        "summary": summarize(rows),
    }
    write_json(args.reports / "threshold_sensitivity_grid.json", report)
    pd.DataFrame(rows).to_csv(args.reports / "threshold_sensitivity_grid.csv", index=False)


if __name__ == "__main__":
    main()
