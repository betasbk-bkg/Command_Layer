from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260808
DEFAULT_B = 20000
TEAM_TO_K = {
    "no_relay_hetero": 0,
    "relay_sparse": 1,
    "balanced_hetero": 2,
    "relay_mid": 3,
    "relay_rich": 4,
}
ENDPOINTS = {
    "safe_delivery_success": "negative",
    "degraded_outcome": "positive",
    "attrition_rate": "positive",
    "mean_effective_map_delay": "negative",
}


def round_float(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def one_sided_p_from_bootstrap(boot: np.ndarray, direction: str) -> float:
    if direction == "negative":
        return float((np.sum(boot >= 0.0) + 1) / (len(boot) + 1))
    if direction == "positive":
        return float((np.sum(boot <= 0.0) + 1) / (len(boot) + 1))
    raise ValueError(direction)


def bootstrap_team_contrast(
    frame: pd.DataFrame,
    *,
    stress: str,
    metric: str,
    cell_cols: list[str],
    b: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float], float]:
    sub = frame[(frame["stress"] == stress) & frame["team"].isin(["no_relay_hetero", "relay_rich"])].copy()
    point = float(
        sub.loc[sub["team"] == "relay_rich", metric].astype(float).mean()
        - sub.loc[sub["team"] == "no_relay_hetero", metric].astype(float).mean()
    )
    grouped = [
        (str(team), group[metric].to_numpy(dtype=float))
        for (team, *cell), group in sub.groupby(["team"] + cell_cols, sort=True)
    ]
    boot = np.empty(b, dtype=float)
    for idx in range(b):
        sampled: dict[str, list[float]] = {"no_relay_hetero": [], "relay_rich": []}
        for team, values in grouped:
            sample_index = rng.integers(0, len(values), len(values))
            sampled[team].append(float(values[sample_index].mean()))
        boot[idx] = float(np.mean(sampled["relay_rich"]) - np.mean(sampled["no_relay_hetero"]))
    ci = np.quantile(boot, [0.025, 0.975])
    p = one_sided_p_from_bootstrap(boot, ENDPOINTS[metric])
    return point, (float(ci[0]), float(ci[1])), p


def chi_square_survival_even_df6(x: float) -> float:
    # Fisher's method combines three p-values, so the statistic has chi-square df=6.
    y = x / 2.0
    return float(math.exp(-y) * (1.0 + y + (y * y) / 2.0))


def fisher_p(p_values: list[float]) -> float:
    statistic = -2.0 * sum(math.log(max(p, 1e-300)) for p in p_values)
    return chi_square_survival_even_df6(statistic)


def stouffer_p(p_values: list[float]) -> float:
    normal = NormalDist()
    z_values = [normal.inv_cdf(1.0 - min(max(p, 1e-300), 1.0 - 1e-16)) for p in p_values]
    z = sum(z_values) / math.sqrt(len(z_values))
    return float(1.0 - normal.cdf(z))


def load_layer(root: Path, layer: str) -> tuple[pd.DataFrame, list[str]]:
    if layer == "continuous":
        frame = pd.read_csv(root / "data" / "relay_sweep_runs.csv")
        if "relay_count" not in frame.columns:
            frame["relay_count"] = frame["team"].map(TEAM_TO_K)
        return frame, ["map_mode", "command_mode"]
    if layer == "grid_world":
        return pd.read_csv(root / "data" / "robotics_grid_runs.csv"), ["command_mode"]
    if layer == "webots":
        return pd.read_csv(root / "data" / "webots_runs.csv"), ["command_mode"]
    raise ValueError(layer)


def webots_permutation_p(root: Path, stress: str, metric: str, delta: float) -> float:
    frame = pd.read_csv(root / "data" / "webots_permutation.csv")
    row = frame[
        (frame["stress"] == stress)
        & (frame["contrast_type"] == "team")
        & (frame["contrast"] == "relay_rich_minus_no_relay_hetero")
        & (frame["metric"] == metric)
    ]
    if row.empty:
        raise KeyError(f"missing Webots permutation row for {stress}/{metric}")
    two_sided = float(row.iloc[0]["p_permutation_two_sided"])
    direction = ENDPOINTS[metric]
    direction_ok = (direction == "negative" and delta < 0.0) or (direction == "positive" and delta > 0.0)
    return float(two_sided / 2.0 if direction_ok else 1.0 - two_sided / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate v1.6 cross-layer direction consistency analysis.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--b", type=int, default=DEFAULT_B)
    args = parser.parse_args()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    for layer in ["continuous", "grid_world", "webots"]:
        frame, cell_cols = load_layer(args.root, layer)
        for stress in ["degraded", "severe"]:
            for metric, direction in ENDPOINTS.items():
                delta, ci, p_boot = bootstrap_team_contrast(
                    frame,
                    stress=stress,
                    metric=metric,
                    cell_cols=cell_cols,
                    b=args.b,
                    rng=rng,
                )
                p_one_sided = webots_permutation_p(args.root, stress, metric, delta) if layer == "webots" else p_boot
                rows.append(
                    {
                        "layer": layer,
                        "stress": stress,
                        "contrast": "relay_rich_minus_no_relay_hetero",
                        "metric": metric,
                        "expected_direction": direction,
                        "delta": round_float(delta),
                        "ci95_low": round_float(ci[0]),
                        "ci95_high": round_float(ci[1]),
                        "one_sided_p": round_float(p_one_sided, 8),
                        "p_source": "permutation" if layer == "webots" else "cell-preserving bootstrap",
                        "n_left": int(len(frame[(frame["stress"] == stress) & (frame["team"] == "relay_rich")])),
                        "n_right": int(len(frame[(frame["stress"] == stress) & (frame["team"] == "no_relay_hetero")])),
                        "sign_consistent": bool(
                            (direction == "negative" and delta < 0.0) or (direction == "positive" and delta > 0.0)
                        ),
                    }
                )

    safe_rows = [row for row in rows if row["metric"] == "safe_delivery_success"]
    combined: dict[str, Any] = {}
    for stress in ["degraded", "severe"]:
        p_values = [float(row["one_sided_p"]) for row in safe_rows if row["stress"] == stress]
        combined[stress] = {
            "safe_delivery_one_sided_p_values": p_values,
            "fisher_p": round_float(fisher_p(p_values), 10),
            "stouffer_p": round_float(stouffer_p(p_values), 10),
        }

    report = {
        "protocol": "direction_consistency_v16",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": int(args.b),
        "interpretation": (
            "Direction consistency only. Effect magnitudes are not treated as directly comparable "
            "across the continuous, grid-world, and Webots layers."
        ),
        "rows": rows,
        "summary": {
            "sign_consistent_cells": int(sum(bool(row["sign_consistent"]) for row in rows)),
            "total_cells": int(len(rows)),
            "safe_delivery_combined_tests": combined,
        },
    }
    write_json(args.reports / "direction_consistency.json", report)
    pd.DataFrame(rows).to_csv(args.reports / "direction_consistency.csv", index=False)


if __name__ == "__main__":
    main()
