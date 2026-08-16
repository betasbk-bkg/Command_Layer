"""Recompute the v1.5 relay-frontier diagnostics from released CSV files.

This script regenerates the three public relay-frontier artifacts:

* reports/layer1_slope_cellpreserving.json
* reports/trackE_selection_frequency_final.json
* reports/freshness_return_cellpreserving_v15.json

The analysis is descriptive. The ratio R = g*beta/c compares a latency-linked
safe-delivery gain term with the observed allocation-level safe-delivery slope.
Because c is an observed net slope over the tested allocation sequence, R is
not a standalone net-benefit decision rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KMAP = {
    "no_relay_hetero": 0,
    "relay_sparse": 1,
    "balanced_hetero": 2,
    "relay_mid": 3,
    "relay_rich": 4,
}
BOOTSTRAP_SEED = 20260808
DEFAULT_B = 2000


def slope_xy(x_values: Any, y_values: Any) -> float:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x = x - x.mean()
    return float((x * y).sum() / (x * x).sum())


def round_float(value: float, ndigits: int = 6) -> float:
    return float(round(float(value), ndigits))


def load_relay(data: Path) -> pd.DataFrame:
    frame = pd.read_csv(data / "relay_sweep_runs.csv")
    frame["k"] = frame["team"].map(KMAP).astype(int)
    return frame


def load_beta_panel(data: Path) -> pd.DataFrame:
    q2 = pd.read_csv(data / "q2_runs.csv")
    grid = q2[q2["stress"].str.startswith("grid_m")].copy()
    grid["map_delay"] = grid["stress"].str.extract(r"grid_m(\d+)_c")[0].astype(int)
    return grid[
        (grid["team"] == "no_relay_hetero")
        & (grid["command_mode"] == "autonomous")
        & (grid["map_mode"] == "delayed")
    ].copy()


def grouped_bootstrap_slope(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    x_col: str,
    y_col: str,
    sign: float,
    rng: np.random.Generator,
    b: int,
) -> tuple[float, np.ndarray]:
    means = frame.groupby(group_cols, sort=True)[y_col].mean().reset_index()
    point = sign * slope_xy(means[x_col], means[y_col])
    groups = [(key, group[[x_col, y_col]]) for key, group in frame.groupby(group_cols, sort=True)]
    boot = np.empty(b, dtype=float)
    for idx in range(b):
        rows = []
        for key, group in groups:
            sample_index = rng.integers(0, len(group), len(group))
            if not isinstance(key, tuple):
                key = (key,)
            record = {column: value for column, value in zip(group_cols, key)}
            record[x_col] = float(group[x_col].iloc[0])
            record[y_col] = float(group[y_col].to_numpy(dtype=float)[sample_index].mean())
            rows.append(record)
        sampled_means = pd.DataFrame(rows)
        boot[idx] = sign * slope_xy(sampled_means[x_col], sampled_means[y_col])
    return point, boot


def compute_layer1(data: Path, *, b: int = DEFAULT_B) -> dict[str, Any]:
    relay = load_relay(data)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata: dict[str, Any] = {}

    for stress in ["degraded", "severe"]:
        stress_frame = relay[relay["stress"] == stress]
        definitions = [(f"{stress}|map=ALL|command=ALL", stress_frame, ["k", "map_mode", "command_mode"])]
        for map_mode in ["delayed", "scout_belief", "no_shared"]:
            definitions.append((f"{stress}|map={map_mode}", stress_frame[stress_frame["map_mode"] == map_mode], ["k", "command_mode"]))
        for command_mode in ["autonomous", "consensus_gated", "crowd_vector", "stale_hold"]:
            definitions.append(
                (
                    f"{stress}|command={command_mode}",
                    stress_frame[stress_frame["command_mode"] == command_mode],
                    ["k", "map_mode"],
                )
            )

        for key, frame, group_cols in definitions:
            point, boot = grouped_bootstrap_slope(
                frame,
                group_cols=group_cols,
                x_col="k",
                y_col="safe_delivery_success",
                sign=-1.0,
                rng=rng,
                b=b,
            )
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            strata[key] = {
                "c": round_float(point),
                "ci_low": round_float(ci_low),
                "ci_high": round_float(ci_high),
                "excludes_zero": bool(ci_low > 1e-12 or ci_high < -1e-12),
                "n_runs": int(len(frame)),
                "boot_frac_positive": round_float(float(np.mean(boot > 0.0)), 4),
            }

    command_exceptions = [
        key
        for key, value in strata.items()
        if "|command=" in key and "|map=" not in key and not value["excludes_zero"]
    ]
    map_keys = [key for key in strata if "|map=" in key and "|command=" not in key]
    command_keys = [key for key in strata if "|command=" in key and "|map=" not in key]
    return {
        "metric": "safe_delivery_success",
        "statistic": "c = -OLS slope vs relay count k in {0,1,2,3,4}",
        "scheme": "cell-preserving bootstrap; equal-weight aggregation of cell means; percentile 95% CI",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": b,
        "strata": strata,
        "summary": {
            "point_estimate_positive_all": bool(all(value["c"] > 0.0 for value in strata.values())),
            "map_strata_excluding_zero": f"{sum(strata[key]['excludes_zero'] for key in map_keys)}/{len(map_keys)}",
            "command_strata_excluding_zero": f"{sum(strata[key]['excludes_zero'] for key in command_keys)}/{len(command_keys)}",
            "command_exceptions": command_exceptions,
            "note": "Per-cell n is 800 for map strata and 600 for command strata; the command exceptions are marginal. Interval endpoints within 1e-12 of zero are treated as touching zero, so a stratum whose lower endpoint rounds to 0.0 is not counted as excluding zero.",
        },
    }


def selection_frequency_for_metric(
    frame: pd.DataFrame,
    *,
    metric: str,
    rule: str,
    rng: np.random.Generator,
    b: int,
) -> dict[str, Any]:
    groups = [
        (key, group[metric].to_numpy(dtype=float))
        for key, group in frame.groupby(["k", "map_mode", "command_mode"], sort=True)
    ]
    freq = np.zeros(5, dtype=float)
    for _ in range(b):
        values_by_k: dict[int, list[float]] = {k: [] for k in range(5)}
        for (k, _map_mode, _command_mode), values in groups:
            sample = values[rng.integers(0, len(values), len(values))]
            values_by_k[int(k)].append(float(sample.mean()))
        scores = np.asarray([np.mean(values_by_k[k]) for k in range(5)], dtype=float)
        best = scores.max() if rule == "max" else scores.min()
        winners = np.where(np.isclose(scores, best, rtol=0.0, atol=1e-12))[0]
        freq[winners] += 1.0 / len(winners)
    freq = freq / b
    return {
        "rule": rule,
        "freq": [round_float(value, 4) for value in freq],
        "k_ge2": round_float(float(freq[2:].sum()), 4),
    }


def compute_track_e(data: Path, *, b: int = DEFAULT_B) -> dict[str, Any]:
    relay = load_relay(data)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = {
        "mean_effective_map_delay": "min",
        "safe_delivery_success": "max",
        "attrition_rate": "min",
        "operational_score": "max",
        "degraded_outcome": "min",
    }
    results = {}
    for stress in ["degraded", "severe"]:
        frame = relay[relay["stress"] == stress]
        for metric, rule in metrics.items():
            results[f"{stress}|{metric}"] = selection_frequency_for_metric(
                frame, metric=metric, rule=rule, rng=rng, b=b
            )
    return {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": b,
        "scheme": "cell-preserving (k x map x command), equal-weight aggregation, ties split evenly",
        "results": results,
    }


def compute_freshness(data: Path, *, b: int = DEFAULT_B) -> dict[str, Any]:
    relay = load_relay(data)
    beta_panel = load_beta_panel(data)
    beta_point = -slope_xy(beta_panel["map_delay"], beta_panel["safe_delivery_success"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    beta_groups = [
        (int(group["map_delay"].iloc[0]), group["safe_delivery_success"].to_numpy(dtype=float))
        for _key, group in beta_panel.groupby("stress", sort=True)
    ]
    stress_specs: dict[str, dict[str, Any]] = {}
    for stress in ["degraded", "severe"]:
        stress_frame = relay[relay["stress"] == stress]
        stress_means = (
            stress_frame.groupby(["k", "map_mode", "command_mode"], sort=True)[
                ["mean_effective_map_delay", "safe_delivery_success"]
            ]
            .mean()
            .reset_index()
        )
        g_point = -slope_xy(stress_means["k"], stress_means["mean_effective_map_delay"])
        c_point = -slope_xy(stress_means["k"], stress_means["safe_delivery_success"])
        delay_groups = [
            (key, group["mean_effective_map_delay"].to_numpy(dtype=float))
            for key, group in stress_frame.groupby(["k", "map_mode", "command_mode"], sort=True)
        ]
        safe_groups = [
            (key, group["safe_delivery_success"].to_numpy(dtype=float))
            for key, group in stress_frame.groupby(["k", "map_mode", "command_mode"], sort=True)
        ]
        stress_specs[stress] = {
            "g_point": g_point,
            "c_point": c_point,
            "delay_groups": delay_groups,
            "safe_groups": safe_groups,
            "boot_r": [],
            "invalid": 0,
        }

    for _ in range(b):
        beta_rows = []
        for map_delay, values in beta_groups:
            sample = values[rng.integers(0, len(values), len(values))]
            beta_rows.append({"map_delay": map_delay, "safe_delivery_success": float(sample.mean())})
        beta_boot_frame = pd.DataFrame(beta_rows)
        beta = -slope_xy(beta_boot_frame["map_delay"], beta_boot_frame["safe_delivery_success"])

        for stress, spec in stress_specs.items():
            delay_rows = []
            for (k, _map_mode, _command_mode), values in spec["delay_groups"]:
                sample = values[rng.integers(0, len(values), len(values))]
                delay_rows.append({"k": int(k), "mean_effective_map_delay": float(sample.mean())})
            safe_rows = []
            for (k, _map_mode, _command_mode), values in spec["safe_groups"]:
                sample = values[rng.integers(0, len(values), len(values))]
                safe_rows.append({"k": int(k), "safe_delivery_success": float(sample.mean())})
            g = -slope_xy(pd.DataFrame(delay_rows)["k"], pd.DataFrame(delay_rows)["mean_effective_map_delay"])
            c = -slope_xy(pd.DataFrame(safe_rows)["k"], pd.DataFrame(safe_rows)["safe_delivery_success"])
            if c <= 0.0:
                spec["invalid"] += 1
            else:
                spec["boot_r"].append(g * beta / c)

    results = {}
    for stress, spec in stress_specs.items():
        g_point = float(spec["g_point"])
        c_point = float(spec["c_point"])
        boot = np.asarray(spec["boot_r"], dtype=float)
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        results[stress] = {
            "g": round_float(g_point),
            "c": round_float(c_point),
            "beta": round_float(beta_point, 7),
            "R": round_float(g_point * beta_point / c_point),
            "ci_low": round_float(ci_low),
            "ci_high": round_float(ci_high),
            "n_resamples_valid": int(len(boot)),
            "n_invalid_denominator": int(spec["invalid"]),
            "n_resamples_R_below_1": int(np.sum(boot < 1.0)),
            "beta_crit_over_beta": round_float(1.0 + c_point / (g_point * beta_point), 4),
        }
    return {
        "statistic": "R = g*beta/c",
        "interpretation": "descriptive comparison of a latency-linked safe-delivery gain term with the observed allocation-level safe-delivery slope; not a standalone net-benefit decision rule",
        "scheme": "cell-preserving bootstrap over (k x map x command) for g,c; stratified by delay-grid cell for beta; equal-weight aggregation; percentile 95% CI",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "B": b,
        "results": results,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate v1.5 relay-frontier analysis artifacts.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--b", type=int, default=DEFAULT_B)
    parser.add_argument(
        "--only",
        choices=["all", "layer1", "trackE", "freshness"],
        default="all",
    )
    args = parser.parse_args()

    outputs = {}
    if args.only in {"all", "layer1"}:
        outputs["layer1"] = compute_layer1(args.data, b=args.b)
        write_json(args.reports / "layer1_slope_cellpreserving.json", outputs["layer1"])
    if args.only in {"all", "trackE"}:
        outputs["trackE"] = compute_track_e(args.data, b=args.b)
        write_json(args.reports / "trackE_selection_frequency_final.json", outputs["trackE"])
    if args.only in {"all", "freshness"}:
        outputs["freshness"] = compute_freshness(args.data, b=args.b)
        write_json(args.reports / "freshness_return_cellpreserving_v15.json", outputs["freshness"])
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
