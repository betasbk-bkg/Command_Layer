from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"


def permutation_pvalue(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n_perm: int = 20000) -> tuple[float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    observed = float(left.mean() - right.mean())
    pooled = np.concatenate([left, right])
    n_left = len(left)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        delta = float(perm[:n_left].mean() - perm[n_left:].mean())
        if abs(delta) >= abs(observed) - 1e-12:
            count += 1
    return observed, float((count + 1) / (n_perm + 1))


def bh_adjust(pvalues: list[float]) -> list[float]:
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        true_rank = n - rank + 1
        value = min(running, pvalues[idx] * n / true_rank)
        adjusted[idx] = value
        running = value
    return adjusted.tolist()


def webots_permutation_tests() -> pd.DataFrame:
    runs = pd.read_csv(DATA / "webots_runs.csv")
    rng = np.random.default_rng(20260713)
    rows: list[dict[str, Any]] = []
    tests = [
        ("relay_rich", "no_relay_hetero", "safe_delivery_success"),
        ("relay_rich", "no_relay_hetero", "degraded_outcome"),
        ("relay_rich", "no_relay_hetero", "attrition_rate"),
        ("relay_rich", "no_relay_hetero", "payload_delivery_rate"),
        ("relay_rich", "no_relay_hetero", "mean_effective_map_delay"),
        ("crowd_vector", "autonomous", "safe_delivery_success"),
        ("consensus_gated", "autonomous", "operational_score"),
    ]
    for stress in ["degraded", "severe"]:
        stress_runs = runs[runs["stress"] == stress]
        for left_name, right_name, metric in tests:
            if left_name in set(stress_runs["team"]):
                left = stress_runs[stress_runs["team"] == left_name][metric].to_numpy()
                right = stress_runs[stress_runs["team"] == right_name][metric].to_numpy()
                contrast_type = "team"
            else:
                left = stress_runs[stress_runs["command_mode"] == left_name][metric].to_numpy()
                right = stress_runs[stress_runs["command_mode"] == right_name][metric].to_numpy()
                contrast_type = "command"
            if len(left) == 0 or len(right) == 0:
                continue
            delta, pvalue = permutation_pvalue(left, right, rng)
            rows.append(
                {
                    "layer": "webots",
                    "stress": stress,
                    "contrast_type": contrast_type,
                    "contrast": f"{left_name}_minus_{right_name}",
                    "metric": metric,
                    "delta": delta,
                    "p_permutation_two_sided": pvalue,
                    "n_left": len(left),
                    "n_right": len(right),
                }
            )
    frame = pd.DataFrame(rows)
    frame["q_bh_within_webots_family"] = bh_adjust(frame["p_permutation_two_sided"].tolist())
    frame.to_csv(DATA / "webots_permutation.csv", index=False, encoding="utf-8")
    return frame


def matched_sign_stability() -> pd.DataFrame:
    runs = pd.read_csv(DATA / "matched_composition_runs.csv")
    rows: list[dict[str, Any]] = []
    metrics = ["safe_delivery_success", "degraded_outcome", "attrition_rate", "mean_effective_map_delay"]
    expected_sign = {
        "safe_delivery_success": -1,
        "degraded_outcome": 1,
        "attrition_rate": 1,
        "mean_effective_map_delay": -1,
    }
    variants = [
        "relay_rich",
        "relay_rich_standardized",
        "relay_rich_scout_matched",
        "relay_rich_scout_matched_standardized",
    ]
    # Deterministic pseudo-folds based on seed order within each cell.
    for stress in ["degraded", "severe"]:
        for map_mode in ["delayed", "scout_belief"]:
            for command_mode in ["autonomous", "crowd_vector"]:
                base_cell = runs[
                    (runs["display_team"] == "no_relay_hetero")
                    & (runs["stress"] == stress)
                    & (runs["map_mode"] == map_mode)
                    & (runs["command_mode"] == command_mode)
                ].sort_values("seed")
                for variant in variants:
                    left_cell = runs[
                        (runs["display_team"] == variant)
                        & (runs["stress"] == stress)
                        & (runs["map_mode"] == map_mode)
                        & (runs["command_mode"] == command_mode)
                    ].sort_values("seed")
                    for fold in range(4):
                        left = left_cell.iloc[fold::4]
                        right = base_cell.iloc[fold::4]
                        for metric in metrics:
                            delta = float(left[metric].mean() - right[metric].mean())
                            rows.append(
                                {
                                    "stress": stress,
                                    "map_mode": map_mode,
                                    "command_mode": command_mode,
                                    "variant": variant,
                                    "fold": fold,
                                    "metric": metric,
                                    "delta": delta,
                                    "expected_direction": expected_sign[metric],
                                    "direction_matches": np.sign(delta) == expected_sign[metric] if abs(delta) > 1e-12 else False,
                                }
                            )
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA / "matched_composition_sign_stability.csv", index=False, encoding="utf-8")
    return frame


def write_report(webots: pd.DataFrame, stability: pd.DataFrame) -> None:
    webots_relay = webots[webots["contrast"].str.startswith("relay_rich_minus_no_relay")].copy()
    stability_summary = (
        stability.groupby(["stress", "metric"], as_index=False)
        .agg(folds=("fold", "count"), direction_match_rate=("direction_matches", "mean"), mean_delta=("delta", "mean"))
        .sort_values(["stress", "metric"])
    )
    lines = [
        "# Statistical Stress Tests",
        "",
        "## Webots Permutation Tests",
        "",
        "Permutation tests use the existing 192 Webots trials and avoid normality assumptions. They are intended as small-sample safeguards, not as a replacement for larger Webots reruns.",
        "",
        webots.sort_values(["stress", "contrast", "metric"]).to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Webots Relay-Only Readout",
        "",
        webots_relay.sort_values(["stress", "metric"]).to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Matched-Composition Sign Stability",
        "",
        "The table reports four deterministic seed-fold checks for matched-composition relay variants against no-relay.",
        "",
        stability_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- Webots relay safe-delivery, degraded-outcome, attrition, and delay effects should survive small-sample scrutiny if permutation p-values are low.",
        "- Command-layer Webots effects should remain secondary if permutation evidence is weaker.",
        "- Matched-composition sign stability is a check against the possibility that one lucky seed block produced the relay result.",
    ]
    (REPORTS / "statistical_stress_tests.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "webots_tests": int(len(webots)),
        "webots_relay_tests": int(len(webots_relay)),
        "matched_stability_rows": int(len(stability)),
        "matched_direction_match_rate": float(stability["direction_matches"].mean()),
    }
    (REPORTS / "statistical_stress_tests.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    webots = webots_permutation_tests()
    stability = matched_sign_stability()
    write_report(webots, stability)
    print(json.dumps({"webots_tests": len(webots), "stability_rows": len(stability)}))


if __name__ == "__main__":
    main()
