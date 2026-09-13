from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


ALPHA = 0.05
TARGET_POWER = 0.80


def round_float(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def power_for_drop(p_control: float, drop: float, n_per_group: int, alternative: str) -> float:
    p_treatment = max(0.0, min(1.0, p_control - drop))
    effect_size = abs(proportion_effectsize(p_control, p_treatment))
    return float(
        NormalIndPower().power(
            effect_size=effect_size,
            nobs1=n_per_group,
            alpha=ALPHA,
            ratio=1.0,
            alternative=alternative,
        )
    )


def mde_for_power(p_control: float, n_per_group: int, alternative: str) -> float:
    grid = np.linspace(0.0, 1.0, 20001)
    powers = np.array([power_for_drop(p_control, float(drop), n_per_group, alternative) for drop in grid])
    eligible = np.flatnonzero(powers >= TARGET_POWER)
    if len(eligible) == 0:
        return float("nan")
    return float(grid[int(eligible[0])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate v1.6 Webots safe-delivery power/MDE analysis.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    frame = pd.read_csv(args.data / "webots_runs.csv")
    rows: list[dict[str, Any]] = []
    for stress in ["degraded", "severe"]:
        sub = frame[(frame["stress"] == stress) & frame["team"].isin(["no_relay_hetero", "relay_rich"])]
        control = sub[sub["team"] == "no_relay_hetero"]["safe_delivery_success"].astype(float)
        treatment = sub[sub["team"] == "relay_rich"]["safe_delivery_success"].astype(float)
        if len(control) != len(treatment):
            raise ValueError(f"unequal Webots group sizes for {stress}: {len(control)} vs {len(treatment)}")
        n_per_group = int(len(control))
        p_control = float(control.mean())
        p_treatment = float(treatment.mean())
        observed_drop = float(p_control - p_treatment)
        rows.append(
            {
                "stress": stress,
                "metric": "safe_delivery_success",
                "contrast": "relay_rich_minus_no_relay_hetero",
                "n_per_group": n_per_group,
                "control_safe_delivery": round_float(p_control),
                "relay_rich_safe_delivery": round_float(p_treatment),
                "observed_delta_relay_minus_control": round_float(p_treatment - p_control),
                "observed_drop_control_minus_relay": round_float(observed_drop),
                "two_sided_alpha": ALPHA,
                "target_power": TARGET_POWER,
                "mde_two_sided": round_float(mde_for_power(p_control, n_per_group, "two-sided"), 4),
                "posthoc_power_two_sided": round_float(
                    power_for_drop(p_control, observed_drop, n_per_group, "two-sided"), 4
                ),
                "mde_one_sided": round_float(mde_for_power(p_control, n_per_group, "larger"), 4),
                "posthoc_power_one_sided": round_float(
                    power_for_drop(p_control, observed_drop, n_per_group, "larger"), 4
                ),
            }
        )

    report = {
        "protocol": "webots_power_mde_v16",
        "method": (
            "Normal approximation two-sample proportion power for safe-delivery contrasts. "
            "The two-sided calculation is the conservative primary readout; one-sided values "
            "are reported only because the manuscript hypothesis specifies the adverse direction."
        ),
        "rows": rows,
    }
    write_json(args.reports / "webots_power_mde.json", report)
    pd.DataFrame(rows).to_csv(args.reports / "webots_power_mde.csv", index=False)


if __name__ == "__main__":
    main()
