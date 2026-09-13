from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def round_float(value: float, digits: int = 6) -> float:
    return float(round(float(value), digits))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate v1.6 beta-threshold mission sensitivity analysis.")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()

    freshness_path = args.reports / "freshness_return_cellpreserving_v15.json"
    freshness = json.loads(freshness_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for stress, values in freshness["results"].items():
        r_point = float(values["R"])
        r_low = float(values["ci_low"])
        r_high = float(values["ci_high"])
        multiplier = 1.0 / r_point
        rows.append(
            {
                "stress": stress,
                "g": round_float(values["g"]),
                "c": round_float(values["c"]),
                "beta": round_float(values["beta"], 7),
                "R": round_float(r_point),
                "R_ci95_low": round_float(r_low),
                "R_ci95_high": round_float(r_high),
                "beta_crit_over_beta": round_float(multiplier, 4),
                "beta_crit_over_beta_ci95_low": round_float(1.0 / r_high, 4),
                "beta_crit_over_beta_ci95_high": round_float(1.0 / r_low, 4),
            }
        )

    report = {
        "protocol": "mission_sensitivity_v16",
        "definition": (
            "beta_crit_over_beta is c/(g*beta) = 1/R, the multiplicative increase "
            "in the safe-delivery value of one unit of effective-map-delay reduction "
            "required for the latency-linked term g*beta to equal the observed "
            "allocation-level safe-delivery slope c."
        ),
        "interpretation": (
            "This is a descriptive mission-sensitivity threshold, not a standalone "
            "net-benefit decision rule."
        ),
        "rows": rows,
    }
    write_json(args.reports / "mission_sensitivity_beta_threshold_v16.json", report)
    pd.DataFrame(rows).to_csv(args.reports / "mission_sensitivity_beta_threshold_v16.csv", index=False)


if __name__ == "__main__":
    main()
