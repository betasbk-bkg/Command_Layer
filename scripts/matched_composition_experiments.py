from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % (2**32 - 1)


def bootstrap_delta(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n_boot: int = 1200) -> tuple[float, float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    deltas = np.empty(n_boot)
    for idx in range(n_boot):
        deltas[idx] = rng.choice(left, len(left), replace=True).mean() - rng.choice(right, len(right), replace=True).mean()
    return float(left.mean() - right.mean()), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def patch_base_for_variant(base: Any, variant: str) -> None:
    original_relay = base.RoleSpec(speed=0.015, sensor=0.115, avoid_gain=0.92, survival=1.08, target_gain=0.90)
    base.ROLES["relay"] = original_relay
    base.TEAM_COMPOSITIONS["relay_rich_scout_matched"] = (
        ["standard"] * 3 + ["scout"] * 3 + ["relay"] * 4 + ["payload"] * 2
    )
    base.TEAM_COMPOSITIONS["relay_rich_scout_matched_standardized"] = (
        ["standard"] * 3 + ["scout"] * 3 + ["relay"] * 4 + ["payload"] * 2
    )
    if variant == "relay_rich_standardized":
        base.ROLES["relay"] = base.ROLES["standard"]
    elif variant == "relay_rich_scout_matched_standardized":
        base.ROLES["relay"] = base.ROLES["standard"]


def team_for_variant(variant: str) -> str:
    return {
        "no_relay_hetero": "no_relay_hetero",
        "balanced_hetero": "balanced_hetero",
        "relay_rich": "relay_rich",
        "relay_rich_standardized": "relay_rich",
        "relay_rich_scout_matched": "relay_rich_scout_matched",
        "relay_rich_scout_matched_standardized": "relay_rich_scout_matched_standardized",
    }[variant]


def simulate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    import swarm_pilot_experiment as base

    variant = str(spec["variant"])
    patch_base_for_variant(base, variant)
    record = base.simulate_run(
        seed=int(spec["seed"]),
        team=team_for_variant(variant),
        map_mode=str(spec["map_mode"]),
        command_mode=str(spec["command_mode"]),
        stress_name=str(spec["stress"]),
        mission_profile="full_delivery",
    )
    record["matched_variant"] = variant
    record["display_team"] = variant
    record["max_steps"] = 180
    record["safe_exposure_cut"] = 0.150
    record["safe_exposure_scale"] = 0.220
    return record


def add_safety_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    import swarm_pilot_experiment as base

    data = frame.copy()
    team_payload = {
        "no_relay_hetero": 2,
        "balanced_hetero": 2,
        "relay_rich": 2,
        "relay_rich_standardized": 2,
        "relay_rich_scout_matched": 2,
        "relay_rich_scout_matched_standardized": 2,
    }
    data["n_agents"] = 12
    data["n_payload"] = data["display_team"].map(team_payload).fillna(2)
    data["reached_fraction"] = data["reached_final"] / data["n_agents"]
    data["alive_fraction"] = data["alive_final"] / data["n_agents"]
    data["payload_fraction"] = (data["payload_reached"] / data["n_payload"]).clip(0, 1)
    data["time_efficiency"] = np.clip(1.0 - data["success_time"] / data["max_steps"], 0.0, 1.0)
    data["exposure_safety"] = np.clip(1.0 - data["exposure_per_agent_step"] / data["safe_exposure_scale"], 0.0, 1.0)
    data["recovery_safety"] = np.clip(1.0 - data["recovery_time"] / 20.0, 0.0, 1.0)
    data["strict_success"] = (
        (data["reached_fraction"] >= 0.75)
        & (data["alive_fraction"] >= 0.67)
        & (data["payload_fraction"] >= 0.50)
    ).astype(int)
    data["operational_score"] = 100.0 * (
        0.24 * data["reached_fraction"]
        + 0.18 * data["payload_fraction"]
        + 0.18 * data["alive_fraction"]
        + 0.16 * data["time_efficiency"]
        + 0.14 * data["exposure_safety"]
        + 0.10 * data["recovery_safety"]
    )
    data["safe_delivery_success"] = (
        (data["strict_success"] == 1)
        & (data["attrition_rate"] < (1.0 / 12.0))
        & (data["exposure_per_agent_step"] <= data["safe_exposure_cut"])
        & (data["recovery_time"] <= 10)
    ).astype(int)
    data["degraded_outcome"] = (
        (data["safe_delivery_success"] == 0)
        | (data["attrition_rate"] >= (1.0 / 12.0))
        | (data["recovery_time"] >= 10)
    ).astype(int)
    return data


def build_design(seeds_per_cell: int) -> list[dict[str, Any]]:
    variants = [
        "no_relay_hetero",
        "balanced_hetero",
        "relay_rich",
        "relay_rich_standardized",
        "relay_rich_scout_matched",
        "relay_rich_scout_matched_standardized",
    ]
    specs: list[dict[str, Any]] = []
    for variant in variants:
        for stress in ["degraded", "severe"]:
            for map_mode in ["delayed", "scout_belief"]:
                for command_mode in ["autonomous", "crowd_vector"]:
                    for rep in range(seeds_per_cell):
                        specs.append(
                            {
                                "variant": variant,
                                "stress": stress,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "seed": stable_seed("auxiliary", variant, stress, map_mode, command_mode, rep),
                            }
                        )
    return specs


def run_matched_composition(seeds_per_cell: int, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_design(seeds_per_cell)
    records: list[dict[str, Any]] = []
    chunk = max(1, len(specs) // max(1, workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, record in enumerate(executor.map(simulate_spec, specs, chunksize=chunk), start=1):
            records.append(record)
            if idx % 500 == 0:
                print(f"matched-composition completed {idx}/{len(specs)}", flush=True)
    runs = add_safety_endpoints(pd.DataFrame(records))
    runs.to_csv(DATA / "matched_composition_runs.csv", index=False, encoding="utf-8")

    summary = (
        runs.groupby(["display_team", "stress", "map_mode", "command_mode"], as_index=False)
        .agg(
            n=("seed", "size"),
            safe_delivery=("safe_delivery_success", "mean"),
            operational_score=("operational_score", "mean"),
            degraded_outcome=("degraded_outcome", "mean"),
            attrition_rate=("attrition_rate", "mean"),
            exposure=("exposure_per_agent_step", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
            payload_fraction=("payload_fraction", "mean"),
        )
        .sort_values(["stress", "map_mode", "command_mode", "display_team"])
    )
    summary.to_csv(DATA / "matched_composition_summary.csv", index=False, encoding="utf-8")
    return runs, summary


def write_matched_contrasts(runs: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260711)
    rows: list[dict[str, Any]] = []
    metrics = [
        "safe_delivery_success",
        "operational_score",
        "degraded_outcome",
        "attrition_rate",
        "exposure_per_agent_step",
        "mean_effective_map_delay",
        "payload_fraction",
    ]
    variants = [
        "balanced_hetero",
        "relay_rich",
        "relay_rich_standardized",
        "relay_rich_scout_matched",
        "relay_rich_scout_matched_standardized",
    ]
    for stress in ["degraded", "severe"]:
        for map_mode in ["delayed", "scout_belief"]:
            for command_mode in ["autonomous", "crowd_vector"]:
                base = runs[
                    (runs["display_team"] == "no_relay_hetero")
                    & (runs["stress"] == stress)
                    & (runs["map_mode"] == map_mode)
                    & (runs["command_mode"] == command_mode)
                ]
                for variant in variants:
                    left = runs[
                        (runs["display_team"] == variant)
                        & (runs["stress"] == stress)
                        & (runs["map_mode"] == map_mode)
                        & (runs["command_mode"] == command_mode)
                    ]
                    for metric in metrics:
                        delta, low, high = bootstrap_delta(left[metric].to_numpy(), base[metric].to_numpy(), rng)
                        rows.append(
                            {
                                "stress": stress,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "contrast": f"{variant}_minus_no_relay",
                                "metric": metric,
                                "delta": delta,
                                "ci95_low": low,
                                "ci95_high": high,
                                "n_left": len(left),
                                "n_right": len(base),
                            }
                        )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(DATA / "matched_composition_contrasts.csv", index=False, encoding="utf-8")
    return contrasts


@dataclass(frozen=True)
class ScoreWeights:
    name: str
    reached: float
    payload: float
    alive: float
    time: float
    exposure: float
    recovery: float


SCORE_WEIGHTS = [
    ScoreWeights("original", 0.24, 0.18, 0.18, 0.16, 0.14, 0.10),
    ScoreWeights("survival_heavy", 0.18, 0.16, 0.28, 0.10, 0.20, 0.08),
    ScoreWeights("payload_heavy", 0.18, 0.30, 0.16, 0.12, 0.14, 0.10),
    ScoreWeights("exposure_heavy", 0.16, 0.16, 0.18, 0.10, 0.30, 0.10),
    ScoreWeights("progress_heavy", 0.34, 0.18, 0.14, 0.16, 0.10, 0.08),
    ScoreWeights("time_heavy", 0.20, 0.16, 0.16, 0.28, 0.12, 0.08),
]


def alternative_score(frame: pd.DataFrame, weights: ScoreWeights) -> pd.Series:
    return 100.0 * (
        weights.reached * frame["reached_fraction"]
        + weights.payload * frame["payload_fraction"]
        + weights.alive * frame["alive_fraction"]
        + weights.time * frame["time_efficiency"]
        + weights.exposure * frame["exposure_safety"]
        + weights.recovery * frame["recovery_safety"]
    )


def write_score_sensitivity() -> pd.DataFrame:
    rng = np.random.default_rng(20260712)
    q2 = pd.read_csv(DATA / "q2_runs.csv")
    relay = pd.read_csv(DATA / "relay_sweep_runs.csv")
    frames = [("q2_confirmatory", q2[q2["stress_family"].isin(["degraded", "severe"])].copy()), ("relay_sweep", relay.copy())]
    rows: list[dict[str, Any]] = []
    for layer, frame in frames:
        for weights in SCORE_WEIGHTS:
            frame = frame.copy()
            frame["alt_score"] = alternative_score(frame, weights)
            for stress in ["degraded", "severe"]:
                subset = frame[frame["stress"] == stress]
                left = subset[subset["team"] == "relay_rich"]["alt_score"].to_numpy()
                right = subset[subset["team"] == "no_relay_hetero"]["alt_score"].to_numpy()
                if len(left) == 0 or len(right) == 0:
                    continue
                delta, low, high = bootstrap_delta(left, right, rng)
                rows.append(
                    {
                        "layer": layer,
                        "stress": stress,
                        "score_weights": weights.name,
                        "contrast": "relay_rich_minus_no_relay",
                        "metric": "alternative_operational_score",
                        "delta": delta,
                        "ci95_low": low,
                        "ci95_high": high,
                        "supports_harm": delta < 0,
                    }
                )
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(DATA / "score_sensitivity.csv", index=False, encoding="utf-8")
    return sensitivity


def write_report(runs: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    severe_safe = contrasts[
        (contrasts["stress"] == "severe")
        & (contrasts["metric"] == "safe_delivery_success")
        & (contrasts["contrast"].str.contains("relay_rich"))
    ]
    delay = contrasts[
        (contrasts["metric"] == "mean_effective_map_delay")
        & (contrasts["contrast"].str.contains("relay_rich"))
    ]
    sensitivity_summary = (
        sensitivity.groupby(["layer", "stress"], as_index=False)
        .agg(weight_sets=("score_weights", "nunique"), harm_rate=("supports_harm", "mean"), mean_delta=("delta", "mean"))
    )
    lines = [
        "# Additional Composition and Sensitivity Diagnostics",
        "",
        "## Purpose",
        "",
        "These analyses probe the robustness of the central relay result: whether it is a team-composition artifact, whether it depends on a particular operational-score weighting, and whether the command-layer effect generalises across strata.",
        "",
        "## Matched-Composition Diagnostic",
        "",
        f"- Runs: {len(runs):,}",
        "- Variants: no relay, balanced relay, relay-rich, relay-rich with standardized relay dynamics, relay-rich with scout count matched to no-relay, and the combined scout-matched/standardized variant.",
        "- Stress regimes: degraded and severe.",
        "- Map modes: delayed and scout_belief.",
        "- Command modes: autonomous and crowd_vector.",
        "",
        "### Severe Safe-Delivery Contrasts Against No-Relay",
        "",
        severe_safe.sort_values(["map_mode", "command_mode", "contrast"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Effective-Delay Contrasts Against No-Relay",
        "",
        delay.sort_values(["stress", "map_mode", "command_mode", "contrast"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Operational-Score Weight Sensitivity",
        "",
        sensitivity_summary.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- If scout-matched relay-rich variants still reduce delay while harming severe safe delivery, the result is harder to dismiss as a missing-scout artifact.",
        "- If standardized relay dynamics weaken but do not reverse the trade-off, the mechanism is partly embodied-role cost but not merely a slow-relay artifact.",
        "- If alternative operational-score weights mostly keep relay-rich deltas negative, the relay conclusion does not depend on one arbitrary composite score.",
        "- These analyses should be treated as auxiliary material unless they are promoted into a revised manuscript with compact reporting.",
    ]
    (REPORTS / "additional_diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "matched_runs": int(len(runs)),
        "matched_conditions": int(summary.shape[0]),
        "score_sensitivity_rows": int(len(sensitivity)),
        "severe_safe_harm_rows": int((severe_safe["delta"] < 0).sum()),
        "severe_safe_rows": int(len(severe_safe)),
    }
    (REPORTS / "auxiliary_diagnostics_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run additional auxiliary diagnostics.")
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    runs, summary = run_matched_composition(args.seeds, args.workers)
    contrasts = write_matched_contrasts(runs)
    sensitivity = write_score_sensitivity()
    write_report(runs, summary, contrasts, sensitivity)
    print(json.dumps({"matched_runs": len(runs), "contrasts": len(contrasts), "score_sensitivity": len(sensitivity)}))


if __name__ == "__main__":
    main()
