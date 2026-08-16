"""Analyze Webots validation results for the manuscript package."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "theory_outputs"


def bootstrap_delta(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n_boot: int = 1500) -> tuple[float, float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return float("nan"), float("nan"), float("nan")
    delta = float(np.mean(left) - np.mean(right))
    boot = []
    for _ in range(n_boot):
        boot.append(float(np.mean(rng.choice(left, len(left), replace=True)) - np.mean(rng.choice(right, len(right), replace=True))))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return delta, float(lo), float(hi)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        runs.groupby(["stress", "team", "relay_count", "command_mode"], as_index=False)
        .agg(
            n=("seed", "count"),
            safe_delivery=("safe_delivery_success", "mean"),
            operational_score=("operational_score", "mean"),
            degraded=("degraded_outcome", "mean"),
            attrition=("attrition_rate", "mean"),
            payload_delivery=("payload_delivery_rate", "mean"),
            exposure=("mean_hazard_exposure", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
            consensus=("mean_command_consensus", "mean"),
            reversal=("command_reversal_rate", "mean"),
        )
        .sort_values(["stress", "team", "command_mode"])
    )
    rng = np.random.default_rng(87231)
    rows = []
    for stress in sorted(runs["stress"].unique()):
        stress_frame = runs[runs["stress"] == stress]
        for metric in [
            "safe_delivery_success",
            "operational_score",
            "degraded_outcome",
            "attrition_rate",
            "payload_delivery_rate",
            "mean_hazard_exposure",
        ]:
            auto = stress_frame[stress_frame["command_mode"] == "autonomous"][metric].to_numpy(float)
            for command_mode in ["crowd_vector", "consensus_gated"]:
                command_values = stress_frame[stress_frame["command_mode"] == command_mode][metric].to_numpy(float)
                delta, lo, hi = bootstrap_delta(command_values, auto, rng)
                rows.append({"stress": stress, "contrast": f"{command_mode}_minus_auto", "metric": metric, "delta": delta, "ci95_low": lo, "ci95_high": hi})
            relay_rich = stress_frame[stress_frame["team"] == "relay_rich"][metric].to_numpy(float)
            no_relay = stress_frame[stress_frame["team"] == "no_relay_hetero"][metric].to_numpy(float)
            delta, lo, hi = bootstrap_delta(relay_rich, no_relay, rng)
            rows.append({"stress": stress, "contrast": "relay_rich_minus_no_relay", "metric": metric, "delta": delta, "ci95_low": lo, "ci95_high": hi})
        relay_rich_delay = stress_frame[stress_frame["team"] == "relay_rich"]["mean_effective_map_delay"].to_numpy(float)
        no_relay_delay = stress_frame[stress_frame["team"] == "no_relay_hetero"]["mean_effective_map_delay"].to_numpy(float)
        delta, lo, hi = bootstrap_delta(relay_rich_delay, no_relay_delay, rng)
        rows.append({"stress": stress, "contrast": "relay_rich_minus_no_relay", "metric": "mean_effective_map_delay", "delta": delta, "ci95_low": lo, "ci95_high": hi})
    contrasts = pd.DataFrame(rows)
    return summary, contrasts


def write_figures(runs: pd.DataFrame) -> None:
    command = (
        runs.groupby(["stress", "command_mode"], as_index=False)
        .agg(safe_delivery=("safe_delivery_success", "mean"), score=("operational_score", "mean"))
    )
    order = ["autonomous", "crowd_vector", "consensus_gated"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, stress in zip(axes, ["degraded", "severe"]):
        sub = command[command["stress"] == stress].set_index("command_mode").reindex(order)
        ax.bar(range(len(order)), sub["safe_delivery"], color=["#64748b", "#2563eb", "#059669"])
        ax.set_title(f"Webots command {stress}")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["auto", "crowd", "gated"], rotation=20)
        ax.set_ylim(0, 1)
        ax.set_ylabel("safe delivery")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_webots_command_safe_delivery.png", dpi=220)
    plt.close(fig)

    relay = (
        runs.groupby(["stress", "relay_count"], as_index=False)
        .agg(
            safe_delivery=("safe_delivery_success", "mean"),
            degraded=("degraded_outcome", "mean"),
            attrition=("attrition_rate", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
        )
        .sort_values(["stress", "relay_count"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    for ax, stress in zip(axes, ["degraded", "severe"]):
        sub = relay[relay["stress"] == stress]
        ax.plot(sub["relay_count"], sub["safe_delivery"], marker="o", color="#2563eb", label="safe")
        ax.plot(sub["relay_count"], sub["degraded"], marker="s", color="#dc2626", label="degraded")
        ax.plot(sub["relay_count"], sub["attrition"], marker="x", color="#9333ea", label="attrition")
        ax2 = ax.twinx()
        ax2.plot(sub["relay_count"], sub["effective_delay"], marker="^", color="#16a34a", label="delay")
        ax.set_title(f"Webots relay {stress}")
        ax.set_xlabel("relay count")
        ax.set_ylim(0, 1)
        ax2.set_ylabel("effective delay")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_webots_relay_tradeoff.png", dpi=220)
    plt.close(fig)


def write_report(runs: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    command_summary = (
        runs.groupby(["stress", "command_mode"], as_index=False)
        .agg(
            safe_delivery=("safe_delivery_success", "mean"),
            operational_score=("operational_score", "mean"),
            degraded=("degraded_outcome", "mean"),
            attrition=("attrition_rate", "mean"),
            payload_delivery=("payload_delivery_rate", "mean"),
        )
        .sort_values(["stress", "command_mode"])
    )
    relay_summary = (
        runs.groupby(["stress", "team", "relay_count"], as_index=False)
        .agg(
            safe_delivery=("safe_delivery_success", "mean"),
            operational_score=("operational_score", "mean"),
            degraded=("degraded_outcome", "mean"),
            attrition=("attrition_rate", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
        )
        .sort_values(["stress", "relay_count"])
    )
    def contrast_line(stress: str, contrast: str, metric: str) -> str:
        row = contrasts[
            (contrasts["stress"] == stress)
            & (contrasts["contrast"] == contrast)
            & (contrasts["metric"] == metric)
        ].iloc[0]
        return f"{row['delta']:.4f}, CI [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"

    report = f"""# Webots Physics Validation Report

## Scope

- Environment: Webots R2025a physics simulation with 12 differential-drive robots.
- Runs: {len(runs):,}
- Human study: not included.
- Main role: add a physics-based robotics validation layer, not replace the continuous/grid-world analyses.

## Command Summary

{command_summary.to_markdown(index=False, floatfmt=".3f")}

## Main Contrasts

{contrasts.to_markdown(index=False, floatfmt=".4f")}

## Relay Summary

{relay_summary.to_markdown(index=False, floatfmt=".3f")}

## Claim-Level Readout

### Relay over-allocation

- Degraded relay-rich vs no-relay safe delivery: {contrast_line("degraded", "relay_rich_minus_no_relay", "safe_delivery_success")}
- Degraded relay-rich vs no-relay degraded outcome: {contrast_line("degraded", "relay_rich_minus_no_relay", "degraded_outcome")}
- Degraded relay-rich vs no-relay attrition: {contrast_line("degraded", "relay_rich_minus_no_relay", "attrition_rate")}
- Degraded relay-rich vs no-relay effective delay: {contrast_line("degraded", "relay_rich_minus_no_relay", "mean_effective_map_delay")}
- Severe relay-rich vs no-relay safe delivery: {contrast_line("severe", "relay_rich_minus_no_relay", "safe_delivery_success")}
- Severe relay-rich vs no-relay degraded outcome: {contrast_line("severe", "relay_rich_minus_no_relay", "degraded_outcome")}
- Severe relay-rich vs no-relay attrition: {contrast_line("severe", "relay_rich_minus_no_relay", "attrition_rate")}
- Severe relay-rich vs no-relay effective delay: {contrast_line("severe", "relay_rich_minus_no_relay", "mean_effective_map_delay")}

This is the strongest Webots result: relay-rich allocation sharply reduces effective delay, but it also worsens safe delivery, degraded outcome, and attrition. The effect is especially strong under severe stress.

### Command layer

- Degraded crowd-vector vs autonomous safe delivery: {contrast_line("degraded", "crowd_vector_minus_auto", "safe_delivery_success")}
- Degraded crowd-vector vs autonomous operational score: {contrast_line("degraded", "crowd_vector_minus_auto", "operational_score")}
- Degraded consensus-gated vs autonomous operational score: {contrast_line("degraded", "consensus_gated_minus_auto", "operational_score")}
- Severe crowd-vector vs autonomous safe delivery: {contrast_line("severe", "crowd_vector_minus_auto", "safe_delivery_success")}
- Severe consensus-gated vs autonomous safe delivery: {contrast_line("severe", "consensus_gated_minus_auto", "safe_delivery_success")}

The Webots command result is conditional. It supports a modest degraded-regime command-layer benefit, especially for operational score, but it does not justify an unconditional command-layer superiority claim.

## Interpretation Rules

- If relay-rich reduces effective delay but safe delivery, attrition, or degraded outcome worsens, this supports the latency-safety trade-off in a physics-based simulator.
- If command aggregation is positive only in one stress regime, interpret it as stress-dependent rather than universal.
- If command aggregation is weak in Webots, treat the result as a transfer boundary for command-layer effects.

## Cross-Environment Interpretation

This Webots panel supports the cross-environment latency-safety trade-off: continuous simulation, relay sweep, grid-world severe transfer, and Webots physics all show that additional relay capacity can reduce delay without guaranteeing safer task completion.

The command-layer claim should remain secondary and environment-dependent. Webots does not overturn the grid-world boundary condition; instead it supports a careful phrasing: command aggregation can improve operational performance in selected continuous/physics regimes, but its safe-delivery benefit is not universal.
"""
    (OUT / "webots_validation_report.md").write_text(report, encoding="utf-8")
    addendum = f"""# Webots Validation Addendum

## Summary

The Webots layer provides a physics-based validation panel for the relay over-allocation trade-off. It should not be used as proof that command aggregation is universally beneficial.

## Strongest Webots-Supported Claim

Relay-rich allocation reduces effective delay but worsens safety-aware outcomes.

- Degraded safe delivery: {contrast_line("degraded", "relay_rich_minus_no_relay", "safe_delivery_success")}
- Degraded attrition: {contrast_line("degraded", "relay_rich_minus_no_relay", "attrition_rate")}
- Severe safe delivery: {contrast_line("severe", "relay_rich_minus_no_relay", "safe_delivery_success")}
- Severe attrition: {contrast_line("severe", "relay_rich_minus_no_relay", "attrition_rate")}
- Severe degraded outcome: {contrast_line("severe", "relay_rich_minus_no_relay", "degraded_outcome")}

## Command-Layer Interpretation

Command aggregation is not the main Webots claim.

- Degraded crowd-vector safe-delivery delta: {contrast_line("degraded", "crowd_vector_minus_auto", "safe_delivery_success")}
- Degraded consensus-gated operational-score delta: {contrast_line("degraded", "consensus_gated_minus_auto", "operational_score")}
- Severe safe-delivery deltas are weak and confidence intervals cross zero.

Recommended phrasing: command-layer effects are environment-dependent and stress-dependent; relay over-allocation produces the robust robotics trade-off.

## Evidence Layer

The Webots R2025a panel contributes one validation layer within a multi-environment robotics-resilience study:

1. Continuous 2D simulation for core mechanism.
2. Relay sweep for allocation trade-off.
3. Grid-world transfer for boundary condition.
4. Webots physics validation for differential-drive multi-robot realism.

The supported interpretation is a trade-off characterization, not a universal new control algorithm.
"""
    (OUT / "webots_validation_addendum.md").write_text(addendum, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Webots validation runs.")
    parser.add_argument("--input", type=Path, default=OUT / "webots_runs.csv")
    args = parser.parse_args()

    runs = pd.read_csv(args.input)
    summary, contrasts = summarize(runs)
    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "webots_condition_summary.csv", index=False, encoding="utf-8")
    contrasts.to_csv(OUT / "webots_contrasts.csv", index=False, encoding="utf-8")
    write_figures(runs)
    write_report(runs, summary, contrasts)
    print(f"Analyzed {len(runs):,} Webots runs")


if __name__ == "__main__":
    main()
