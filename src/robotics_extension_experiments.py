from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "theory_outputs"


def stable_seed(*parts: object) -> int:
    key = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % (2**32 - 1)


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.zeros_like(vec, dtype=float)
    return vec / norm


def bootstrap_delta(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, n_boot: int = 1200) -> tuple[float, float, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    deltas = np.empty(n_boot)
    for idx in range(n_boot):
        deltas[idx] = rng.choice(left, len(left), replace=True).mean() - rng.choice(right, len(right), replace=True).mean()
    return float(left.mean() - right.mean()), float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def add_safety_endpoints(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "n_agents" not in data:
        team_sizes = {
            "no_relay_hetero": 12,
            "relay_sparse": 12,
            "balanced_hetero": 12,
            "relay_mid": 12,
            "relay_rich": 12,
        }
        data["n_agents"] = data["team"].map(team_sizes).fillna(12)
    if "n_payload" not in data:
        data["n_payload"] = 2
    data["reached_fraction"] = data["reached_final"] / data["n_agents"]
    data["alive_fraction"] = data["alive_final"] / data["n_agents"]
    data["payload_fraction"] = data["payload_reached"] / data["n_payload"].replace(0, np.nan)
    data["payload_fraction"] = data["payload_fraction"].fillna(data["reached_fraction"]).clip(0, 1)
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


# ---------------------------------------------------------------------------
# Robustness study on the original continuous simulator.


def make_robust_profiles() -> list[dict[str, object]]:
    import swarm_pilot_experiment as base

    profiles: list[dict[str, object]] = []
    scales = [
        {"noise": 1.00, "loss": 1.00, "dropout": 1.00, "delay": 1.00},
        {"noise": 0.85, "loss": 0.80, "dropout": 0.50, "delay": 0.90},
        {"noise": 1.15, "loss": 1.20, "dropout": 1.50, "delay": 1.10},
        {"noise": 0.85, "loss": 1.20, "dropout": 1.50, "delay": 1.10},
        {"noise": 1.15, "loss": 0.80, "dropout": 0.50, "delay": 0.90},
        {"noise": 1.30, "loss": 1.10, "dropout": 1.00, "delay": 1.00},
        {"noise": 0.90, "loss": 1.35, "dropout": 1.25, "delay": 1.20},
        {"noise": 1.10, "loss": 0.90, "dropout": 1.75, "delay": 0.85},
        {"noise": 1.00, "loss": 1.00, "dropout": 2.00, "delay": 1.00},
    ]
    for stress_name in ["degraded", "severe"]:
        base_profile = base.STRESS_PROFILES[stress_name]
        for idx, scale in enumerate(scales):
            profile = dict(base_profile)
            profile["map_delay"] = int(max(0, round(float(profile["map_delay"]) * float(scale["delay"]))))
            profile["command_delay"] = int(max(0, round(float(profile["command_delay"]) * float(scale["delay"]))))
            profile["packet_loss"] = min(0.60, float(profile["packet_loss"]) * float(scale["loss"]))
            profile["burst_loss"] = min(0.70, float(profile["burst_loss"]) * float(scale["loss"]))
            profile["command_noise"] = min(0.55, float(profile["command_noise"]) * float(scale["noise"]))
            profile["scout_dropout"] = min(0.0080, float(profile["scout_dropout"]) * float(scale["dropout"]))
            profile["relay_dropout"] = min(0.0070, float(profile["relay_dropout"]) * float(scale["dropout"]))
            profiles.append(
                {
                    "variant": f"{stress_name}_robust_{idx:02d}",
                    "stress_base": stress_name,
                    "profile": profile,
                    **scale,
                }
            )
    return profiles


def simulate_robust_spec(spec: dict[str, object]) -> dict[str, object]:
    import swarm_pilot_experiment as base

    stress_name = str(spec["stress"])
    base.STRESS_PROFILES[stress_name] = dict(spec["profile"])  # process-local mutation
    record = base.simulate_run(
        seed=int(spec["seed"]),
        team=str(spec["team"]),
        map_mode=str(spec["map_mode"]),
        command_mode=str(spec["command_mode"]),
        stress_name=stress_name,
        mission_profile="full_delivery",
    )
    record["stress_base"] = spec["stress_base"]
    record["variant"] = spec["variant"]
    record["noise_scale"] = spec["noise"]
    record["loss_scale"] = spec["loss"]
    record["dropout_scale"] = spec["dropout"]
    record["delay_scale"] = spec["delay"]
    record["max_steps"] = 180
    record["safe_exposure_cut"] = 0.150
    record["safe_exposure_scale"] = 0.220
    return record


def run_robustness(seeds_per_condition: int = 8, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = make_robust_profiles()
    specs: list[dict[str, object]] = []
    for profile in profiles:
        for team in ["no_relay_hetero", "relay_rich"]:
            for map_mode in ["delayed", "scout_belief"]:
                for command_mode in ["autonomous", "crowd_vector"]:
                    for rep in range(seeds_per_condition):
                        specs.append(
                            {
                                **profile,
                                "stress": profile["variant"],
                                "team": team,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "seed": stable_seed("robust", profile["variant"], team, map_mode, command_mode, rep),
                            }
                        )
    records: list[dict[str, object]] = []
    chunk = max(1, len(specs) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, record in enumerate(executor.map(simulate_robust_spec, specs, chunksize=chunk), start=1):
            records.append(record)
            if idx % 500 == 0:
                print(f"robustness completed {idx}/{len(specs)}")
    runs = add_safety_endpoints(pd.DataFrame(records))
    runs.to_csv(OUT / "robotics_robustness_runs.csv", index=False, encoding="utf-8")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(20260710)
    for variant, frame in runs.groupby("variant"):
        for metric in ["safe_delivery_success", "operational_score", "degraded_outcome", "attrition_rate"]:
            crowd = frame[frame["command_mode"] == "crowd_vector"][metric].to_numpy(float)
            auto = frame[frame["command_mode"] == "autonomous"][metric].to_numpy(float)
            delta, lo, hi = bootstrap_delta(crowd, auto, rng)
            rows.append(
                {
                    "variant": variant,
                    "stress_base": frame["stress_base"].iloc[0],
                    "contrast": "crowd_minus_auto",
                    "metric": metric,
                    "delta": delta,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "expected_direction": "positive" if metric in {"safe_delivery_success", "operational_score"} else "negative",
                }
            )
            relay_rich = frame[frame["team"] == "relay_rich"][metric].to_numpy(float)
            no_relay = frame[frame["team"] == "no_relay_hetero"][metric].to_numpy(float)
            delta, lo, hi = bootstrap_delta(relay_rich, no_relay, rng)
            rows.append(
                {
                    "variant": variant,
                    "stress_base": frame["stress_base"].iloc[0],
                    "contrast": "relay_rich_minus_no_relay",
                    "metric": metric,
                    "delta": delta,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "expected_direction": "negative" if metric in {"safe_delivery_success", "operational_score"} else "positive",
                }
            )
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(OUT / "robotics_robustness_contrasts.csv", index=False, encoding="utf-8")
    return runs, contrasts


# ---------------------------------------------------------------------------
# Independent grid-world reproduction.


@dataclass(frozen=True)
class GridRole:
    sensor: float
    survival: float
    target_gain: float
    avoid_gain: float
    command_gain: float


GRID_ROLES = {
    "standard": GridRole(sensor=4.0, survival=1.00, target_gain=1.00, avoid_gain=1.05, command_gain=0.55),
    "scout": GridRole(sensor=7.0, survival=0.92, target_gain=0.95, avoid_gain=1.25, command_gain=0.55),
    "relay": GridRole(sensor=4.5, survival=1.06, target_gain=0.82, avoid_gain=0.82, command_gain=0.45),
    "payload": GridRole(sensor=3.5, survival=0.86, target_gain=1.18, avoid_gain=0.78, command_gain=0.50),
}

GRID_TEAMS = {
    "no_relay_hetero": ["standard"] * 7 + ["scout"] * 3 + ["payload"] * 2,
    "relay_sparse": ["standard"] * 6 + ["scout"] * 3 + ["relay"] * 1 + ["payload"] * 2,
    "balanced_hetero": ["standard"] * 5 + ["scout"] * 3 + ["relay"] * 2 + ["payload"] * 2,
    "relay_rich": ["standard"] * 4 + ["scout"] * 2 + ["relay"] * 4 + ["payload"] * 2,
}

GRID_STRESS = {
    "degraded": {"map_delay": 6, "command_delay": 3, "packet_loss": 0.12, "command_noise": 0.22, "relay_dropout": 0.0010},
    "severe": {"map_delay": 12, "command_delay": 5, "packet_loss": 0.24, "command_noise": 0.36, "relay_dropout": 0.0025},
}

GRID_DIRS = np.array(
    [
        [1, 0],
        [1, 1],
        [1, -1],
        [0, 1],
        [0, -1],
        [-1, 0],
        [-1, 1],
        [-1, -1],
    ],
    dtype=float,
)
GRID_DIRS = GRID_DIRS / np.linalg.norm(GRID_DIRS, axis=1, keepdims=True)


def grid_hazard(pos: np.ndarray, t: int, phase: float, width: int, height: int) -> float:
    x = float(pos[0]) / width
    y = float(pos[1]) / height
    tx = t / 120.0
    ridge_x = 0.50 + 0.08 * math.sin(2.2 * math.pi * tx + phase)
    ridge = 0.58 * math.exp(-0.5 * ((x - ridge_x) / 0.065) ** 2)
    pocket1 = 0.62 * math.exp(-0.5 * (((x - 0.68) / 0.10) ** 2 + ((y - (0.34 + 0.07 * math.sin(phase + 3 * tx))) / 0.10) ** 2))
    pocket2 = 0.45 * math.exp(-0.5 * (((x - (0.36 + 0.04 * math.cos(phase + 4 * tx))) / 0.10) ** 2 + ((y - 0.68) / 0.10) ** 2))
    return float(np.clip(ridge + pocket1 + pocket2, 0, 1))


def grid_grad(map_func, pos: np.ndarray, width: int, height: int) -> np.ndarray:
    x, y = pos.astype(float)
    px = np.array([min(width - 1, x + 1), y])
    nx = np.array([max(0, x - 1), y])
    py = np.array([x, min(height - 1, y + 1)])
    ny = np.array([x, max(0, y - 1)])
    return np.array([map_func(px) - map_func(nx), map_func(py) - map_func(ny)])


def simulate_grid_run(spec: dict[str, object]) -> dict[str, object]:
    rng = np.random.default_rng(int(spec["seed"]))
    width, height, max_steps = 42, 26, 120
    roles = GRID_TEAMS[str(spec["team"])]
    n_agents = len(roles)
    n_payload = sum(1 for role in roles if role == "payload")
    stress = GRID_STRESS[str(spec["stress"])]
    relay_count = roles.count("relay")
    relay_bonus = min(0.62, 0.155 * relay_count)
    effective_map_delay = int(round(float(stress["map_delay"]) * (1.0 - relay_bonus)))
    packet_loss = max(0.0, float(stress["packet_loss"]) * (1.0 - relay_bonus))
    command_delay = int(stress["command_delay"])
    start = np.array([2.0, height / 2.0])
    target = np.array([width - 3.0, height / 2.0])
    positions = np.zeros((n_agents, 2), dtype=float)
    for idx in range(n_agents):
        positions[idx] = start + np.array([rng.integers(0, 3), rng.integers(-4, 5)], dtype=float)
    positions[:, 0] = np.clip(positions[:, 0], 1, width - 2)
    positions[:, 1] = np.clip(positions[:, 1], 1, height - 2)
    alive = np.ones(n_agents, dtype=bool)
    reached = np.zeros(n_agents, dtype=bool)
    phase = rng.uniform(0, 2 * math.pi)
    command_queue: list[np.ndarray | None] = []
    last_command = np.zeros(2)
    exposure = 0.0
    early_exposure = 0.0
    recovery_time = 0
    recovering = False
    alive_counts: list[int] = []
    map_mse_values: list[float] = []
    consensus_values: list[float] = []

    for t in range(max_steps):
        scout_positions = positions[alive & np.array([role == "scout" for role in roles])]

        def true_map(p: np.ndarray, time: int = t) -> float:
            return grid_hazard(p, max(0, time), phase, width, height)

        def perceived_map(p: np.ndarray) -> float:
            if spec["map_mode"] == "no_shared":
                return 0.12
            delayed = true_map(p, t - effective_map_delay)
            if spec["map_mode"] == "delayed":
                return delayed
            if spec["map_mode"] == "scout_belief":
                if len(scout_positions) == 0:
                    coverage = 0.0
                else:
                    distances = np.linalg.norm(scout_positions - p, axis=1)
                    coverage = float(np.max(np.exp(-0.5 * (distances / 7.0) ** 2)))
                sensed = true_map(p, t - effective_map_delay // 3)
                noise = rng.normal(0, 0.035)
                return float(np.clip(coverage * (sensed + noise) + (1 - coverage) * (0.35 * delayed + 0.10), 0, 1))
            return true_map(p)

        centroid = positions[alive].mean(axis=0) if np.any(alive) else start.copy()
        target_dir = unit(target - centroid)
        command = np.zeros(2)
        if spec["command_mode"] in {"crowd_vector", "consensus_gated", "stale_hold"}:
            choices = []
            utilities = []
            for direction in GRID_DIRS:
                lookahead = np.clip(centroid + 4.0 * direction, [0, 0], [width - 1, height - 1])
                utility = float(np.dot(direction, target_dir) - 1.75 * perceived_map(lookahead))
                utility += rng.normal(0, float(stress["command_noise"]))
                utilities.append(utility)
                choices.append(direction)
            best = np.argsort(utilities)[-5:]
            selected = np.array([choices[int(idx)] for idx in best])
            aggregate = selected.mean(axis=0)
            command = unit(aggregate)
            consensus = float(np.linalg.norm(aggregate))
            consensus_values.append(consensus)
            if spec["command_mode"] == "consensus_gated":
                risk_ahead = perceived_map(np.clip(centroid + 4.0 * command, [0, 0], [width - 1, height - 1]))
                if consensus < 0.52 or risk_ahead > 0.38:
                    command = 0.20 * command
            if rng.random() < packet_loss:
                command_queue.append(None)
            else:
                command_queue.append(command.copy())
            if len(command_queue) > command_delay:
                queued = command_queue.pop(0)
                command = last_command.copy() if queued is None and spec["command_mode"] == "stale_hold" else queued
                if command is None:
                    command = np.zeros(2)
            else:
                command = last_command.copy() if spec["command_mode"] == "stale_hold" else np.zeros(2)
            last_command = command.copy()

        for idx, role in enumerate(roles):
            if not alive[idx] or reached[idx]:
                continue
            if role == "relay" and rng.random() < float(stress["relay_dropout"]):
                alive[idx] = False
                continue
            role_spec = GRID_ROLES[role]
            pos = positions[idx]
            perceived = perceived_map(pos)
            avoid = -unit(grid_grad(perceived_map, pos, width, height)) if perceived > 0.10 else np.zeros(2)
            cohesion = unit(centroid - pos)
            move_vec = role_spec.target_gain * unit(target - pos) + role_spec.avoid_gain * perceived * avoid + 0.18 * cohesion
            if spec["command_mode"] in {"crowd_vector", "consensus_gated", "stale_hold"}:
                move_vec += role_spec.command_gain * command
            direction = GRID_DIRS[int(np.argmax(GRID_DIRS @ unit(move_vec)))]
            new_pos = np.clip(np.round(pos + direction), [0, 0], [width - 1, height - 1])
            positions[idx] = new_pos

        step_exposure = 0.0
        for idx, role in enumerate(roles):
            if not alive[idx] or reached[idx]:
                continue
            hazard = true_map(positions[idx], t)
            step_exposure += hazard
            kill_prob = max(0.0, (hazard - 0.55) * 0.10 / GRID_ROLES[role].survival)
            if rng.random() < kill_prob:
                alive[idx] = False
        exposure += step_exposure
        if t < int(max_steps * 0.42):
            early_exposure += step_exposure
        alive_counts.append(int(alive.sum()))

        reached |= alive & (np.linalg.norm(positions - target, axis=1) <= 2.0)
        if t % 6 == 0:
            probes = np.column_stack([rng.integers(0, width, size=20), rng.integers(0, height, size=20)]).astype(float)
            mse = np.mean([(perceived_map(p) - true_map(p, t)) ** 2 for p in probes])
            map_mse_values.append(float(mse))
        per_alive_exposure = step_exposure / max(1, int(alive.sum()))
        if per_alive_exposure > 0.42:
            recovering = True
        if recovering:
            recovery_time += 1
            if per_alive_exposure < 0.22:
                recovering = False

        payload_reached = sum(1 for role, did_reach in zip(roles, reached) if role == "payload" and did_reach)
        strict_success = (
            reached.sum() >= math.ceil(0.75 * n_agents)
            and alive.sum() >= math.ceil(0.67 * n_agents)
            and payload_reached >= math.ceil(0.50 * n_payload)
        )
        if strict_success:
            success_time = t
            break
    else:
        success_time = max_steps
        payload_reached = sum(1 for role, did_reach in zip(roles, reached) if role == "payload" and did_reach)

    attrition_rate = (n_agents - int(alive.sum())) / n_agents
    exposure_per_agent_step = exposure / max(1, sum(alive_counts))
    return {
        "environment": "grid_world",
        "seed": int(spec["seed"]),
        "team": spec["team"],
        "relay_count": relay_count,
        "map_mode": spec["map_mode"],
        "command_mode": spec["command_mode"],
        "stress": spec["stress"],
        "n_agents": n_agents,
        "n_payload": n_payload,
        "max_steps": max_steps,
        "success_time": int(success_time),
        "alive_final": int(alive.sum()),
        "reached_final": int(reached.sum()),
        "payload_reached": int(payload_reached),
        "attrition_rate": float(attrition_rate),
        "exposure_per_agent_step": float(exposure_per_agent_step),
        "early_exposure_per_agent_step": float(early_exposure / max(1, int(max_steps * 0.42) * n_agents)),
        "recovery_time": int(recovery_time),
        "mean_effective_map_delay": float(effective_map_delay),
        "early_map_mse": float(np.mean(map_mse_values)) if map_mse_values else 0.0,
        "mean_consensus": float(np.mean(consensus_values)) if consensus_values else 0.0,
        "safe_exposure_cut": 0.170,
        "safe_exposure_scale": 0.250,
    }


def run_grid_reproduction(seeds_per_condition: int = 24, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs: list[dict[str, object]] = []
    for team in ["no_relay_hetero", "relay_sparse", "balanced_hetero", "relay_rich"]:
        for map_mode in ["delayed", "scout_belief", "no_shared"]:
            for command_mode in ["autonomous", "crowd_vector", "consensus_gated", "stale_hold"]:
                for stress in ["degraded", "severe"]:
                    for rep in range(seeds_per_condition):
                        specs.append(
                            {
                                "team": team,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "stress": stress,
                                "seed": stable_seed("grid", team, map_mode, command_mode, stress, rep),
                            }
                        )
    records: list[dict[str, object]] = []
    chunk = max(1, len(specs) // (workers * 8))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for idx, record in enumerate(executor.map(simulate_grid_run, specs, chunksize=chunk), start=1):
            records.append(record)
            if idx % 500 == 0:
                print(f"grid-world completed {idx}/{len(specs)}")
    runs = add_safety_endpoints(pd.DataFrame(records))
    runs.to_csv(OUT / "robotics_grid_runs.csv", index=False, encoding="utf-8")
    summary = (
        runs.groupby(["stress", "team", "map_mode", "command_mode"])
        .agg(
            n=("seed", "size"),
            safe_delivery=("safe_delivery_success", "mean"),
            operational_score=("operational_score", "mean"),
            degraded=("degraded_outcome", "mean"),
            attrition=("attrition_rate", "mean"),
            exposure=("exposure_per_agent_step", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "robotics_grid_summary.csv", index=False, encoding="utf-8")

    rng = np.random.default_rng(20260710)
    rows: list[dict[str, object]] = []
    for stress, stress_frame in runs.groupby("stress"):
        for metric in ["safe_delivery_success", "operational_score", "degraded_outcome", "attrition_rate", "mean_effective_map_delay"]:
            auto = stress_frame[stress_frame["command_mode"] == "autonomous"][metric].to_numpy(float)
            for command_mode in ["crowd_vector", "consensus_gated", "stale_hold"]:
                command_values = stress_frame[stress_frame["command_mode"] == command_mode][metric].to_numpy(float)
                delta, lo, hi = bootstrap_delta(command_values, auto, rng)
                rows.append({"stress": stress, "contrast": f"{command_mode}_minus_auto", "metric": metric, "delta": delta, "ci95_low": lo, "ci95_high": hi})
            relay_rich = stress_frame[stress_frame["team"] == "relay_rich"][metric].to_numpy(float)
            no_relay = stress_frame[stress_frame["team"] == "no_relay_hetero"][metric].to_numpy(float)
            delta, lo, hi = bootstrap_delta(relay_rich, no_relay, rng)
            rows.append({"stress": stress, "contrast": "relay_rich_minus_no_relay", "metric": metric, "delta": delta, "ci95_low": lo, "ci95_high": hi})
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(OUT / "robotics_grid_contrasts.csv", index=False, encoding="utf-8")
    return runs, contrasts


def write_extension_reports(grid_runs: pd.DataFrame, grid_contrasts: pd.DataFrame, robust_runs: pd.DataFrame, robust_contrasts: pd.DataFrame) -> None:
    robust_signs = (
        robust_contrasts.assign(
            expected_pass=lambda df: np.where(
                df["expected_direction"] == "positive",
                df["delta"] > 0,
                df["delta"] < 0,
            )
        )
        .groupby(["contrast", "metric", "stress_base"])
        .agg(
            variants=("variant", "nunique"),
            expected_pass_rate=("expected_pass", "mean"),
            mean_delta=("delta", "mean"),
        )
        .reset_index()
    )
    robust_signs.to_csv(OUT / "robotics_robustness_sign_summary.csv", index=False, encoding="utf-8")

    grid_command = (
        grid_runs.groupby(["stress", "command_mode"])
        .agg(
            safe_delivery=("safe_delivery_success", "mean"),
            operational_score=("operational_score", "mean"),
            degraded=("degraded_outcome", "mean"),
            attrition=("attrition_rate", "mean"),
        )
        .reset_index()
    )
    grid_relay = (
        grid_runs.groupby(["stress", "team", "relay_count"])
        .agg(
            safe_delivery=("safe_delivery_success", "mean"),
            operational_score=("operational_score", "mean"),
            degraded=("degraded_outcome", "mean"),
            attrition=("attrition_rate", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
        )
        .reset_index()
        .sort_values(["stress", "relay_count"])
    )

    report = f"""# Robotics Extension Report

## Scope

- Independent grid-world reproduction runs: {len(grid_runs):,}
- Continuous-simulator robustness runs: {len(robust_runs):,}
- Human study: not included.

## Why This Matters

This extension adds two validation layers: an independent grid-world navigation environment and stress-parameter robustness sweeps in the original simulator.

## Grid-World Command Summary

{grid_command.to_markdown(index=False, floatfmt=".3f")}

## Grid-World Main Contrasts

{grid_contrasts.to_markdown(index=False, floatfmt=".4f")}

## Grid-World Relay Summary

{grid_relay.to_markdown(index=False, floatfmt=".3f")}

## Original-Simulator Robustness Sign Summary

{robust_signs.to_markdown(index=False, floatfmt=".3f")}

## Interpretation

- If a grid-world command policy aligns with the continuous simulator, the command-layer claim is no longer tied to one kinematic toy model. If ungated crowd control fails, treat it as a boundary condition and emphasize risk-gated command layers.
- If relay-rich still reduces delay while harming safety/attrition, the relay over-allocation claim becomes a robotics allocation trade-off rather than an artifact of the original role parameters.
- Robustness sign rates below 0.70 should be treated as conditional evidence, not a main claim.
"""
    (OUT / "robotics_extension_report.md").write_text(report, encoding="utf-8")

    blueprint = f"""# Transfer and Robustness Summary

## Scope

The continuous simulator is used as a controlled testbed for a robotics-relevant allocation question: when delayed hazard information is present, should designers add relay capacity or use command-layer aggregation to preserve mission safety?

## Additional Validation Layers

1. Independent environment replication: continuous 2D model plus grid-world navigation.
2. Parameter robustness: command noise, packet loss, delay, and dropout are varied around degraded/severe regimes.
3. Role-allocation sweep: relay count is varied while payload/scout structure is held comparable.
4. Conservative claims: relay-rich harm is claimed; exact one-relay optimality is not.

## What Can Be Claimed

- Command aggregation improves or preserves operational resilience across controlled continuous regimes; grid-world results should be used to distinguish ungated crowd control from risk-gated command layers.
- Relay-rich allocation reliably lowers effective delay but can reduce safe delivery through role opportunity cost and attrition.
- Diagnostic signals are useful as auxiliary monitoring, not as the central robotics contribution.

## Limitations

- No hardware or ROS/Gazebo validation.
- No real operator input.
- Grid-world is an independent abstraction, not a full physics simulator.
- Operational-score weights must remain frozen and justified.

## Interpretation

The strongest interpretation is an autonomous-systems resilience and multi-robot design trade-off result, not a pure control-theory or hardware deployment result.
"""
    (OUT / "transfer_robustness_summary.md").write_text(blueprint, encoding="utf-8")


def write_extension_figures(grid_runs: pd.DataFrame, robust_contrasts: pd.DataFrame) -> None:
    grid_command = (
        grid_runs.groupby(["stress", "command_mode"])
        .agg(safe_delivery=("safe_delivery_success", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    for ax, stress in zip(axes, ["degraded", "severe"]):
        order = ["autonomous", "crowd_vector", "consensus_gated", "stale_hold"]
        sub = grid_command[grid_command["stress"] == stress].set_index("command_mode").loc[order]
        ax.bar(range(len(order)), sub["safe_delivery"], color=["#6b7280", "#2563eb", "#059669", "#d97706"])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(["auto", "crowd", "gated", "stale"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Safe delivery")
        ax.set_title(f"Grid-world {stress}")
    fig.tight_layout()
    fig.savefig(OUT / "fig_grid_command_reproduction.png", dpi=220)
    plt.close(fig)

    grid_relay = (
        grid_runs.groupby(["stress", "relay_count"])
        .agg(
            safe_delivery=("safe_delivery_success", "mean"),
            degraded=("degraded_outcome", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True)
    for ax, stress in zip(axes, ["degraded", "severe"]):
        sub = grid_relay[grid_relay["stress"] == stress]
        ax.plot(sub["relay_count"], sub["safe_delivery"], marker="o", color="#2563eb", label="safe")
        ax.plot(sub["relay_count"], sub["degraded"], marker="s", color="#dc2626", label="degraded")
        ax.set_xlabel("Relay count")
        ax.set_ylim(0, 1)
        ax.set_title(f"Grid-world relay {stress}")
        ax2 = ax.twinx()
        ax2.plot(sub["relay_count"], sub["effective_delay"], marker="^", color="#16a34a", label="delay")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grid_relay_tradeoff.png", dpi=220)
    plt.close(fig)

    signs = (
        robust_contrasts.assign(
            expected_pass=lambda df: np.where(df["expected_direction"] == "positive", df["delta"] > 0, df["delta"] < 0)
        )
        .groupby(["contrast", "metric"])
        .agg(pass_rate=("expected_pass", "mean"))
        .reset_index()
    )
    labels = signs["contrast"] + "\n" + signs["metric"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(range(len(signs)), signs["pass_rate"], color="#4f46e5")
    ax.axhline(0.70, color="#9ca3af", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Expected sign rate")
    ax.set_xticks(range(len(signs)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_title("Robustness sign consistency")
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness_signs.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    robust_runs, robust_contrasts = run_robustness(seeds_per_condition=8, workers=workers)
    grid_runs, grid_contrasts = run_grid_reproduction(seeds_per_condition=24, workers=workers)
    write_extension_reports(grid_runs, grid_contrasts, robust_runs, robust_contrasts)
    write_extension_figures(grid_runs, robust_contrasts)
    print(
        json.dumps(
            {
                "robustness_runs": int(len(robust_runs)),
                "grid_runs": int(len(grid_runs)),
                "outputs": [
                    "robotics_extension_report.md",
                    "transfer_robustness_summary.md",
                    "robotics_grid_runs.csv",
                    "robotics_grid_contrasts.csv",
                    "robotics_robustness_runs.csv",
                    "robotics_robustness_contrasts.csv",
                    "fig_grid_command_reproduction.png",
                    "fig_grid_relay_tradeoff.png",
                    "fig_robustness_signs.png",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
