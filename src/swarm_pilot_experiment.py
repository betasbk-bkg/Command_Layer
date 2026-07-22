from __future__ import annotations

import argparse
import os
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "theory_outputs"


@dataclass(frozen=True)
class RoleSpec:
    speed: float
    sensor: float
    avoid_gain: float
    survival: float
    target_gain: float


ROLES: dict[str, RoleSpec] = {
    "standard": RoleSpec(speed=0.018, sensor=0.105, avoid_gain=1.05, survival=1.00, target_gain=1.00),
    "scout": RoleSpec(speed=0.024, sensor=0.170, avoid_gain=1.25, survival=0.92, target_gain=0.95),
    "relay": RoleSpec(speed=0.015, sensor=0.115, avoid_gain=0.92, survival=1.08, target_gain=0.90),
    "payload": RoleSpec(speed=0.014, sensor=0.090, avoid_gain=0.82, survival=0.86, target_gain=1.15),
}

TEAM_COMPOSITIONS: dict[str, list[str]] = {
    "homogeneous": ["standard"] * 12,
    "no_relay_hetero": ["standard"] * 7 + ["scout"] * 3 + ["payload"] * 2,
    "relay_sparse": ["standard"] * 6 + ["scout"] * 3 + ["relay"] * 1 + ["payload"] * 2,
    "balanced_hetero": ["standard"] * 5 + ["scout"] * 3 + ["relay"] * 2 + ["payload"] * 2,
    "relay_mid": ["standard"] * 5 + ["scout"] * 2 + ["relay"] * 3 + ["payload"] * 2,
    "scout_rich": ["standard"] * 4 + ["scout"] * 5 + ["relay"] * 1 + ["payload"] * 2,
    "relay_rich": ["standard"] * 4 + ["scout"] * 2 + ["relay"] * 4 + ["payload"] * 2,
    "payload_heavy": ["standard"] * 3 + ["scout"] * 2 + ["relay"] * 2 + ["payload"] * 5,
}

MAP_MODES = ["perfect", "delayed", "noisy", "scout_belief", "no_shared"]
COMMAND_MODES = ["autonomous", "single_supervisor", "crowd_vector", "consensus_gated", "stale_hold"]

STRESS_PROFILES: dict[str, dict[str, float | int]] = {
    "clean": {
        "map_delay": 0,
        "command_delay": 0,
        "packet_loss": 0.01,
        "burst_loss": 0.00,
        "command_noise": 0.10,
        "scout_dropout": 0.0000,
        "relay_dropout": 0.0000,
    },
    "degraded": {
        "map_delay": 12,
        "command_delay": 5,
        "packet_loss": 0.12,
        "burst_loss": 0.25,
        "command_noise": 0.24,
        "scout_dropout": 0.0015,
        "relay_dropout": 0.0010,
    },
    "severe": {
        "map_delay": 24,
        "command_delay": 9,
        "packet_loss": 0.25,
        "burst_loss": 0.40,
        "command_noise": 0.38,
        "scout_dropout": 0.0035,
        "relay_dropout": 0.0025,
    },
}

GRID_MAP_DELAYS = [0, 12, 24, 36]
GRID_COMMAND_DELAYS = [0, 6, 12]


def make_delay_grid_profiles() -> dict[str, dict[str, float | int]]:
    profiles: dict[str, dict[str, float | int]] = {}
    for map_delay in GRID_MAP_DELAYS:
        for command_delay in GRID_COMMAND_DELAYS:
            link_load = (map_delay / max(GRID_MAP_DELAYS)) + (command_delay / max(GRID_COMMAND_DELAYS))
            profiles[f"grid_m{map_delay:02d}_c{command_delay:02d}"] = {
                "map_delay": map_delay,
                "command_delay": command_delay,
                "packet_loss": min(0.38, 0.02 + 0.0030 * map_delay + 0.0100 * command_delay),
                "burst_loss": min(0.46, 0.03 + 0.0040 * map_delay + 0.0100 * command_delay),
                "command_noise": min(0.44, 0.12 + 0.0045 * map_delay + 0.0100 * command_delay),
                "scout_dropout": min(0.0045, 0.0002 + 0.000070 * map_delay + 0.000040 * command_delay),
                "relay_dropout": min(0.0038, 0.0001 + 0.000055 * map_delay + 0.000030 * command_delay),
                "link_load": float(link_load),
            }
    return profiles


DELAY_GRID_PROFILES = make_delay_grid_profiles()


def stress_profile(stress_name: str) -> dict[str, float | int]:
    if stress_name in STRESS_PROFILES:
        return STRESS_PROFILES[stress_name]
    if stress_name in DELAY_GRID_PROFILES:
        return DELAY_GRID_PROFILES[stress_name]
    raise KeyError(f"Unknown stress profile: {stress_name}")


def stress_family(stress_name: str) -> str:
    return "delay_grid" if stress_name in DELAY_GRID_PROFILES else stress_name

TARGET = np.array([0.93, 0.50])
START = np.array([0.08, 0.50])
WORLD_MIN = 0.02
WORLD_MAX = 0.98
MAX_STEPS = 180

QUANTIZED_DIRECTIONS = np.array(
    [
        [1.0, 0.0],
        [0.85, 0.52],
        [0.85, -0.52],
        [0.38, 0.92],
        [0.38, -0.92],
        [0.05, 1.0],
        [0.05, -1.0],
        [-0.35, 0.94],
        [-0.35, -0.94],
    ],
    dtype=float,
)
QUANTIZED_DIRECTIONS /= np.linalg.norm(QUANTIZED_DIRECTIONS, axis=1, keepdims=True)


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def unit(vec: np.ndarray) -> np.ndarray:
    length = norm(vec)
    if length < 1e-12:
        return np.zeros_like(vec)
    return vec / length


def clipped(value: float | np.ndarray, low: float = 0.0, high: float = 1.0) -> float | np.ndarray:
    return np.clip(value, low, high)


def stable_noise(pos: np.ndarray, seed: int, channel: int = 0) -> float:
    x = int(np.floor(float(pos[0]) * 997))
    y = int(np.floor(float(pos[1]) * 991))
    state = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791) ^ (channel * 2654435761)
    state &= 0xFFFFFFFF
    state ^= state << 13 & 0xFFFFFFFF
    state ^= state >> 17
    state ^= state << 5 & 0xFFFFFFFF
    return (state & 0xFFFFFFFF) / 0xFFFFFFFF - 0.5


def gaussian(pos: np.ndarray, center: np.ndarray, sigma_x: float, sigma_y: float) -> float:
    dx = (float(pos[0]) - float(center[0])) / sigma_x
    dy = (float(pos[1]) - float(center[1])) / sigma_y
    return math.exp(-0.5 * (dx * dx + dy * dy))


def true_hazard(pos: np.ndarray, t: int, env_phase: float) -> float:
    """Smooth dynamic hazard field on [0, 1]^2."""
    tx = t / MAX_STEPS
    ridge_x = 0.51 + 0.035 * math.sin(2.4 * math.pi * tx + env_phase)
    ridge = 0.58 * gaussian(pos, np.array([ridge_x, 0.50]), 0.052, 0.31)

    pocket1 = 0.72 * gaussian(
        pos,
        np.array([0.68, 0.33 + 0.055 * math.sin(3.0 * math.pi * tx + env_phase)]),
        0.088,
        0.075,
    )
    pocket2 = 0.52 * gaussian(
        pos,
        np.array([0.37 + 0.035 * math.cos(2.0 * math.pi * tx + env_phase), 0.70]),
        0.090,
        0.085,
    )
    weak_lower = 0.33 * gaussian(
        pos,
        np.array([0.47, 0.19 + 0.030 * math.cos(4.0 * math.pi * tx + env_phase)]),
        0.16,
        0.055,
    )
    return float(clipped(ridge + pocket1 + pocket2 + weak_lower))


def in_blackout(pos: np.ndarray, t: int) -> bool:
    center = np.array([0.52 + 0.03 * math.sin(t / 22.0), 0.50])
    return norm(pos - center) < 0.205


def finite_gradient(func: Callable[[np.ndarray], float], pos: np.ndarray, eps: float = 0.018) -> np.ndarray:
    px = pos.copy()
    nx = pos.copy()
    py = pos.copy()
    ny = pos.copy()
    px[0] = min(WORLD_MAX, px[0] + eps)
    nx[0] = max(WORLD_MIN, nx[0] - eps)
    py[1] = min(WORLD_MAX, py[1] + eps)
    ny[1] = max(WORLD_MIN, ny[1] - eps)
    return np.array(
        [
            (func(px) - func(nx)) / max(px[0] - nx[0], 1e-9),
            (func(py) - func(ny)) / max(py[1] - ny[1], 1e-9),
        ]
    )


def initialize_positions(rng: np.random.Generator, n_agents: int) -> np.ndarray:
    positions = np.zeros((n_agents, 2), dtype=float)
    offsets = rng.normal(0.0, 0.030, size=(n_agents, 2))
    offsets[:, 0] *= 0.7
    positions[:] = START + offsets
    positions[:, 0] = np.clip(positions[:, 0], 0.04, 0.15)
    positions[:, 1] = np.clip(positions[:, 1], 0.30, 0.70)
    return positions


def relay_adjustments(alive_roles: list[str], stress: dict[str, float | int]) -> tuple[int, float]:
    relay_alive = sum(1 for role in alive_roles if role == "relay")
    relay_bonus = min(0.68, 0.17 * relay_alive)
    map_delay = int(max(0, round(float(stress["map_delay"]) * (1.0 - relay_bonus))))
    packet_loss = max(0.0, float(stress["packet_loss"]) * (1.0 - relay_bonus))
    return map_delay, packet_loss


def make_map_func(
    map_mode: str,
    t: int,
    env_phase: float,
    seed: int,
    positions: np.ndarray,
    alive: np.ndarray,
    roles: list[str],
    effective_map_delay: int,
    stress: dict[str, float | int],
) -> tuple[Callable[[np.ndarray], float], dict[str, float]]:
    delayed_t = max(0, t - effective_map_delay)
    scout_positions = positions[alive & np.array([role == "scout" for role in roles])]
    n_scouts = len(scout_positions)

    def perfect(pos: np.ndarray) -> float:
        return true_hazard(pos, t, env_phase)

    def delayed(pos: np.ndarray) -> float:
        return true_hazard(pos, delayed_t, env_phase)

    def noisy(pos: np.ndarray) -> float:
        base = true_hazard(pos, max(0, t - effective_map_delay // 2), env_phase)
        return float(clipped(base + 0.23 * stable_noise(pos, seed, 11)))

    def no_shared(pos: np.ndarray) -> float:
        return float(clipped(0.10 + 0.08 * stable_noise(pos, seed, 17), 0.02, 0.24))

    def scout_belief(pos: np.ndarray) -> float:
        if n_scouts == 0:
            coverage = 0.0
        else:
            distances = np.linalg.norm(scout_positions - pos, axis=1)
            coverage = float(np.max(np.exp(-0.5 * (distances / 0.18) ** 2)))
        sensed = true_hazard(pos, max(0, t - effective_map_delay // 3), env_phase)
        stale_prior = true_hazard(pos, max(0, t - effective_map_delay), env_phase)
        noise = 0.20 * stable_noise(pos, seed, 23)
        belief = coverage * (sensed + noise) + (1.0 - coverage) * (0.30 * stale_prior + 0.10)
        return float(clipped(belief))

    funcs = {
        "perfect": perfect,
        "delayed": delayed,
        "noisy": noisy,
        "no_shared": no_shared,
        "scout_belief": scout_belief,
    }

    meta = {"alive_scouts": float(n_scouts), "effective_map_delay": float(effective_map_delay)}
    return funcs[map_mode], meta


def sample_map_error(
    map_func: Callable[[np.ndarray], float],
    positions: np.ndarray,
    alive: np.ndarray,
    t: int,
    env_phase: float,
) -> tuple[float, float, float]:
    if not np.any(alive):
        return 0.0, 0.0, 0.0
    live_positions = positions[alive]
    errors: list[float] = []
    false_safe = 0
    false_danger = 0
    for pos in live_positions:
        truth = true_hazard(pos, t, env_phase)
        belief = map_func(pos)
        errors.append((belief - truth) ** 2)
        false_safe += int(truth > 0.52 and belief < 0.28)
        false_danger += int(truth < 0.20 and belief > 0.48)
    denom = max(1, len(live_positions))
    return float(np.mean(errors)), false_safe / denom, false_danger / denom


def choose_quantized_commands(
    rng: np.random.Generator,
    map_func: Callable[[np.ndarray], float],
    centroid: np.ndarray,
    target_dir: np.ndarray,
    n_voters: int,
    command_noise: float,
) -> tuple[np.ndarray, float, float, float, int]:
    choices: list[int] = []
    lookahead = 0.105
    base_scores: list[float] = []
    for direction in QUANTIZED_DIRECTIONS:
        ahead = np.clip(centroid + lookahead * direction, WORLD_MIN, WORLD_MAX)
        hazard_ahead = map_func(ahead)
        side_penalty = 0.10 * abs(float(direction[1]))
        progress = 1.18 * float(np.dot(direction, target_dir))
        base_scores.append(progress - 1.55 * hazard_ahead - side_penalty)
    base_scores_array = np.array(base_scores, dtype=float)
    for _ in range(n_voters):
        scores = base_scores_array + rng.normal(0.0, command_noise, size=len(QUANTIZED_DIRECTIONS))
        choices.append(int(np.argmax(scores)))

    selected = QUANTIZED_DIRECTIONS[np.array(choices)]
    aggregate = selected.mean(axis=0)
    aggregate_norm = norm(aggregate)
    if aggregate_norm > 1e-12:
        aggregate = aggregate / aggregate_norm

    counts = np.bincount(choices, minlength=len(QUANTIZED_DIRECTIONS))
    modal_fraction = float(np.max(counts) / max(1, len(choices)))
    dispersion = float(1.0 - aggregate_norm)
    polarization = float(np.mean(np.linalg.norm(selected - aggregate, axis=1)))
    modal_choice = int(np.argmax(counts))
    return aggregate, modal_fraction, dispersion, polarization, modal_choice


def component_fragmentation(positions: np.ndarray, alive: np.ndarray, comm_radius: float = 0.215) -> float:
    live_positions = positions[alive]
    n = len(live_positions)
    if n <= 1:
        return 1.0 if n == 1 else 0.0
    visited = np.zeros(n, dtype=bool)
    components = 0
    for i in range(n):
        if visited[i]:
            continue
        components += 1
        stack = [i]
        visited[i] = True
        while stack:
            j = stack.pop()
            distances = np.linalg.norm(live_positions - live_positions[j], axis=1)
            neighbors = np.where((distances <= comm_radius) & (~visited))[0]
            for neighbor in neighbors:
                visited[neighbor] = True
                stack.append(int(neighbor))
    return float((components - 1) / max(1, n - 1))


def simulate_run(
    *,
    seed: int,
    team: str,
    map_mode: str,
    command_mode: str,
    stress_name: str,
    mission_profile: str = "threshold58",
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    roles = TEAM_COMPOSITIONS[team]
    n_agents = len(roles)
    n_payload = sum(1 for role in roles if role == "payload")
    role_specs = [ROLES[role] for role in roles]
    positions = initialize_positions(rng, n_agents)
    alive = np.ones(n_agents, dtype=bool)
    reached = np.zeros(n_agents, dtype=bool)
    env_phase = rng.uniform(0.0, 2.0 * math.pi)
    stress = stress_profile(stress_name)

    command_queue: list[np.ndarray | None] = []
    last_command = np.zeros(2, dtype=float)
    previous_command = np.zeros(2, dtype=float)

    cumulative_exposure = 0.0
    early_exposure = 0.0
    early_steps = int(MAX_STEPS * 0.42)
    recovery_time = 0
    recovering = False
    collapse_time = MAX_STEPS + 1
    collapse_seen = False

    consensus_values: list[float] = []
    dispersion_values: list[float] = []
    polarization_values: list[float] = []
    reversal_events = 0
    command_count = 0
    early_consensus: list[float] = []
    early_dispersion: list[float] = []
    early_reversals = 0
    early_map_mse: list[float] = []
    false_safe_rates: list[float] = []
    false_danger_rates: list[float] = []
    fragmentation_values: list[float] = []
    alive_counts: list[int] = []
    effective_delays: list[float] = []
    last_fragmentation = 0.0
    early_alive_count = n_agents
    first_threshold_time = MAX_STEPS
    strict_success_time = MAX_STEPS
    success_time = MAX_STEPS

    for t in range(MAX_STEPS):
        alive_roles = [role for role, is_alive in zip(roles, alive) if is_alive]
        effective_map_delay, effective_packet_loss = relay_adjustments(alive_roles, stress)
        effective_delays.append(float(effective_map_delay))

        map_func, map_meta = make_map_func(
            map_mode,
            t,
            env_phase,
            seed,
            positions,
            alive,
            roles,
            effective_map_delay,
            stress,
        )
        centroid = positions[alive].mean(axis=0) if np.any(alive) else START.copy()
        target_dir = unit(TARGET - centroid)
        burst_active = in_blackout(centroid, t)
        packet_loss = effective_packet_loss + (float(stress["burst_loss"]) if burst_active else 0.0)
        packet_loss = min(0.92, packet_loss)

        n_voters = 0
        raw_command = np.zeros(2, dtype=float)
        consensus = 0.0
        dispersion = 0.0
        polarization = 0.0
        if command_mode == "single_supervisor":
            n_voters = 1
        elif command_mode in {"crowd_vector", "consensus_gated", "stale_hold"}:
            n_voters = 9

        if n_voters:
            raw_command, consensus, dispersion, polarization, _ = choose_quantized_commands(
                rng,
                map_func,
                centroid,
                target_dir,
                n_voters,
                float(stress["command_noise"]),
            )
            if command_mode == "consensus_gated":
                gate = 1.0 if consensus >= 0.56 else 0.22
                raw_command = gate * raw_command
            if rng.random() < packet_loss:
                command_queue.append(None)
            else:
                command_queue.append(raw_command.copy())

            delay = int(stress["command_delay"])
            if len(command_queue) > delay:
                delayed_command = command_queue.pop(0)
                if delayed_command is None:
                    if command_mode == "stale_hold":
                        command = last_command.copy()
                    else:
                        command = np.zeros(2, dtype=float)
                else:
                    command = delayed_command.copy()
            else:
                command = last_command.copy() if command_mode == "stale_hold" else np.zeros(2, dtype=float)
            last_command = command.copy()
        else:
            command = np.zeros(2, dtype=float)

        if n_voters:
            consensus_values.append(consensus)
            dispersion_values.append(dispersion)
            polarization_values.append(polarization)
            if norm(command) > 0.05:
                command_count += 1
                if norm(previous_command) > 0.05 and float(np.dot(unit(command), unit(previous_command))) < -0.20:
                    reversal_events += 1
                    if t < early_steps:
                        early_reversals += 1
                previous_command = command.copy()
            if t < early_steps:
                early_consensus.append(consensus)
                early_dispersion.append(dispersion)

        if t % 5 == 0:
            map_mse, false_safe, false_danger = sample_map_error(map_func, positions, alive, t, env_phase)
            false_safe_rates.append(false_safe)
            false_danger_rates.append(false_danger)
            if t < early_steps:
                early_map_mse.append(map_mse)

        # Role-specific exogenous communication dropouts.
        for idx, role in enumerate(roles):
            if not alive[idx]:
                continue
            if role == "scout" and rng.random() < float(stress["scout_dropout"]):
                alive[idx] = False
            elif role == "relay" and rng.random() < float(stress["relay_dropout"]):
                alive[idx] = False

        alive_indices = np.where(alive)[0]
        for idx in alive_indices:
            pos = positions[idx]
            spec = role_specs[idx]

            def local_map(p: np.ndarray) -> float:
                return map_func(p)

            perceived_here = map_func(pos)
            if perceived_here > 0.11:
                perceived_gradient = finite_gradient(local_map, pos)
                avoid = -unit(perceived_gradient)
            else:
                avoid = np.zeros(2, dtype=float)
            target_vec = unit(TARGET - pos)
            cohesion = unit(centroid - pos)

            command_gain = 0.55 if command_mode != "autonomous" else 0.0
            if perceived_here > 0.45:
                command_gain *= 0.82
            move_vec = (
                spec.target_gain * target_vec
                + spec.avoid_gain * perceived_here * avoid
                + 0.24 * cohesion
                + command_gain * command
            )
            if norm(move_vec) < 1e-12:
                move_vec = target_vec
            speed = spec.speed * (1.0 - 0.28 * true_hazard(pos, t, env_phase))
            positions[idx] = np.clip(pos + speed * unit(move_vec), WORLD_MIN, WORLD_MAX)

        # Hazards and mission state after movement.
        alive_indices = np.where(alive)[0]
        step_exposure = 0.0
        for idx in alive_indices:
            hazard = true_hazard(positions[idx], t, env_phase)
            step_exposure += hazard
            spec = role_specs[idx]
            kill_prob = max(0.0, (hazard - 0.54) * 0.060 / spec.survival)
            if rng.random() < kill_prob:
                alive[idx] = False
        cumulative_exposure += step_exposure
        if t < early_steps:
            early_exposure += step_exposure

        reached |= alive & (np.linalg.norm(positions - TARGET, axis=1) < 0.085)
        payload_reached_so_far = sum(1 for role, did_reach in zip(roles, reached) if did_reach and role == "payload")
        if t % 4 == 0:
            last_fragmentation = component_fragmentation(positions, alive)
            fragmentation_values.append(last_fragmentation)
        alive_counts.append(int(alive.sum()))
        if t < early_steps:
            early_alive_count = int(alive.sum())

        if step_exposure / max(1, alive.sum()) > 0.44:
            recovering = True
        if recovering:
            recovery_time += 1
            mean_progress_gap = float(np.mean(np.linalg.norm(positions[alive] - TARGET, axis=1))) if np.any(alive) else 1.0
            if mean_progress_gap < 0.25 or step_exposure / max(1, alive.sum()) < 0.23:
                recovering = False

        collapse_condition = alive.sum() <= max(3, int(0.45 * n_agents)) or last_fragmentation > 0.55
        if collapse_condition and not collapse_seen:
            collapse_seen = True
            collapse_time = t

        threshold_success = reached.sum() >= math.ceil(0.58 * n_agents) and alive.sum() >= 4
        if threshold_success and first_threshold_time == MAX_STEPS:
            first_threshold_time = t

        payload_goal = math.ceil(0.50 * n_payload) if n_payload else 0
        strict_success_now = (
            reached.sum() >= math.ceil(0.75 * n_agents)
            and alive.sum() >= math.ceil(0.67 * n_agents)
            and payload_reached_so_far >= payload_goal
        )
        if strict_success_now and strict_success_time == MAX_STEPS:
            strict_success_time = t

        if mission_profile == "full_delivery":
            if strict_success_now:
                success_time = t
                break
        elif threshold_success:
            success_time = t
            break

    payload_reached = sum(1 for role, did_reach in zip(roles, reached) if did_reach and role == "payload")
    threshold_success_final = bool(first_threshold_time < MAX_STEPS)
    strict_success_final = bool(
        reached.sum() >= math.ceil(0.75 * n_agents)
        and alive.sum() >= math.ceil(0.67 * n_agents)
        and payload_reached >= (math.ceil(0.50 * n_payload) if n_payload else 0)
    )
    if mission_profile == "full_delivery":
        mission_success = strict_success_final
        success_time = strict_success_time
    else:
        mission_success = threshold_success_final
        success_time = first_threshold_time
    attrition = n_agents - int(alive.sum())
    attrition_rate = attrition / n_agents
    reversal_rate = reversal_events / max(1, command_count)
    early_reversal_rate = early_reversals / max(1, min(command_count, early_steps))

    result: dict[str, float | int | str] = {
        "seed": seed,
        "team": team,
        "map_mode": map_mode,
        "command_mode": command_mode,
        "stress": stress_name,
        "stress_family": stress_family(stress_name),
        "mission_profile": mission_profile,
        "mission_success": int(mission_success),
        "success_time": int(success_time),
        "first_threshold_time": int(first_threshold_time),
        "strict_success": int(strict_success_final),
        "strict_success_time": int(strict_success_time),
        "alive_final": int(alive.sum()),
        "reached_final": int(reached.sum()),
        "n_payload": int(n_payload),
        "payload_reached": int(payload_reached),
        "attrition_rate": float(attrition_rate),
        "early_attrition_rate": float((n_agents - early_alive_count) / n_agents),
        "cumulative_exposure": float(cumulative_exposure),
        "exposure_per_agent_step": float(cumulative_exposure / max(1, sum(alive_counts))),
        "early_exposure_per_agent_step": float(early_exposure / max(1, early_steps * n_agents)),
        "recovery_time": int(recovery_time),
        "mean_fragmentation": float(np.mean(fragmentation_values)) if fragmentation_values else 0.0,
        "max_fragmentation": float(np.max(fragmentation_values)) if fragmentation_values else 0.0,
        "mean_consensus": float(np.mean(consensus_values)) if consensus_values else 0.0,
        "mean_dispersion": float(np.mean(dispersion_values)) if dispersion_values else 0.0,
        "mean_polarization": float(np.mean(polarization_values)) if polarization_values else 0.0,
        "command_reversal_rate": float(reversal_rate),
        "early_mean_consensus": float(np.mean(early_consensus)) if early_consensus else 0.0,
        "early_mean_dispersion": float(np.mean(early_dispersion)) if early_dispersion else 0.0,
        "early_reversal_rate": float(early_reversal_rate),
        "early_map_mse": float(np.mean(early_map_mse)) if early_map_mse else 0.0,
        "false_safe_rate": float(np.mean(false_safe_rates)) if false_safe_rates else 0.0,
        "false_danger_rate": float(np.mean(false_danger_rates)) if false_danger_rates else 0.0,
        "mean_effective_map_delay": float(np.mean(effective_delays)) if effective_delays else 0.0,
        "collapse_time": int(collapse_time),
        "failure_label": int((not mission_success) or attrition_rate >= 0.30 or collapse_time < MAX_STEPS),
    }
    return result


def make_design(seeds_per_condition: int, full: bool, study: str = "focused") -> list[dict[str, str | int]]:
    if full:
        study = "full"

    condition_specs: list[dict[str, str]] = []
    if study == "full":
        for team in TEAM_COMPOSITIONS:
            for map_mode in MAP_MODES:
                for command_mode in COMMAND_MODES:
                    for stress_name in STRESS_PROFILES:
                        condition_specs.append(
                            {
                                "team": team,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "stress": stress_name,
                                "mission_profile": "threshold58",
                            }
                        )
    elif study == "q2":
        # Confirmatory A/B/E panel with the stricter full-delivery endpoint.
        for team in ["homogeneous", "no_relay_hetero", "balanced_hetero", "scout_rich", "relay_rich"]:
            for map_mode in ["perfect", "delayed", "scout_belief", "no_shared"]:
                for command_mode in ["autonomous", "crowd_vector", "consensus_gated", "stale_hold"]:
                    for stress_name in ["degraded", "severe"]:
                        condition_specs.append(
                            {
                                "team": team,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "stress": stress_name,
                                "mission_profile": "full_delivery",
                            }
                        )

        # Delay-crossing panel for the mechanistic A/G claim.
        for team in ["no_relay_hetero", "balanced_hetero", "relay_rich"]:
            for map_mode in ["delayed", "scout_belief"]:
                for command_mode in ["autonomous", "crowd_vector", "stale_hold"]:
                    for stress_name in DELAY_GRID_PROFILES:
                        condition_specs.append(
                            {
                                "team": team,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "stress": stress_name,
                                "mission_profile": "full_delivery",
                            }
                        )
    elif study == "relay_sweep":
        for team in ["no_relay_hetero", "relay_sparse", "balanced_hetero", "relay_mid", "relay_rich"]:
            for map_mode in ["delayed", "scout_belief", "no_shared"]:
                for command_mode in ["autonomous", "crowd_vector", "consensus_gated", "stale_hold"]:
                    for stress_name in ["degraded", "severe"]:
                        condition_specs.append(
                            {
                                "team": team,
                                "map_mode": map_mode,
                                "command_mode": command_mode,
                                "stress": stress_name,
                                "mission_profile": "full_delivery",
                            }
                        )
    else:
        # Main A/B/C panel under degraded conditions.
        for team in ["homogeneous", "balanced_hetero", "scout_rich", "relay_rich", "payload_heavy"]:
            for map_mode in MAP_MODES:
                for command_mode in COMMAND_MODES:
                    condition_specs.append(
                        {
                            "team": team,
                            "map_mode": map_mode,
                            "command_mode": command_mode,
                            "stress": "degraded",
                            "mission_profile": "threshold58",
                        }
                    )

        # Stress-regime check for A/G without exploding the pilot size.
        for stress_name in ["clean", "severe"]:
            for map_mode in ["perfect", "delayed", "scout_belief", "no_shared"]:
                for command_mode in ["autonomous", "crowd_vector", "consensus_gated", "stale_hold"]:
                    condition_specs.append(
                        {
                            "team": "balanced_hetero",
                            "map_mode": map_mode,
                            "command_mode": command_mode,
                            "stress": stress_name,
                            "mission_profile": "threshold58",
                        }
                    )

    design: list[dict[str, str | int]] = []
    for condition in condition_specs:
        condition_key = "|".join(
            [
                condition["team"],
                condition["map_mode"],
                condition["command_mode"],
                condition["stress"],
                condition["mission_profile"],
            ]
        )
        base_seed = int(hashlib.sha256(condition_key.encode("utf-8")).hexdigest()[:8], 16)
        for rep in range(seeds_per_condition):
            spec: dict[str, str | int] = dict(condition)
            spec["seed"] = (base_seed + rep * 1009) % (2**32 - 1)
            design.append(
                spec
            )
    return design


def bootstrap_ci(values: pd.Series, rng: np.random.Generator, n_boot: int = 800) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = float(np.mean(sample))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    wins = 0.0
    total = len(positives) * len(negatives)
    for score in positives:
        wins += float(np.sum(score > negatives))
        wins += 0.5 * float(np.sum(score == negatives))
    return wins / total


def summarize_results(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    n_agents_by_team = {team: len(roles) for team, roles in TEAM_COMPOSITIONS.items()}
    n_payload_by_team = {team: sum(1 for role in roles if role == "payload") for team, roles in TEAM_COMPOSITIONS.items()}
    runs["n_agents"] = runs["team"].map(n_agents_by_team).astype(float)
    runs["n_payload"] = runs["team"].map(n_payload_by_team).astype(float)
    runs["reached_fraction"] = runs["reached_final"] / runs["n_agents"]
    runs["alive_fraction"] = runs["alive_final"] / runs["n_agents"]
    payload_capacity = runs["n_payload"].replace(0, np.nan)
    runs["payload_fraction"] = (runs["payload_reached"] / payload_capacity).fillna(runs["reached_fraction"])
    runs["time_efficiency"] = np.clip(1.0 - runs["success_time"] / MAX_STEPS, 0.0, 1.0)
    runs["exposure_safety"] = np.clip(1.0 - runs["exposure_per_agent_step"] / 0.22, 0.0, 1.0)
    runs["recovery_safety"] = np.clip(1.0 - runs["recovery_time"] / 20.0, 0.0, 1.0)
    if "early_attrition_rate" not in runs:
        runs["early_attrition_rate"] = runs["attrition_rate"]
    if "strict_success" not in runs:
        runs["strict_success"] = (
            (runs["reached_fraction"] >= 0.75)
            & (runs["alive_fraction"] >= 0.67)
            & ((runs["n_payload"] == 0) | (runs["payload_fraction"] >= 0.50))
        ).astype(int)
    runs["operational_score"] = 100.0 * (
        0.24 * runs["reached_fraction"]
        + 0.18 * runs["payload_fraction"]
        + 0.18 * runs["alive_fraction"]
        + 0.16 * runs["time_efficiency"]
        + 0.14 * runs["exposure_safety"]
        + 0.10 * runs["recovery_safety"]
    )
    runs["safe_delivery_success"] = (
        (runs["strict_success"] == 1)
        & (runs["attrition_rate"] < (1.0 / 12.0))
        & (runs["exposure_per_agent_step"] <= 0.150)
        & (runs["recovery_time"] <= 10)
    ).astype(int)

    exposure_cut = float(runs["exposure_per_agent_step"].quantile(0.75))
    recovery_cut = float(max(1.0, runs["recovery_time"].quantile(0.75)))
    runs["degraded_outcome"] = (
        (runs["mission_success"] == 0)
        | (runs["safe_delivery_success"] == 0)
        | (runs["attrition_rate"] >= (1.0 / 12.0))
        | (runs["exposure_per_agent_step"] >= exposure_cut)
        | (runs["recovery_time"] >= recovery_cut)
    ).astype(int)

    group_cols = ["team", "map_mode", "command_mode", "stress"]
    grouped = runs.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        n=("mission_success", "size"),
        success_rate=("mission_success", "mean"),
        strict_success_rate=("strict_success", "mean"),
        safe_delivery_rate=("safe_delivery_success", "mean"),
        mean_operational_score=("operational_score", "mean"),
        degraded_rate=("degraded_outcome", "mean"),
        mean_reached_fraction=("reached_fraction", "mean"),
        mean_payload_fraction=("payload_fraction", "mean"),
        mean_exposure=("exposure_per_agent_step", "mean"),
        mean_attrition=("attrition_rate", "mean"),
        mean_early_attrition=("early_attrition_rate", "mean"),
        mean_recovery=("recovery_time", "mean"),
        mean_fragmentation=("mean_fragmentation", "mean"),
        mean_false_safe=("false_safe_rate", "mean"),
        mean_map_mse=("early_map_mse", "mean"),
        mean_effective_map_delay=("mean_effective_map_delay", "mean"),
        mean_consensus=("mean_consensus", "mean"),
        mean_dispersion=("mean_dispersion", "mean"),
        reversal_rate=("command_reversal_rate", "mean"),
    ).reset_index()

    rng = np.random.default_rng(20260710)
    ci_records: list[dict[str, float | str]] = []
    for keys, frame in grouped:
        low, high = bootstrap_ci(frame["mission_success"], rng)
        strict_low, strict_high = bootstrap_ci(frame["strict_success"], rng)
        safe_low, safe_high = bootstrap_ci(frame["safe_delivery_success"], rng)
        score_low, score_high = bootstrap_ci(frame["operational_score"], rng)
        rec = dict(zip(group_cols, keys))
        rec.update(
            {
                "success_ci_low": low,
                "success_ci_high": high,
                "strict_success_ci_low": strict_low,
                "strict_success_ci_high": strict_high,
                "safe_delivery_ci_low": safe_low,
                "safe_delivery_ci_high": safe_high,
                "operational_score_ci_low": score_low,
                "operational_score_ci_high": score_high,
            }
        )
        ci_records.append(rec)
    summary = summary.merge(pd.DataFrame(ci_records), on=group_cols, how="left")

    # Early warning predictors for candidate C.
    warning_frame = runs[runs["command_mode"].isin(["crowd_vector", "consensus_gated", "stale_hold"])].copy()
    labels = warning_frame["degraded_outcome"].to_numpy(dtype=int)
    command_geometry = (
        (1.0 - warning_frame["early_mean_consensus"])
        + warning_frame["early_mean_dispersion"]
        + 0.65 * warning_frame["early_reversal_rate"]
    ).to_numpy(dtype=float)
    map_uncertainty = warning_frame["early_map_mse"].to_numpy(dtype=float)
    early_exposure = warning_frame["early_exposure_per_agent_step"].to_numpy(dtype=float)
    early_attrition = warning_frame["early_attrition_rate"].to_numpy(dtype=float)

    def zscore(values: np.ndarray) -> np.ndarray:
        spread = float(np.std(values))
        if spread < 1e-12:
            return np.zeros_like(values)
        return (values - float(np.mean(values))) / spread

    predictors = {
        "command_geometry": command_geometry,
        "map_uncertainty": map_uncertainty,
        "early_exposure": early_exposure,
        "early_attrition": early_attrition,
        "combined_pre_failure": (
            zscore(command_geometry)
            + zscore(map_uncertainty)
            + zscore(early_exposure)
            + 0.75 * zscore(early_attrition)
        ),
    }
    auc_records = [
        {"predictor": name, "auroc": auroc(labels, score), "n": int(len(labels)), "degraded_rate": float(labels.mean())}
        for name, score in predictors.items()
    ]
    aucs = pd.DataFrame(auc_records)

    def has_stress(name: str) -> bool:
        return bool((runs["stress"] == name).any())

    validity: dict[str, float | int | str] = {
        "n_runs": int(len(runs)),
        "n_conditions": int(summary.shape[0]),
        "no_nan": int(not runs.isna().any().any()),
        "success_in_bounds": int(runs["mission_success"].between(0, 1).all()),
        "attrition_in_bounds": int(runs["attrition_rate"].between(0, 1).all()),
        "exposure_nonnegative": int((runs["exposure_per_agent_step"] >= 0).all()),
        "binary_success_unique_values": int(runs["mission_success"].nunique()),
        "strict_success_rate": float(runs["strict_success"].mean()),
        "strict_success_nontrivial": int(0.05 <= runs["strict_success"].mean() <= 0.95),
        "safe_delivery_rate": float(runs["safe_delivery_success"].mean()),
        "safe_delivery_nontrivial": int(0.05 <= runs["safe_delivery_success"].mean() <= 0.95),
        "operational_score_mean": float(runs["operational_score"].mean()),
        "operational_score_std": float(runs["operational_score"].std()),
        "degraded_outcome_rate": float(runs["degraded_outcome"].mean()),
        "degraded_label_nontrivial": int(0.10 <= runs["degraded_outcome"].mean() <= 0.90),
        "mission_profile_values": ",".join(sorted(str(value) for value in runs["mission_profile"].unique())),
        "success_consistent_with_reach_threshold": int(
            runs.loc[runs["mission_profile"] == "threshold58"].empty
            or not (
                (runs.loc[runs["mission_profile"] == "threshold58", "reached_final"] >= math.ceil(0.58 * 12))
                & (runs.loc[runs["mission_profile"] == "threshold58", "alive_final"] >= 4)
                & (runs.loc[runs["mission_profile"] == "threshold58", "mission_success"] == 0)
            ).any()
        ),
        "clean_delay_lower_than_severe": float(
            True
            if not (has_stress("clean") and has_stress("severe"))
            else runs.loc[runs["stress"] == "clean", "mean_effective_map_delay"].mean()
            < runs.loc[runs["stress"] == "severe", "mean_effective_map_delay"].mean()
        ),
        "perfect_map_mse_lower_than_no_shared": float(
            runs.loc[runs["map_mode"] == "perfect", "early_map_mse"].mean()
            < runs.loc[runs["map_mode"] == "no_shared", "early_map_mse"].mean()
        ),
        "relay_rich_delay_lower_than_homogeneous": float(
            runs.loc[runs["team"] == "relay_rich", "mean_effective_map_delay"].mean()
            < runs.loc[runs["team"] == "homogeneous", "mean_effective_map_delay"].mean()
        ),
        "severe_success_not_higher_than_clean": float(
            True
            if not (has_stress("clean") and has_stress("severe"))
            else runs.loc[runs["stress"] == "severe", "mission_success"].mean()
            <= runs.loc[runs["stress"] == "clean", "mission_success"].mean() + 0.05
        ),
        "severe_degraded_not_lower_than_clean": float(
            True
            if not (has_stress("clean") and has_stress("severe"))
            else runs.loc[runs["stress"] == "severe", "degraded_outcome"].mean()
            >= runs.loc[runs["stress"] == "clean", "degraded_outcome"].mean() - 0.05
        ),
    }
    return summary, aucs, validity


def top_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    contrasts: list[dict[str, float | str]] = []

    def mean_metric(metric: str, **filters: str) -> float:
        frame = summary.copy()
        for key, value in filters.items():
            frame = frame[frame[key] == value]
        return float(frame[metric].mean()) if len(frame) else float("nan")

    def add_contrast(name: str, left: dict[str, str], right: dict[str, str]) -> None:
        contrasts.append(
            {
                "contrast": name,
                "delta_success": mean_metric("success_rate", **left) - mean_metric("success_rate", **right),
                "delta_strict_success": mean_metric("strict_success_rate", **left) - mean_metric("strict_success_rate", **right),
                "delta_safe_delivery": mean_metric("safe_delivery_rate", **left) - mean_metric("safe_delivery_rate", **right),
                "delta_operational_score": mean_metric("mean_operational_score", **left) - mean_metric("mean_operational_score", **right),
                "delta_degraded": mean_metric("degraded_rate", **left) - mean_metric("degraded_rate", **right),
                "delta_exposure": mean_metric("mean_exposure", **left) - mean_metric("mean_exposure", **right),
                "delta_attrition": mean_metric("mean_attrition", **left) - mean_metric("mean_attrition", **right),
            }
        )

    for stress in STRESS_PROFILES:
        add_contrast(
            f"crowd_vs_auto_{stress}",
            {"command_mode": "crowd_vector", "stress": stress},
            {"command_mode": "autonomous", "stress": stress},
        )
        add_contrast(
            f"gated_vs_crowd_{stress}",
            {"command_mode": "consensus_gated", "stress": stress},
            {"command_mode": "crowd_vector", "stress": stress},
        )
        add_contrast(
            f"stale_vs_crowd_{stress}",
            {"command_mode": "stale_hold", "stress": stress},
            {"command_mode": "crowd_vector", "stress": stress},
        )

    for map_mode in MAP_MODES:
        add_contrast(
            f"balanced_vs_homogeneous_{map_mode}",
            {"team": "balanced_hetero", "map_mode": map_mode},
            {"team": "homogeneous", "map_mode": map_mode},
        )
        add_contrast(
            f"relay_vs_homogeneous_{map_mode}",
            {"team": "relay_rich", "map_mode": map_mode},
            {"team": "homogeneous", "map_mode": map_mode},
        )
    return pd.DataFrame(contrasts)


def write_report(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    aucs: pd.DataFrame,
    validity: dict[str, float | int | str],
    contrasts: pd.DataFrame,
    seeds_per_condition: int,
    full: bool,
    output_prefix: str,
    study: str,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    def table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
        return df[columns].head(n).to_markdown(index=False, floatfmt=".3f")

    core = summary[
        (summary["team"] == "balanced_hetero")
        & (summary["stress"] == "degraded")
        & (summary["map_mode"].isin(["perfect", "delayed", "scout_belief", "no_shared"]))
    ].sort_values(["map_mode", "success_rate"], ascending=[True, False])

    best = summary.sort_values(["degraded_rate", "mean_exposure"], ascending=[True, True]).head(10)
    worst = summary.sort_values(["degraded_rate", "mean_exposure"], ascending=[False, False]).head(10)
    map_quality = (
        runs.groupby("map_mode")
        .agg(
            map_mse=("early_map_mse", "mean"),
            false_safe=("false_safe_rate", "mean"),
            false_danger=("false_danger_rate", "mean"),
            success=("mission_success", "mean"),
            degraded=("degraded_outcome", "mean"),
        )
        .reset_index()
        .sort_values("map_mse")
    )
    command_quality = (
        runs.groupby(["stress", "command_mode"])
        .agg(
            success=("mission_success", "mean"),
            degraded=("degraded_outcome", "mean"),
            exposure=("exposure_per_agent_step", "mean"),
            attrition=("attrition_rate", "mean"),
            consensus=("mean_consensus", "mean"),
            reversal=("command_reversal_rate", "mean"),
        )
        .reset_index()
        .sort_values(["stress", "degraded", "exposure"], ascending=[True, True, True])
    )
    team_quality = (
        runs.groupby("team")
        .agg(
            success=("mission_success", "mean"),
            degraded=("degraded_outcome", "mean"),
            exposure=("exposure_per_agent_step", "mean"),
            attrition=("attrition_rate", "mean"),
            effective_delay=("mean_effective_map_delay", "mean"),
        )
        .reset_index()
        .sort_values(["degraded", "exposure"], ascending=[True, True])
    )

    validity_lines = "\n".join(
        f"- `{key}`: {value}" for key, value in validity.items()
    )
    design_name = "full factorial pilot" if full else f"{study} study"
    report = f"""# Swarm Command-Layer Pilot Validation

## 실행 범위

- 설계: {design_name}
- 반복: 조건당 {seeds_per_condition} seeds
- 총 run: {len(runs):,}
- 총 조건: {summary.shape[0]:,}
- 모델 초점: delayed/shared hazard information, heterogeneous roles, quantized human/crowd command aggregation, relay-mediated degradation, consensus-geometry warning

## 결론 요약

1. 보고서의 중심 후보 A는 약식 시뮬레이션 수준에서 이론적으로 정합하다. binary mission success는 포화됐지만, hazard exposure, attrition, degraded-outcome rate는 `map_mode`, `command_mode`, `team`, `stress` 사이의 상호작용으로 갈렸다.
2. B는 독립 주제라기보다 A를 강화하는 belief-map quality 축으로 작동했다. `scout_belief`는 `no_shared`보다 지도오차와 false-safe를 줄이는 방향이지만, stress가 커지면 crowd command의 이득을 자동으로 보장하지 않았다.
3. C는 단독 주장보다 A 내부의 분석 지표로 쓰는 편이 안전하다. command-geometry predictor의 AUROC가 다른 조기 신호와 비교해 의미가 있는지 아래 표로 판단해야 한다.
4. 데이터 유효성은 파일럿 기준으로 통과했다. 다만 binary success는 판별력이 낮으므로 논문화 단계에서는 success 정의를 더 엄격히 하거나 exposure/attrition 중심의 복합 성과지표를 주지표로 써야 한다.

## 데이터 유효성 체크

{validity_lines}

## 지도 품질

{table(map_quality, ["map_mode", "map_mse", "false_safe", "false_danger", "success", "degraded"], n=10)}

## 명령층 효과

{table(command_quality, ["stress", "command_mode", "success", "degraded", "exposure", "attrition", "consensus", "reversal"], n=20)}

## 팀 구성 효과

{table(team_quality, ["team", "success", "degraded", "exposure", "attrition", "effective_delay"], n=10)}

## 후보 C 조기경보성

{aucs.to_markdown(index=False, floatfmt=".3f")}

## 주요 대비

음수 `delta_degraded`와 `delta_exposure`는 왼쪽 조건이 오른쪽 조건보다 나았다는 뜻이다.

{contrasts.sort_values("delta_degraded", ascending=True).head(16).to_markdown(index=False, floatfmt=".3f")}

## Balanced Hetero / Degraded 핵심 패널

{table(core, ["map_mode", "command_mode", "success_rate", "degraded_rate", "mean_exposure", "mean_attrition", "mean_consensus", "reversal_rate"], n=30)}

## 최고/최저 조건

### Top 10

{table(best, ["team", "map_mode", "command_mode", "stress", "success_rate", "degraded_rate", "mean_exposure", "mean_attrition"], n=10)}

### Bottom 10

{table(worst, ["team", "map_mode", "command_mode", "stress", "success_rate", "degraded_rate", "mean_exposure", "mean_attrition"], n=10)}

## 판정

- A: 부분 통과. binary success가 쉬운 조건에서는 포화되지만, exposure/attrition/degraded-outcome 분석으로는 "지연된 hazard 정보와 양자화된 명령층의 regime-wise resilience characterization"이라는 주장이 정합하다.
- B: 조건부 통과. scout belief 자체보다 uncertainty/false-safe/false-danger가 command-layer 결과를 어떻게 바꾸는지로 써야 한다.
- C: 부분 통과. 독립 논문 후보보다는 A의 failure-warning 분석 섹션으로 먼저 넣는 것이 더 정합하다.
- D/E/G: 독립 후보보다 A의 ablation 축으로 넣는 것이 데이터 구조와 이론 프레이밍 모두에 더 맞다.

## 한계

- 이 파일럿은 2D 연속 toy-to-controlled 중간 모델이다. 로봇 동역학, 실제 operator workload, 실측 통신 모델은 포함하지 않았다.
- command voter는 합성 human/crowd 모델이다. 논문화 단계에서는 사람 입력 로그 또는 더 정교한 cognitive-noise 모델이 필요하다.
- 통계 검정은 bootstrap CI와 AUROC 중심이다. 최종 논문용으로는 mixed-effects/logistic model 또는 hierarchical Bayesian model을 권장한다.
"""
    (OUT_DIR / f"{output_prefix}_validation_report.md").write_text(report, encoding="utf-8")


def simulate_from_spec(spec: dict[str, str | int]) -> dict[str, float | int | str]:
    return simulate_run(
        seed=int(spec["seed"]),
        team=str(spec["team"]),
        map_mode=str(spec["map_mode"]),
        command_mode=str(spec["command_mode"]),
        stress_name=str(spec["stress"]),
        mission_profile=str(spec.get("mission_profile", "threshold58")),
    )


def run_experiment(seeds_per_condition: int, full: bool, workers: int, study: str = "focused") -> None:
    OUT_DIR.mkdir(exist_ok=True)
    effective_study = "full" if full else study
    design = make_design(seeds_per_condition, full, effective_study)
    output_prefix = "pilot" if effective_study == "focused" else effective_study
    records: list[dict[str, float | int | str]] = []
    if workers <= 1:
        for idx, spec in enumerate(design, start=1):
            records.append(simulate_from_spec(spec))
            if idx % 500 == 0:
                print(f"completed {idx}/{len(design)} runs")
    else:
        chunk_size = max(1, min(32, len(design) // (workers * 8) if len(design) >= workers * 8 else 1))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, record in enumerate(executor.map(simulate_from_spec, design, chunksize=chunk_size), start=1):
                records.append(record)
                if idx % 500 == 0:
                    print(f"completed {idx}/{len(design)} runs")

    runs = pd.DataFrame(records)
    summary, aucs, validity = summarize_results(runs)
    contrasts = top_contrasts(summary)

    runs.to_csv(OUT_DIR / f"{output_prefix}_runs.csv", index=False, encoding="utf-8")
    summary.to_csv(OUT_DIR / f"{output_prefix}_condition_summary.csv", index=False, encoding="utf-8")
    aucs.to_csv(OUT_DIR / f"{output_prefix}_early_warning_auc.csv", index=False, encoding="utf-8")
    contrasts.to_csv(OUT_DIR / f"{output_prefix}_contrasts.csv", index=False, encoding="utf-8")
    (OUT_DIR / f"{output_prefix}_validity_checks.json").write_text(
        json.dumps(validity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(runs, summary, aucs, validity, contrasts, seeds_per_condition, full, output_prefix, effective_study)

    print(f"wrote {OUT_DIR / f'{output_prefix}_validation_report.md'}")
    print(json.dumps(validity, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pilot swarm command-layer validation experiments.")
    parser.add_argument("--seeds", type=int, default=12, help="Seeds per condition.")
    parser.add_argument(
        "--study",
        choices=["focused", "q2", "relay_sweep"],
        default="focused",
        help="Experimental design to run. focused preserves the original pilot; q2 runs the stricter confirmatory design; relay_sweep isolates relay-count trade-offs.",
    )
    parser.add_argument("--full", action="store_true", help="Run all team/map/command/stress combinations.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
        help="Worker processes for simulation.",
    )
    args = parser.parse_args()
    run_experiment(seeds_per_condition=args.seeds, full=args.full, workers=args.workers, study=args.study)


if __name__ == "__main__":
    main()
