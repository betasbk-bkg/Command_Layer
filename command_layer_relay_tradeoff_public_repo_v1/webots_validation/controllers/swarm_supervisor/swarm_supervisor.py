"""Batch supervisor for physics-based multi-robot validation in Webots.

The supervisor implements the delayed hazard map, command layer, relay-delay
model, trial resets, and metric logging. The individual robots only receive
wheel speeds, so Webots still handles rigid-body contacts and differential-drive
actuation.
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from controller import Supervisor


N_AGENTS = 12
TIME_STEP = 64
MAX_STEPS = 760
GOAL_X = 2.15
START_X = -2.15
ROBOT_Z = 0.055
WHEEL_RADIUS = 0.035
AXLE_LENGTH = 0.124
MAX_WHEEL_SPEED = 10.5
COMMAND_DIRECTIONS = np.array(
    [
        [1.0, 0.0],
        [0.707, 0.707],
        [0.0, 1.0],
        [-0.707, 0.707],
        [-1.0, 0.0],
        [-0.707, -0.707],
        [0.0, -1.0],
        [0.707, -0.707],
    ],
    dtype=float,
)

TEAM_ROLES = {
    "no_relay_hetero": ["standard"] * 7 + ["scout"] * 3 + ["payload"] * 2,
    "relay_sparse": ["standard"] * 6 + ["scout"] * 3 + ["relay"] * 1 + ["payload"] * 2,
    "balanced_hetero": ["standard"] * 5 + ["scout"] * 3 + ["relay"] * 2 + ["payload"] * 2,
    "relay_rich": ["standard"] * 4 + ["scout"] * 2 + ["relay"] * 4 + ["payload"] * 2,
}

ROLE_SPEC = {
    "standard": {"speed": 0.265, "survival": 1.00, "target": 1.00, "avoid": 1.10, "command": 0.52},
    "scout": {"speed": 0.285, "survival": 0.92, "target": 0.94, "avoid": 1.35, "command": 0.48},
    "relay": {"speed": 0.235, "survival": 1.08, "target": 0.72, "avoid": 0.84, "command": 0.38},
    "payload": {"speed": 0.225, "survival": 0.82, "target": 1.18, "avoid": 0.82, "command": 0.46},
}

STRESS_PROFILES = {
    "degraded": {
        "map_delay": 18,
        "command_delay": 5,
        "packet_loss": 0.10,
        "command_noise": 0.24,
        "relay_dropout": 0.00055,
        "hazard_pressure": 1.00,
    },
    "severe": {
        "map_delay": 34,
        "command_delay": 8,
        "packet_loss": 0.22,
        "command_noise": 0.38,
        "relay_dropout": 0.00135,
        "hazard_pressure": 1.18,
    },
}

STATIC_BLOCKS = [
    {"center": np.array([-0.45, 0.72]), "half": np.array([0.37, 0.17])},
    {"center": np.array([0.52, -0.63]), "half": np.array([0.37, 0.17])},
    {"center": np.array([0.18, 0.05]), "half": np.array([0.12, 0.34])},
]


def unit(vec: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        return np.zeros(2, dtype=float)
    return vec / length


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def angle_wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def gaussian(pos: np.ndarray, center: np.ndarray, sx: float, sy: float) -> float:
    dx = (float(pos[0]) - float(center[0])) / sx
    dy = (float(pos[1]) - float(center[1])) / sy
    return math.exp(-0.5 * (dx * dx + dy * dy))


def stable_noise(pos: np.ndarray, seed: int, channel: int) -> float:
    value = math.sin(pos[0] * 12.9898 + pos[1] * 78.233 + seed * 0.013 + channel * 9.17) * 43758.5453
    return value - math.floor(value)


def true_hazard(pos: np.ndarray, step: int, phase: float, pressure: float) -> float:
    t = step / 80.0
    center_a = np.array([-0.18 + 0.24 * math.sin(0.42 * t + phase), 0.42 * math.cos(0.36 * t + phase)])
    center_b = np.array([0.60 + 0.18 * math.sin(0.31 * t + 0.7 * phase), -0.52 + 0.16 * math.cos(0.27 * t)])
    moving_front = np.array([-0.75 + 0.0045 * step, 0.0])
    hazard = (
        0.58 * gaussian(pos, center_a, 0.45, 0.34)
        + 0.48 * gaussian(pos, center_b, 0.38, 0.30)
        + 0.36 * gaussian(pos, moving_front, 0.34, 1.25)
    )
    for block in STATIC_BLOCKS:
        hazard += 0.20 * gaussian(pos, block["center"], block["half"][0] + 0.20, block["half"][1] + 0.18)
    return clamp(pressure * hazard, 0.0, 1.0)


def map_hazard(pos: np.ndarray, step: int, phase: float, stress: dict[str, float], delay: int, seed: int) -> float:
    delayed = max(0, step - delay)
    base = true_hazard(pos, delayed, phase, float(stress["hazard_pressure"]))
    noise = 0.11 * (stable_noise(pos, seed, delayed % 11) - 0.5)
    return clamp(base + noise, 0.0, 1.0)


def hazard_gradient(pos: np.ndarray, step: int, phase: float, stress: dict[str, float], delay: int, seed: int) -> np.ndarray:
    eps = 0.045
    dx = np.array([eps, 0.0])
    dy = np.array([0.0, eps])
    gx = map_hazard(pos + dx, step, phase, stress, delay, seed) - map_hazard(pos - dx, step, phase, stress, delay, seed)
    gy = map_hazard(pos + dy, step, phase, stress, delay, seed) - map_hazard(pos - dy, step, phase, stress, delay, seed)
    return np.array([gx, gy], dtype=float) / (2.0 * eps)


def obstacle_repulsion(pos: np.ndarray) -> np.ndarray:
    force = np.zeros(2, dtype=float)
    for block in STATIC_BLOCKS:
        delta = pos - block["center"]
        outside = np.maximum(np.abs(delta) - block["half"], 0.0)
        distance = float(np.linalg.norm(outside))
        if distance < 0.30:
            axis = np.sign(delta) * np.maximum(np.abs(delta) - block["half"], 0.02)
            direction = unit(axis)
            force += direction * (0.30 - distance) / 0.30
    wall_margin = 1.85
    if pos[1] > wall_margin:
        force += np.array([0.0, -(pos[1] - wall_margin) * 2.2])
    if pos[1] < -wall_margin:
        force += np.array([0.0, (-wall_margin - pos[1]) * 2.2])
    if pos[0] > 2.35:
        force += np.array([-(pos[0] - 2.35) * 2.0, 0.0])
    if pos[0] < -2.35:
        force += np.array([(-2.35 - pos[0]) * 2.0, 0.0])
    return force


def relay_adjustments(roles: list[str], stress: dict[str, float]) -> tuple[int, float]:
    relay_count = roles.count("relay")
    relay_bonus = min(0.64, 0.16 * relay_count)
    effective_delay = int(round(float(stress["map_delay"]) * (1.0 - relay_bonus)))
    effective_packet_loss = max(0.0, float(stress["packet_loss"]) * (1.0 - 0.70 * relay_bonus))
    return effective_delay, effective_packet_loss


def choose_command(
    centroid: np.ndarray,
    step: int,
    rng: np.random.Generator,
    phase: float,
    stress: dict[str, float],
    delay: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    target_direction = unit(np.array([GOAL_X, 0.0]) - centroid)
    votes = []
    for voter in range(31):
        scores = []
        for direction in COMMAND_DIRECTIONS:
            ahead = np.clip(centroid + 0.52 * direction, [-2.35, -1.85], [2.35, 1.85])
            risk = map_hazard(ahead, step, phase, stress, delay, seed + voter)
            block_force = float(np.dot(obstacle_repulsion(ahead), direction))
            score = (
                1.15 * float(np.dot(direction, target_direction))
                - 1.35 * risk
                + 0.25 * block_force
                + rng.normal(0.0, float(stress["command_noise"]))
            )
            scores.append(score)
        votes.append(COMMAND_DIRECTIONS[int(np.argmax(scores))])
    aggregate = np.mean(np.asarray(votes), axis=0)
    consensus = clamp(float(np.linalg.norm(aggregate)), 0.0, 1.0)
    return unit(aggregate), consensus


class BatchSupervisor(Supervisor):
    def __init__(self) -> None:
        super().__init__()
        self.emitter = self.getDevice("emitter")
        self.nodes = [self.getFromDef(f"BOT_{idx}") for idx in range(N_AGENTS)]
        self.translation_fields = [node.getField("translation") for node in self.nodes]
        self.rotation_fields = [node.getField("rotation") for node in self.nodes]
        self.config = self._load_config()
        self.output_path = Path(self.config["output_path"])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict[str, Any]:
        path = os.environ.get("WEBOTS_SWARM_CONFIG")
        if not path:
            raise RuntimeError("WEBOTS_SWARM_CONFIG is not set")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def reset_trial(self, rng: np.random.Generator) -> None:
        base_y = np.linspace(-1.10, 1.10, N_AGENTS)
        jitter = rng.normal(0.0, 0.045, size=(N_AGENTS, 2))
        for idx in range(N_AGENTS):
            x = START_X + float(jitter[idx, 0])
            y = float(base_y[idx] + jitter[idx, 1])
            yaw = float(rng.normal(0.0, 0.04))
            self.translation_fields[idx].setSFVec3f([x, y, ROBOT_Z])
            self.rotation_fields[idx].setSFRotation([0.0, 0.0, 1.0, yaw])
            self.nodes[idx].resetPhysics()
        self._send_stop()
        for _ in range(8):
            self.step(TIME_STEP)

    def _send_stop(self) -> None:
        self.emitter.send(json.dumps({f"BOT_{idx}": [0.0, 0.0] for idx in range(N_AGENTS)}))

    def positions(self) -> np.ndarray:
        return np.asarray([field.getSFVec3f()[:2] for field in self.translation_fields], dtype=float)

    def yaw(self, idx: int) -> float:
        rotation = self.rotation_fields[idx].getSFRotation()
        sign = 1.0 if rotation[2] >= 0 else -1.0
        return angle_wrap(sign * float(rotation[3]))

    def wheel_speeds(self, desired: np.ndarray, yaw: float, max_speed: float) -> tuple[float, float]:
        if float(np.linalg.norm(desired)) < 1e-6:
            return 0.0, 0.0
        desired_unit = unit(desired)
        desired_angle = math.atan2(float(desired_unit[1]), float(desired_unit[0]))
        error = angle_wrap(desired_angle - yaw)
        linear = max_speed * max(0.0, math.cos(error))
        angular = clamp(3.8 * error, -4.2, 4.2)
        left = (linear - angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
        right = (linear + angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
        return clamp(left, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED), clamp(right, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

    def run_trial(self, spec: dict[str, Any]) -> dict[str, Any]:
        seed = int(spec["seed"])
        rng = np.random.default_rng(seed)
        stress = STRESS_PROFILES[str(spec["stress"])]
        roles = TEAM_ROLES[str(spec["team"])]
        relay_count = roles.count("relay")
        effective_delay, packet_loss = relay_adjustments(roles, stress)
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        command_queue: list[np.ndarray | None] = []
        last_command = np.zeros(2, dtype=float)
        previous_command = np.zeros(2, dtype=float)
        consensus_values: list[float] = []
        reversal_count = 0
        command_count = 0
        hazard_exposure = 0.0
        alive = np.ones(N_AGENTS, dtype=bool)
        delivered = np.zeros(N_AGENTS, dtype=bool)
        completion_step = MAX_STEPS

        self.reset_trial(rng)

        for step in range(MAX_STEPS):
            pos = self.positions()
            alive_positions = pos[alive]
            centroid = np.mean(alive_positions, axis=0) if len(alive_positions) else np.array([START_X, 0.0])

            command = np.zeros(2, dtype=float)
            if spec["command_mode"] in {"crowd_vector", "consensus_gated", "stale_hold"}:
                raw_command, consensus = choose_command(centroid, step, rng, phase, stress, effective_delay, seed)
                if str(spec["command_mode"]) == "consensus_gated":
                    risk_ahead = map_hazard(centroid + 0.62 * raw_command, step, phase, stress, effective_delay, seed)
                    raw_command = raw_command * (0.30 if risk_ahead > 0.46 else 1.0)
                consensus_values.append(consensus)
                if rng.random() < packet_loss:
                    command_queue.append(None)
                else:
                    command_queue.append(raw_command.copy())
                if len(command_queue) > int(stress["command_delay"]):
                    delayed = command_queue.pop(0)
                    if delayed is None:
                        command = last_command.copy() if spec["command_mode"] == "stale_hold" else np.zeros(2)
                    else:
                        command = delayed.copy()
                else:
                    command = last_command.copy() if spec["command_mode"] == "stale_hold" else np.zeros(2)
                if float(np.linalg.norm(command)) > 0.05:
                    command_count += 1
                    if float(np.linalg.norm(previous_command)) > 0.05 and float(np.dot(unit(command), unit(previous_command))) < -0.2:
                        reversal_count += 1
                    previous_command = command.copy()
                last_command = command.copy()

            commands: dict[str, list[float]] = {}
            for idx in range(N_AGENTS):
                role = roles[idx]
                if not alive[idx]:
                    commands[f"BOT_{idx}"] = [0.0, 0.0]
                    continue

                hazard = true_hazard(pos[idx], step, phase, float(stress["hazard_pressure"]))
                hazard_exposure += hazard
                role_spec = ROLE_SPEC[role]
                if role == "relay" and rng.random() < float(stress["relay_dropout"]):
                    alive[idx] = False
                    commands[f"BOT_{idx}"] = [0.0, 0.0]
                    continue
                attrition_p = max(0.0, hazard - 0.62) * 0.00125 / float(role_spec["survival"])
                if rng.random() < attrition_p:
                    alive[idx] = False
                    commands[f"BOT_{idx}"] = [0.0, 0.0]
                    continue

                if pos[idx, 0] >= GOAL_X and abs(pos[idx, 1]) <= 1.65:
                    delivered[idx] = True

                target = unit(np.array([GOAL_X, 0.0]) - pos[idx])
                avoid = -unit(hazard_gradient(pos[idx], step, phase, stress, effective_delay, seed)) if hazard > 0.12 else np.zeros(2)
                obstacle = obstacle_repulsion(pos[idx])
                separation = np.zeros(2, dtype=float)
                for other_idx in range(N_AGENTS):
                    if idx == other_idx or not alive[other_idx]:
                        continue
                    delta = pos[idx] - pos[other_idx]
                    distance = float(np.linalg.norm(delta))
                    if 1e-4 < distance < 0.23:
                        separation += unit(delta) * (0.23 - distance) / 0.23
                desired = (
                    float(role_spec["target"]) * target
                    + float(role_spec["avoid"]) * avoid
                    + 1.85 * obstacle
                    + 0.72 * separation
                    + float(role_spec["command"]) * command
                )
                if delivered[idx]:
                    desired = 0.25 * unit(np.array([2.28, np.clip(pos[idx, 1], -1.55, 1.55)]) - pos[idx])
                left, right = self.wheel_speeds(desired, self.yaw(idx), float(role_spec["speed"]))
                commands[f"BOT_{idx}"] = [left, right]

            self.emitter.send(json.dumps(commands))
            if self.step(TIME_STEP) == -1:
                break

            if step > 120 and np.all(delivered[[idx for idx, role in enumerate(roles) if role == "payload"]]):
                completion_step = step
                break

        self._send_stop()
        for _ in range(4):
            self.step(TIME_STEP)

        final_pos = self.positions()
        progress = np.clip((final_pos[:, 0] - START_X) / (GOAL_X - START_X), 0.0, 1.0)
        payload_indices = [idx for idx, role in enumerate(roles) if role == "payload"]
        payload_delivery = float(np.mean(delivered[payload_indices])) if payload_indices else float(np.mean(delivered))
        mean_progress = float(np.mean(progress))
        attrition_rate = float(1.0 - np.mean(alive))
        exposure = float(hazard_exposure / max(1, MAX_STEPS * N_AGENTS))
        exposure_norm = clamp(exposure / 0.24, 0.0, 1.0)
        mission_success = int(payload_delivery >= 0.75)
        safe_delivery = int(mission_success and attrition_rate < (1.0 / N_AGENTS) and exposure < 0.22)
        degraded = int((not safe_delivery) or attrition_rate >= (1.0 / N_AGENTS) or exposure > 0.24)
        operational_score = 100.0 * (
            0.42 * payload_delivery
            + 0.24 * mean_progress
            + 0.20 * (1.0 - attrition_rate)
            + 0.14 * (1.0 - exposure_norm)
        )

        return {
            "environment": "webots_physics",
            "team": spec["team"],
            "relay_count": relay_count,
            "map_mode": "delayed_hazard",
            "command_mode": spec["command_mode"],
            "stress": spec["stress"],
            "seed": seed,
            "mission_success": mission_success,
            "safe_delivery_success": safe_delivery,
            "degraded_outcome": degraded,
            "payload_delivery_rate": payload_delivery,
            "mean_progress": mean_progress,
            "operational_score": operational_score,
            "attrition_rate": attrition_rate,
            "mean_hazard_exposure": exposure,
            "mean_effective_map_delay": effective_delay,
            "packet_loss": packet_loss,
            "completion_step": completion_step,
            "mean_command_consensus": float(np.mean(consensus_values)) if consensus_values else 0.0,
            "command_reversal_rate": float(reversal_count / max(1, command_count)),
        }

    def run(self) -> None:
        rows = []
        fieldnames: list[str] | None = None
        for idx, spec in enumerate(self.config["trials"], start=1):
            row = self.run_trial(spec)
            rows.append(row)
            fieldnames = list(row.keys())
            print(f"WEBOTS_TRIAL {idx}/{len(self.config['trials'])} {row}", flush=True)

        if fieldnames is None:
            raise RuntimeError("No trials were configured")
        with open(self.output_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"WEBOTS_RESULTS {self.output_path}", flush=True)
        self.simulationQuit(0)


BatchSupervisor().run()
