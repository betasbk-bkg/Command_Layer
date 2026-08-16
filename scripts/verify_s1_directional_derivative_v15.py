from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import swarm_pilot_experiment as sim  # noqa: E402
from lambda2_utils import ascent_directions, eigengap_tolerance, fiedler, lambda2_gradient, laplacian, weight_matrix  # noqa: E402


LAMBDA2_SIGMA = sim.LAMBDA2_SIGMA


def smoke_specs() -> list[dict[str, str | int]]:
    specs = [
        row
        for row in sim.make_design(40, False, "s1_lambda2")
        if row["map_mode"] == "delayed" and row["command_mode"] == "autonomous"
    ]
    return sorted(
        specs,
        key=lambda row: (
            row["team"],
            row["map_mode"],
            row["command_mode"],
            row["stress"],
            int(row["seed"]),
        ),
    )


def evaluate_state(
    *,
    positions: sim.np.ndarray,
    alive: sim.np.ndarray,
    roles: tuple[str, ...],
    spec_row: dict[str, str | int],
    step: int,
    h: float = 1e-6,
) -> tuple[dict[str, object] | None, str | None]:
    alive_pre = sim.np.where(alive)[0]
    if len(alive_pre) <= 2:
        return None, "low_population"
    local_positions = positions[alive_pre]
    weights = weight_matrix(local_positions, LAMBDA2_SIGMA)
    graph_laplacian = laplacian(weights)
    lambda2, fiedler_vec, eigengap = fiedler(graph_laplacian)
    tol = eigengap_tolerance(graph_laplacian)
    if eigengap <= tol:
        return None, "degenerate"
    gradients = lambda2_gradient(local_positions, weights, fiedler_vec, LAMBDA2_SIGMA)
    _, _, directions, degenerate = ascent_directions(local_positions, LAMBDA2_SIGMA)
    if degenerate:
        return None, "degenerate"

    relay_local_indices = [
        local_idx for local_idx, global_idx in enumerate(alive_pre) if roles[int(global_idx)] == "relay"
    ]
    if not relay_local_indices:
        return None, "no_relay"

    directional_derivative = float(
        sum(float(sim.np.dot(gradients[local_idx], directions[local_idx])) for local_idx in relay_local_indices)
    )
    if directional_derivative <= 0.0:
        return None, "nonpositive_directional_derivative"

    perturbed = local_positions.copy()
    for local_idx in relay_local_indices:
        perturbed[local_idx] = sim.np.clip(
            perturbed[local_idx] + h * directions[local_idx], sim.WORLD_MIN, sim.WORLD_MAX
        )
    lambda2_after = float(fiedler(laplacian(weight_matrix(perturbed, LAMBDA2_SIGMA)))[0])
    passed = bool(lambda2_after > float(lambda2))
    record = {
        "team": spec_row["team"],
        "map_mode": spec_row["map_mode"],
        "command_mode": spec_row["command_mode"],
        "stress": spec_row["stress"],
        "seed": int(spec_row["seed"]),
        "step": int(step),
        "n_alive": int(len(alive_pre)),
        "n_relay_alive": int(len(relay_local_indices)),
        "lambda2_before": float(lambda2),
        "lambda2_after": lambda2_after,
        "delta_lambda2": float(lambda2_after - float(lambda2)),
        "eigengap": float(eigengap),
        "eigengap_tolerance": float(tol),
        "directional_derivative": directional_derivative,
        "passed": passed,
    }
    return record, None


def advance_one_step(
    *,
    rng: sim.np.random.Generator,
    positions: sim.np.ndarray,
    alive: sim.np.ndarray,
    reached: sim.np.ndarray,
    roles: tuple[str, ...],
    role_specs: list[sim.RoleSpec],
    stress: dict[str, float],
    map_mode: str,
    stress_name: str,
    seed: int,
    env_phase: float,
    step: int,
) -> None:
    alive_roles = [role for role, is_alive in zip(roles, alive) if is_alive]
    effective_map_delay, _ = sim.relay_adjustments(alive_roles, stress)
    map_func, _ = sim.make_map_func(
        map_mode,
        step,
        env_phase,
        seed,
        positions,
        alive,
        roles,
        effective_map_delay,
        stress,
    )
    centroid = positions[alive].mean(axis=0) if sim.np.any(alive) else sim.START.copy()

    for idx, role in enumerate(roles):
        if not alive[idx]:
            continue
        if role == "scout" and rng.random() < float(stress["scout_dropout"]):
            alive[idx] = False
        elif role == "relay" and rng.random() < float(stress["relay_dropout"]):
            alive[idx] = False

    _, _, dirs_all, degenerate = ascent_directions(positions[sim.np.where(alive)[0]], LAMBDA2_SIGMA) if alive.sum() > 2 else (0.0, 0.0, None, True)
    alive_after = sim.np.where(alive)[0]
    direction_by_global = {}
    if not degenerate and dirs_all is not None:
        for local_idx, global_idx in enumerate(alive_after):
            direction_by_global[int(global_idx)] = dirs_all[local_idx]

    command = sim.np.zeros(2, dtype=float)
    for idx in alive_after:
        pos = positions[idx]
        role_spec = role_specs[idx]

        def local_map(point: sim.np.ndarray) -> float:
            return map_func(point)

        perceived_here = map_func(pos)
        if perceived_here > 0.11:
            perceived_gradient = sim.finite_gradient(local_map, pos)
            avoid = -sim.unit(perceived_gradient)
        else:
            avoid = sim.np.zeros(2, dtype=float)
        target_vec = sim.unit(sim.TARGET - pos)
        cohesion = sim.unit(centroid - pos)
        drive = cohesion
        if roles[int(idx)] == "relay" and int(idx) in direction_by_global:
            drive = direction_by_global[int(idx)]
        move_vec = (
            role_spec.target_gain * target_vec
            + role_spec.avoid_gain * perceived_here * avoid
            + 0.24 * drive
            + 0.0 * command
        )
        if sim.norm(move_vec) < 1e-12:
            move_vec = target_vec
        speed = role_spec.speed * (1.0 - 0.28 * sim.true_hazard(pos, step, env_phase))
        positions[idx] = sim.np.clip(pos + speed * sim.unit(move_vec), sim.WORLD_MIN, sim.WORLD_MAX)

    for idx in sim.np.where(alive)[0]:
        hazard = sim.true_hazard(positions[idx], step, env_phase)
        role_spec = role_specs[idx]
        kill_prob = max(0.0, (hazard - 0.54) * 0.060 / role_spec.survival)
        if rng.random() < kill_prob:
            alive[idx] = False
    reached |= alive & (sim.np.linalg.norm(positions - sim.TARGET, axis=1) < 0.085)


def main() -> None:
    collected: list[dict[str, object]] = []
    excluded_counts: dict[str, int] = {}
    max_needed = 20
    max_steps_per_run = sim.MAX_STEPS

    for spec_row in smoke_specs():
        rng = sim.np.random.default_rng(int(spec_row["seed"]))
        roles = sim.TEAM_COMPOSITIONS[str(spec_row["team"])]
        role_specs = [sim.ROLES[role] for role in roles]
        positions = sim.initialize_positions(rng, len(roles))
        alive = sim.np.ones(len(roles), dtype=bool)
        reached = sim.np.zeros(len(roles), dtype=bool)
        env_phase = rng.uniform(0.0, 2.0 * sim.math.pi)
        stress = sim.stress_profile(str(spec_row["stress"]))

        for step in range(max_steps_per_run):
            record, excluded_reason = evaluate_state(
                positions=positions,
                alive=alive,
                roles=roles,
                spec_row=spec_row,
                step=step,
            )
            if record is not None:
                collected.append(record)
                if len(collected) >= max_needed:
                    break
            else:
                excluded_counts[excluded_reason or "unknown"] = excluded_counts.get(excluded_reason or "unknown", 0) + 1

            advance_one_step(
                rng=rng,
                positions=positions,
                alive=alive,
                reached=reached,
                roles=roles,
                role_specs=role_specs,
                stress=stress,
                map_mode=str(spec_row["map_mode"]),
                stress_name=str(spec_row["stress"]),
                seed=int(spec_row["seed"]),
                env_phase=env_phase,
                step=step,
            )
        if len(collected) >= max_needed:
            break

    passed_count = sum(1 for row in collected if row["passed"])
    report = {
        "protocol": "I2_directional_derivative_smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule": "first 20 eligible step states under sorted smoke design; eligibility requires eigengap > tolerance and positive relay directional derivative",
        "h": 1e-6,
        "eligible_states": len(collected),
        "passed_states": passed_count,
        "excluded_counts_before_collection_complete": excluded_counts,
        "passed": len(collected) == max_needed and passed_count == max_needed,
        "states": collected,
    }
    out = ROOT / "reports" / "s1_directional_derivative_gate.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    gates_path = ROOT / "reports" / "s1_smoke_implementation_gates.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    gates["gates"]["I2_directional_derivative_smoke"] = bool(report["passed"])
    gates["i2_report"] = str(out)
    gates["all_gates_passed"] = all(value is True for value in gates["gates"].values())
    gates_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in ["eligible_states", "passed_states", "passed"]}, indent=2))


if __name__ == "__main__":
    main()
