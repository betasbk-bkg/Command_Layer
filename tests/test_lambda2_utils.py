from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for rel in ("scripts", "src"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

from lambda2_utils import (  # noqa: E402
    COMM_RADIUS,
    ascent_directions,
    eigengap_tolerance,
    fiedler,
    kernel_sigma,
    laplacian,
    lambda2_gradient,
    weight_matrix,
)
from swarm_pilot_experiment import lambda2_control_snapshot  # noqa: E402


def test_kernel_half_weight() -> None:
    for radius in (0.1, COMM_RADIUS, 0.5):
        sigma = kernel_sigma(radius)
        assert abs(math.exp(-(radius * radius) / (2.0 * sigma * sigma)) - 0.5) < 1e-12


def test_equilateral_lambda2_value() -> None:
    d = 0.2
    positions = np.array([[0.0, 0.0], [d, 0.0], [0.5 * d, math.sqrt(3.0) * 0.5 * d]])
    sigma = kernel_sigma()
    W = weight_matrix(positions, sigma)
    values = np.linalg.eigvalsh(laplacian(W))
    w = math.exp(-(d * d) / (2.0 * sigma * sigma))
    assert abs(float(values[1]) - 3.0 * w) < 1e-10


def test_finite_difference_gradient() -> None:
    rng = np.random.default_rng(20260808)
    sigma = kernel_sigma()
    for n in (4, 8, 12):
        positions = rng.uniform(0.1, 0.9, size=(n, 2))
        W = weight_matrix(positions, sigma)
        L = laplacian(W)
        _, v, gap = fiedler(L)
        gradients = lambda2_gradient(positions, W, v, sigma)
        assert gap > 0.01
        h = 1e-6
        for idx in range(n):
            for dim in range(2):
                shifted_plus = positions.copy()
                shifted_minus = positions.copy()
                shifted_plus[idx, dim] += h
                shifted_minus[idx, dim] -= h
                lam_plus = fiedler(laplacian(weight_matrix(shifted_plus, sigma)))[0]
                lam_minus = fiedler(laplacian(weight_matrix(shifted_minus, sigma)))[0]
                finite_difference = (lam_plus - lam_minus) / (2.0 * h)
                error = abs(finite_difference - gradients[idx, dim])
                scale = max(1.0, abs(finite_difference), abs(gradients[idx, dim]))
                assert error / scale < 1e-6


def test_matrix_symmetry_and_normalization() -> None:
    positions = np.array([[0.1, 0.2], [0.3, 0.4], [0.55, 0.23], [0.8, 0.7]])
    W = weight_matrix(positions, kernel_sigma())
    L = laplacian(W)
    assert np.allclose(W, W.T)
    assert np.allclose(np.diag(W), 0.0)
    assert np.allclose(L @ np.ones(len(positions)), 0.0)


def test_degeneracy_detection() -> None:
    d = 0.2
    positions = np.array([[0.0, 0.0], [d, 0.0], [0.5 * d, math.sqrt(3.0) * 0.5 * d]])
    _, gap, _, degenerate = ascent_directions(positions, kernel_sigma())
    assert abs(gap) <= eigengap_tolerance(laplacian(weight_matrix(positions, kernel_sigma())))
    assert degenerate


def test_directional_derivative_increases_lambda2() -> None:
    positions = np.array(
        [[0.12, 0.11], [0.25, 0.31], [0.41, 0.19], [0.57, 0.62], [0.77, 0.43]],
        dtype=float,
    )
    sigma = kernel_sigma()
    lam2, gap, directions, degenerate = ascent_directions(positions, sigma)
    assert not degenerate
    assert gap > 0.01
    moved = positions + 1e-6 * directions
    moved_lam2, _, _, _ = ascent_directions(moved, sigma)
    assert moved_lam2 > lam2


def test_degenerate_policy_path_falls_back() -> None:
    d = 0.2
    positions = np.array([[0.0, 0.0], [d, 0.0], [0.5 * d, math.sqrt(3.0) * 0.5 * d]])
    alive = np.array([True, True, True])
    snapshot = lambda2_control_snapshot(positions, alive, "s1_lambda2")
    assert snapshot["degenerate"] is True
    assert snapshot["directions"] is None
    assert snapshot["low_population"] is False
