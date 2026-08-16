from __future__ import annotations

import math

import numpy as np


COMM_RADIUS = 0.215


def kernel_sigma(comm_radius: float = COMM_RADIUS) -> float:
    """Return sigma such that a pair at comm_radius has Gaussian weight 0.5."""
    return comm_radius / math.sqrt(2.0 * math.log(2.0))


def weight_matrix(positions: np.ndarray, sigma: float) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    diffs = positions[:, None, :] - positions[None, :, :]
    sq_dist = np.sum(diffs * diffs, axis=2)
    weights = np.exp(-sq_dist / (2.0 * sigma * sigma))
    np.fill_diagonal(weights, 0.0)
    return weights


def laplacian(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    return np.diag(np.sum(weights, axis=1)) - weights


def eigengap_tolerance(L: np.ndarray) -> float:
    n = int(L.shape[0])
    return n * math.sqrt(np.finfo(float).eps) * float(np.linalg.norm(L, 2))


def fiedler(L: np.ndarray) -> tuple[float, np.ndarray, float]:
    values, vectors = np.linalg.eigh(np.asarray(L, dtype=float))
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    if len(values) < 3:
        raise ValueError("Fiedler vector requires at least three nodes")
    lam2 = float(values[1])
    gap = float(values[2] - values[1])
    v = vectors[:, 1].astype(float, copy=True)
    v -= np.mean(v)
    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        v /= norm
    return lam2, v, gap


def lambda2_gradient(positions: np.ndarray, weights: np.ndarray, v: np.ndarray, sigma: float) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    v = np.asarray(v, dtype=float)
    gradients = np.zeros_like(positions, dtype=float)
    inv_sigma2 = 1.0 / (sigma * sigma)
    for i in range(len(positions)):
        diffs = positions[i] - positions
        coeff = weights[i] * (v[i] - v) ** 2
        gradients[i] = -inv_sigma2 * np.sum(coeff[:, None] * diffs, axis=0)
    return gradients


def ascent_directions(positions: np.ndarray, sigma: float | None = None) -> tuple[float, float, np.ndarray, bool]:
    positions = np.asarray(positions, dtype=float)
    if len(positions) <= 2:
        raise ValueError("At least three positions are required")
    sigma = kernel_sigma() if sigma is None else float(sigma)
    weights = weight_matrix(positions, sigma)
    L = laplacian(weights)
    lam2, v, gap = fiedler(L)
    degenerate = bool(gap <= eigengap_tolerance(L))
    gradients = lambda2_gradient(positions, weights, v, sigma)
    directions = np.zeros_like(gradients)
    if not degenerate:
        norms = np.linalg.norm(gradients, axis=1)
        nonzero = norms >= 1e-12
        directions[nonzero] = gradients[nonzero] / norms[nonzero, None]
    return lam2, gap, directions, degenerate
