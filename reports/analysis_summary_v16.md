# Sensitivity and Consistency Analysis Summary (v1.6.0)

This note summarizes the scripted analyses added in version 1.6. The analyses use saved CSV panels only; no new robot simulations are introduced.

## Mission-Sensitivity Threshold

The descriptive freshness-return ratio is `R = g*beta/c`. The mission-sensitivity threshold is `beta_crit_over_beta = c/(g*beta) = 1/R`.

| Stress | R | 95% CI for R | beta_crit/beta | Reciprocal CI |
|---|---:|---:|---:|---:|
| degraded | 0.3630 | [0.1132, 0.7613] | 2.7547 | [1.3135, 8.8331] |
| severe | 0.5044 | [0.1534, 0.9402] | 1.9825 | [1.0636, 6.5199] |

Interpretation: this is a descriptive mission-sensitivity threshold, not a standalone net-benefit decision rule.

## Safe-Delivery Threshold Sensitivity

Safe delivery is recomputed as `strict_success == 1`, `alive_final == n_agents`, `exposure_per_agent_step <= E`, and `recovery_time <= T_rec`.

| Stress | Reference E=0.150, T_rec=10: weight on k<=1 | Plausible region min weight on k<=1 | Entire grid min weight on k<=1 | Entire grid max weight on k>=3 | Boundary max weight on k=2 |
|---|---:|---:|---:|---:|---:|
| degraded | 0.9831 | 0.9760 | 0.4523 | 0.0170 | 0.5460 |
| severe | 0.9990 | 0.9792 | 0.9100 | 0.0032 | 0.0900 |

The plausible region is `E in [0.10,0.20]` and `T_rec in [10,16]`. The full grid is the 8 x 7 grid defined in `threshold_sensitivity_grid.json`.

## Direction Consistency

The direction-consistency analysis compares relay-rich against no-relay allocations across the continuous, grid-world, and Webots layers. It tests sign consistency, not equality of magnitudes across layers.

| Stress | Continuous p | Grid-world p | Webots p | Fisher p | Stouffer p |
|---|---:|---:|---:|---:|---:|
| degraded | 5.0e-05 | 0.2257 | 0.01942 | 2.9347e-05 | 5.3590e-05 |
| severe | 5.0e-05 | 0.01205 | 2.5e-05 | 5.1e-09 | 1.9e-09 |

Across four endpoints, two stress regimes, and three layers, the relay-rich versus no-relay contrast matched the expected direction in 24/24 cells.

## Webots Safe-Delivery Power/MDE

The Webots power readout uses two-sample proportion power with `n=24` per group. The two-sided calculation is the conservative primary readout.

| Stress | Control safe delivery | Relay-rich safe delivery | Observed delta | Two-sided MDE | Two-sided post-hoc power |
|---|---:|---:|---:|---:|---:|
| degraded | 0.5833 | 0.2500 | -0.3333 | 0.3825 | 0.6678 |
| severe | 0.6250 | 0.0000 | -0.6250 | 0.3890 | 1.0000 |

Interpretation: the degraded Webots contrast remains underpowered for moderate effects, whereas the severe Webots contrast is larger than the two-sided 80% MDE.
