# Reproduction Guide

## Environment

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

The claim-level checker uses saved CSV outputs and does not require rerunning Webots.

## Claim-Level Reproduction

Run:

```powershell
python scripts/reproducibility_check.py --root . --output checks/reproducibility_manual.json
```

The checker verifies:

- required repository files;
- run counts for continuous, relay-sweep, grid-world, and Webots datasets;
- key bootstrap contrast values;
- raw mean contrast reconstruction for grid-world and Webots;
- diagnostic AUROC values;
- v1.5 S1 comparison-policy outputs;
- v1.5 relay-frontier artifacts recomputed from raw CSV files;
- v1.6 threshold-sensitivity, direction-consistency, Webots power/MDE, and mission-sensitivity outputs;
- Webots supplementary movie usability, including pixel-level nonblack and scene-contrast checks for the individual and side-by-side movies.

## Full Simulation Notes

The original simulation scripts are included under `src/`. The Webots validation files are included under `webots_validation/`. Full Webots reruns require Webots R2025a and are slower than the claim-level checker.

## Relay-Frontier Analysis

Run:

```powershell
python scripts/analyze_relay_frontier_v15.py --data data --reports reports
```

This regenerates the v1.5 cell-preserving relay-frontier artifacts from the
released relay-sweep and Q2 delay-grid panels, with no additional simulation.
It writes:

- `reports/layer1_slope_cellpreserving.json`
- `reports/trackE_selection_frequency_final.json`
- `reports/freshness_return_cellpreserving_v15.json`

The ratio R = g*beta/c is reported as a descriptive decomposition term, not as a
standalone net-benefit decision rule.

For the freshness-return artifact only, the compatibility command is:

```powershell
python scripts/analysis_freshness_return.py data
```

## Sensitivity and Consistency Analyses

Run:

```powershell
python scripts/threshold_sensitivity_v16.py --data data --reports reports
python scripts/direction_consistency_v16.py --root . --reports reports
python scripts/webots_power_mde_v16.py --data data --reports reports
python scripts/mission_sensitivity_v16.py --reports reports
```

These regenerate:

- `reports/threshold_sensitivity_grid.json`
- `reports/direction_consistency.json`
- `reports/webots_power_mde.json`
- `reports/mission_sensitivity_beta_threshold_v16.json`

The threshold analysis intentionally recomputes safe delivery using `alive_final == n_agents` rather than a floating-point attrition comparison at the one-loss boundary.
