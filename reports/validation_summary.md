# Validation Summary

This repository supports the associated manuscript by providing saved simulation outputs, figures, Webots validation files, and a claim-level reproducibility checker.

## Evidence Layers

- Continuous delayed-hazard simulation: 9,024 saved runs.
- Relay-count ablation: 4,800 saved runs.
- Independent grid-world transfer environment: 2,304 saved runs.
- Webots R2025a differential-drive validation: 192 saved runs.

## Claim Boundary

The supported claim is conservative: command-layer effects are environment-dependent, while relay over-allocation repeatedly reduces effective information delay without guaranteeing safer task completion. The package does not claim physical robot deployment, human-subject validation, unconditional command-layer superiority, or relay-rich superiority.

## Key Reproducibility Gate

The included checker verifies required files, row counts, table-level contrasts, raw mean reconstructions for grid-world and Webots contrasts, diagnostic AUROC values, v1.5 comparison-policy outputs, Webots movie QA, and v1.6 sensitivity and consistency analysis outputs. A package passes only when all checks succeed; the released package passes 345/345.

## Sensitivity and Consistency Analyses

Version 1.6.1 adds scripted threshold-sensitivity, cross-layer direction-consistency, Webots power/MDE, and mission-sensitivity analyses. These analyses do not add new robot simulations. They use the released relay-sweep, grid-world, and Webots CSV panels to clarify scope, sensitivity, and statistical interpretation.
