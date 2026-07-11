# Validation Summary

This repository supports the associated manuscript by providing saved simulation outputs, figures, Webots validation files, and a claim-level reproducibility checker.

## Evidence Layers

- Continuous delayed-hazard simulation: 9,024 saved runs.
- Relay-count ablation: 4,800 saved runs.
- Independent grid-world transfer environment: 2,304 saved runs.
- Webots R2025a differential-drive validation: 192 saved runs.

## Claim Boundary

The supported claim is conservative: command-layer effects are environment-dependent, while relay over-allocation repeatedly reduces effective information delay without guaranteeing safer task completion. The package does not claim hardware deployment, human-subject validation, universal command superiority, or relay-rich superiority.

## Key Reproducibility Gate

The included checker verifies required files, row counts, table-level contrasts, raw mean reconstructions for grid-world and Webots contrasts, and diagnostic AUROC values. A package passes when at least 95% of checks succeed.
