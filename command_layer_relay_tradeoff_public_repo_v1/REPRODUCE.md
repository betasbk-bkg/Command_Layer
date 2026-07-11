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
- diagnostic AUROC values.

## Full Simulation Notes

The original simulation scripts are included under `src/`. The Webots validation files are included under `webots_validation/`. Full Webots reruns require Webots R2025a and are slower than the claim-level checker.
