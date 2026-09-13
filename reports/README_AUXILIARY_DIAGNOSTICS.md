# Auxiliary Diagnostics

This directory documents auxiliary diagnostics that accompany the main relay and
command-layer analyses. Each diagnostic is backed by a script in `scripts/` and by
archived outputs in `data/` and `reports/`.

## Contents

- Matched-composition diagnostic over 1,536 continuous-simulator runs, in which
  scout count is matched and relay dynamics are standardized across allocations.
- Operational-score sensitivity readouts under alternative component weightings.
- Permutation tests over the 192 archived Webots trials.
- Sign-stability checks for the matched-composition contrasts across folds.

## Archived outputs

- `data/matched_composition_runs.csv`: 1,536 matched-composition runs.
- `data/matched_composition_contrasts.csv`: 280 bootstrap contrasts.
- `data/matched_composition_summary.csv`: matched-condition summary.
- `data/matched_composition_sign_stability.csv`: fold sign-stability checks.
- `data/score_sensitivity.csv`: operational-score sensitivity.
- `data/webots_permutation.csv`: 14 Webots permutation tests.
- `reports/additional_diagnostics.md`, `reports/statistical_stress_tests.md`,
  `reports/statistical_stress_tests.json`, `reports/auxiliary_diagnostics_summary.json`.

## Scope

These diagnostics probe the robustness of the reported relay result. They are not
part of the primary evidence chain, and no channel-decomposition analysis is
included in this package.

## Re-run commands

From the root of this repository:

```
python scripts/matched_composition_experiments.py --seeds 32 --workers 8
python scripts/statistical_stress_tests.py
```

Regenerating these outputs changes their hashes; `MANIFEST.sha256` records the
archived versions.
