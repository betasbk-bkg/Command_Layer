# Auxiliary Diagnostics Add-on

This add-on contains only the auxiliary auxiliary diagnostics that are
backed by local scripts and archived outputs:

- matched-composition diagnostic, including 1,536 runs;
- operational-score sensitivity readouts;
- Webots permutation tests over the existing 192 Webots trials;
- matched-composition sign-stability checks.

It deliberately does not include a channel-decomposition analysis. The revised
manuscript and response letter should not claim that channel decomposition is
fully scripted in the released repository unless a current, checked script is
added separately.

## Drop-in layout

Copy the directories in this add-on into the root of `Command_Layer_repo_v1.5.0`:

```text
scripts/matched_composition_experiments.py
scripts/statistical_stress_tests.py
data/matched_composition_*.csv, data/score_sensitivity.csv, and data/webots_permutation.csv
reports/statistical_stress_tests.json and reports/auxiliary_diagnostics_summary.json
reports/statistical_stress_tests.md and reports/additional_diagnostics.md
```

The scripts are adjusted for the public repository layout:

- inputs are read from `data/`;
- generated CSV outputs are written to `data/`;
- generated Markdown/JSON summaries are written to `reports/`;
- `matched_composition_experiments.py` imports `src/swarm_pilot_experiment.py`.

## Existing outputs

The included outputs were generated before packaging and are provided for
transparent review:

- `data/matched_composition_runs.csv`: 1,536 matched-composition runs.
- `data/matched_composition_contrasts.csv`: 280 bootstrap contrasts.
- `data/matched_composition_summary.csv`: matched-condition summary.
- `data/matched_composition_sign_stability.csv`: fold sign-stability checks.
- `data/score_sensitivity.csv`: operational-score sensitivity.
- `data/webots_permutation.csv`: 14 Webots permutation tests.

## Re-run commands

From the root of the public repository:

```powershell
python scripts\matched_composition_experiments.py --seeds 32 --workers 8
python scripts\statistical_stress_tests.py
```

If these files are incorporated into the public release, regenerate
`MANIFEST.sha256` and extend the reproducibility checker or documentation so
the manuscript no longer claims outputs that are not in the archived package.
