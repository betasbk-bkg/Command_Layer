# Command-Layer Resilience and Relay Over-Allocation Trade-Offs

This repository package supports the associated manuscript:

**Command-Layer Resilience and Relay Over-Allocation Trade-Offs in Heterogeneous Multi-Robot Teams Under Delayed Hazard Information**

The package contains simulation code, Webots validation files, saved seed-level results, comparison-policy outputs, analysis tables, six figure files (five manuscript figures plus one supporting grid-relay figure), citation metadata, validation reports, Webots supplementary movie files, a side-by-side Webots comparison movie, and claim-level reproducibility checks. It does not include the manuscript file.

Archived on Zenodo (all versions): https://doi.org/10.5281/zenodo.21310895
This concept DOI resolves to the latest published release; this package is the v1.5.0 release package.

## Main Claim Boundary

The package supports a conservative claim:

- Command-layer resilience is environment-dependent.
- Relay over-allocation reduces effective delay but can harm safe delivery and attrition.

It does not claim physical robot deployment, human-subject validation, unconditional command-layer superiority, or relay-rich superiority.

## Relay-Frontier Analysis

The relay allocation trade-off is summarized with cell-preserving relay-frontier diagnostics. The descriptive ratio R = g*beta/c compares a latency-linked safe-delivery gain term with the observed allocation-level safe-delivery slope. Because c is an observed net slope over the tested allocation sequence, R is not a standalone net-benefit decision rule. The analysis is fully scripted and uses only the released panels, with no additional simulation:

```powershell
python scripts/analyze_relay_frontier_v15.py --data data --reports reports
```

Expected outputs are recorded in `reports/layer1_slope_cellpreserving.json`, `reports/trackE_selection_frequency_final.json`, and `reports/freshness_return_cellpreserving_v15.json`.

## Comparison-Policy Additions (v1.5.0)

Version 1.5.0 adds a sealed analysis package:

- S1: a canonical algebraic-connectivity-ascent relay steering comparison policy.
- S2: a communication-proxy-optimal greedy allocation formalization.
- Webots supplementary movies for the sealed replicate-3 pair, recorded by replaying the original Webots batch-order prefix and verifying the target rows against the archived CSV. These movies are illustrative supplements: safe delivery is the strict composite endpoint defined in the manuscript (Section IV-B), not mere goal crossing, so a replay can show payload arrival at the goal band while safe delivery is 0 when attrition exceeds the 1/12 threshold.
- A side-by-side Webots comparison movie showing the same severe/autonomous setting for no-relay and relay-rich allocations. Goal, start-line, role-color, and legend elements are visual-only additions; disabled or immobilized bodies are not removed from the Webots scene.
- `data/webots_movie_replicate_contrasts.csv`: the replicate-wise contrast table for all eight severe/autonomous candidate pairs (no-relay versus relay-rich), including the operational-score contrast used by the sealed median-bracket selection rule and a flag marking the replayed replicate.
- Auxiliary diagnostics: `scripts/matched_composition_experiments.py` and `scripts/statistical_stress_tests.py` with their saved outputs (`data/matched_composition_*.csv, data/score_sensitivity.csv, and data/webots_permutation.csv`, `reports/matched_composition_* / statistical_stress_tests`), covering the 1,536-run matched-composition panel and the Webots permutation stress tests referenced in the manuscript.

Primary S1 comparisons exclude `degraded_outcome` because that label is panel-relative. The S1 branch decision is reported in `reports/s1_lambda2_summary.json`.

## Funding and AI Disclosure

- Funding: This research received no specific grant funding from any funding agency in the public, commercial, or not-for-profit sectors.
- Use of generative AI: ChatGPT/Codex (OpenAI) assisted with manuscript drafting, code drafting/organization, workflow execution, and reproducible computational checks. Numerical outputs were produced by the released scripts and saved configurations, with the author responsible for the study design, verification, scientific claims, and final interpretation.

## Quick Check

```powershell
python scripts/reproducibility_check.py --root .
```

The pre-release gate used before packaging required at least three independent checker runs; the released package passes all checks (303/303), and the checker reports PASS only when every check succeeds. Gate logs are not included in the public package; users can rerun the checker with the command above.
