# Result Brief for v1.5.0

## Execution Status

- Baseline replay passed before the v1.5.0 additions.
- The v1.4.0 reproducibility checker passed on the prior release: 78/78.
- Expanded v1.5.0 checker passed after the v1.5.0 additions and movie-pixel QA: 303/303.
- S1 production data: 3,840 unique rows, with 320 smoke rows reused under unchanged M0 code hashes and 3,520 remaining rows run once.
- S1 paired join against the reinstrumented baseline: 3,840/3,840 matched, missing pairs 0.
- Webots movies: both sealed replicate-3 target rows were replayed through the original Webots batch-order prefix, recorded only at the target trial, and matched `data/webots_runs.csv` exactly.

## S1 Branch Decision

The S1 comparison policy is a canonical algebraic-connectivity-ascent relay steering policy. It is not a claimed SOTA implementation and is not tuned to the observed data.

Stress-specific branch outcome:

- Degraded: B, unresolved.
- Severe: B, unresolved.

Primary paired k=4 safe-delivery contrast, S1 minus baseline:

- Degraded: +0.0104, 95% CI [-0.0083, +0.0292].
- Severe: +0.0125, 95% CI [-0.0021, +0.0271].

Because both confidence intervals include zero, the S1 safety effect is unresolved at the panel resolution. The result should not be written as rescue, worsening, robustness, or superiority.

Secondary relay-count contrasts are reported only as secondary diagnostics. Some intermediate relay counts show small safe-delivery improvements under S1, but these were not the pre-specified primary comparison and must not replace the k=4 branch decision:

- Degraded k=2: +0.0250, 95% CI [+0.0104, +0.0396].
- Severe k=1: +0.0271, 95% CI [+0.0124, +0.0417].
- Severe k=2: +0.0146, 95% CI [+0.0021, +0.0271].
- Severe k=3: +0.0250, 95% CI [+0.0104, +0.0396].

These effects are small relative to the k=0 to k=4 allocation-level safe-delivery cost and should be described as secondary evidence that the comparison policy can help some intermediate relay allocations, not as a rescue of relay-rich over-allocation.

## Pareto Rescue Test

Independent cell-preserving bootstrap, S1 k=4 versus baseline k=0:

- Degraded safe delivery: -0.1167, 95% CI [-0.1771, -0.0521].
- Degraded effective delay: -7.6922, 95% CI [-7.7500, -7.6347].
- Severe safe delivery: -0.1438, 95% CI [-0.2021, -0.0854].
- Severe effective delay: -14.7585, 95% CI [-14.9128, -14.5964].

The comparison lowers effective delay but remains significantly below the no-relay allocation in safe delivery. Pareto rescue is false in both stress regimes.

## Manuscript Interpretation

Use this wording direction:

> A canonical connectivity-ascent relay steering comparison did not resolve the k=4 safety cost at the resolution of this panel. It slightly increased k=4 safe delivery in point estimate, but the confidence intervals included zero in both stress regimes. Against the no-relay allocation, k=4 retained a large effective-delay advantage but remained significantly worse in safe delivery. Thus the added comparison does not show a Pareto rescue of relay-rich allocation; it narrows the interpretation to the tested relay-count and relay-steering stack rather than overturning the allocation trade-off.

Avoid:

- "S1 proves the trade-off is intrinsic."
- "S1 rescues relay-rich allocation."
- "S1 worsens safety."
- "S1 improves connectivity and therefore safety."
- Any `P(...)` phrasing for bootstrap frequencies.

## Webots Movie Status

Selected pair:

- `no_relay_hetero`, autonomous, severe, replicate 3, seed 967915163.
- `relay_rich`, autonomous, severe, replicate 3, seed 678182620.

Reason:

- Replicate 3 was selected by the sealed operational-score contrast rule. Safe delivery and attrition were not used for selection.

Final movie files:

- `revision_outputs/webots_movie_no_relay_hetero_autonomous_severe_rep3.mp4`
- `revision_outputs/webots_movie_relay_rich_autonomous_severe_rep3.mp4`
- `revision_outputs/webots_movie_side_by_side_tradeoff.mp4`

Both individual replay files are under 100 MB, passed pixel-level nonblack and scene-contrast QA, and their recorded target rows exactly match the archived Webots CSV rows. The side-by-side comparison movie is derived from the two verified replay files and includes role-color and goal-zone annotations for readability. Disabled or immobilized Webots bodies are not removed from the scene, preserving the recorded physics visualization.
