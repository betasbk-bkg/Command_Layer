# Additional Composition and Sensitivity Diagnostics

## Purpose

These analyses probe the robustness of the central relay result: whether it is a team-composition artifact, whether it depends on a particular operational-score weighting, and whether the command-layer effect generalises across strata.

## Matched-Composition Diagnostic

- Runs: 1,536
- Variants: no relay, balanced relay, relay-rich, relay-rich with standardized relay dynamics, relay-rich with scout count matched to no-relay, and the combined scout-matched/standardized variant.
- Stress regimes: degraded and severe.
- Map modes: delayed and scout_belief.
- Command modes: autonomous and crowd_vector.

### Severe Safe-Delivery Contrasts Against No-Relay

| stress   | map_mode     | command_mode   | contrast                                             | metric                |   delta |   ci95_low |   ci95_high |   n_left |   n_right |
|:---------|:-------------|:---------------|:-----------------------------------------------------|:----------------------|--------:|-----------:|------------:|---------:|----------:|
| severe   | delayed      | autonomous     | relay_rich_minus_no_relay                            | safe_delivery_success | -0.2500 |    -0.4688 |      0.0000 |       32 |        32 |
| severe   | delayed      | autonomous     | relay_rich_scout_matched_minus_no_relay              | safe_delivery_success | -0.2812 |    -0.5000 |     -0.0312 |       32 |        32 |
| severe   | delayed      | autonomous     | relay_rich_scout_matched_standardized_minus_no_relay | safe_delivery_success | -0.4062 |    -0.6250 |     -0.1875 |       32 |        32 |
| severe   | delayed      | autonomous     | relay_rich_standardized_minus_no_relay               | safe_delivery_success | -0.3750 |    -0.5938 |     -0.1562 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_minus_no_relay                            | safe_delivery_success | -0.1875 |    -0.4062 |      0.0312 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_scout_matched_minus_no_relay              | safe_delivery_success | -0.2188 |    -0.4375 |      0.0000 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_scout_matched_standardized_minus_no_relay | safe_delivery_success | -0.2812 |    -0.5000 |     -0.0625 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_standardized_minus_no_relay               | safe_delivery_success | -0.2188 |    -0.4375 |      0.0312 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_minus_no_relay                            | safe_delivery_success |  0.0000 |    -0.2188 |      0.2188 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_scout_matched_minus_no_relay              | safe_delivery_success | -0.0625 |    -0.2812 |      0.1562 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_scout_matched_standardized_minus_no_relay | safe_delivery_success | -0.0938 |    -0.3125 |      0.0938 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_standardized_minus_no_relay               | safe_delivery_success | -0.1250 |    -0.3438 |      0.0625 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_minus_no_relay                            | safe_delivery_success | -0.0938 |    -0.3125 |      0.1562 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_scout_matched_minus_no_relay              | safe_delivery_success | -0.1250 |    -0.3438 |      0.1250 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_scout_matched_standardized_minus_no_relay | safe_delivery_success | -0.2188 |    -0.4070 |      0.0000 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_standardized_minus_no_relay               | safe_delivery_success | -0.0625 |    -0.2812 |      0.1875 |       32 |        32 |

### Effective-Delay Contrasts Against No-Relay

| stress   | map_mode     | command_mode   | contrast                                             | metric                   |    delta |   ci95_low |   ci95_high |   n_left |   n_right |
|:---------|:-------------|:---------------|:-----------------------------------------------------|:-------------------------|---------:|-----------:|------------:|---------:|----------:|
| degraded | delayed      | autonomous     | relay_rich_minus_no_relay                            | mean_effective_map_delay |  -7.7773 |    -7.9345 |     -7.5864 |       32 |        32 |
| degraded | delayed      | autonomous     | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay |  -7.7648 |    -7.9046 |     -7.5859 |       32 |        32 |
| degraded | delayed      | autonomous     | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay |  -7.8205 |    -7.9638 |     -7.6643 |       32 |        32 |
| degraded | delayed      | autonomous     | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay |  -7.7178 |    -7.9101 |     -7.4541 |       32 |        32 |
| degraded | delayed      | crowd_vector   | relay_rich_minus_no_relay                            | mean_effective_map_delay |  -7.9139 |    -7.9967 |     -7.8265 |       32 |        32 |
| degraded | delayed      | crowd_vector   | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay |  -7.7708 |    -7.9232 |     -7.5890 |       32 |        32 |
| degraded | delayed      | crowd_vector   | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay |  -7.6429 |    -7.8320 |     -7.4166 |       32 |        32 |
| degraded | delayed      | crowd_vector   | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay |  -7.6965 |    -7.8795 |     -7.4904 |       32 |        32 |
| degraded | scout_belief | autonomous     | relay_rich_minus_no_relay                            | mean_effective_map_delay |  -7.8670 |    -8.0000 |     -7.6883 |       32 |        32 |
| degraded | scout_belief | autonomous     | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay |  -7.8795 |    -7.9754 |     -7.7496 |       32 |        32 |
| degraded | scout_belief | autonomous     | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay |  -7.8254 |    -7.9526 |     -7.6676 |       32 |        32 |
| degraded | scout_belief | autonomous     | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay |  -7.7324 |    -7.9168 |     -7.5209 |       32 |        32 |
| degraded | scout_belief | crowd_vector   | relay_rich_minus_no_relay                            | mean_effective_map_delay |  -7.6583 |    -7.8403 |     -7.4414 |       32 |        32 |
| degraded | scout_belief | crowd_vector   | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay |  -7.8614 |    -7.9653 |     -7.7411 |       32 |        32 |
| degraded | scout_belief | crowd_vector   | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay |  -7.8146 |    -7.9676 |     -7.6417 |       32 |        32 |
| degraded | scout_belief | crowd_vector   | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay |  -7.6633 |    -7.8637 |     -7.4318 |       32 |        32 |
| severe   | delayed      | autonomous     | relay_rich_minus_no_relay                            | mean_effective_map_delay | -14.9210 |   -15.3585 |    -14.4094 |       32 |        32 |
| severe   | delayed      | autonomous     | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay | -15.4067 |   -15.7392 |    -15.0066 |       32 |        32 |
| severe   | delayed      | autonomous     | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay | -14.8178 |   -15.3918 |    -14.1833 |       32 |        32 |
| severe   | delayed      | autonomous     | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay | -14.5014 |   -15.0831 |    -13.8469 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_minus_no_relay                            | mean_effective_map_delay | -14.9079 |   -15.4154 |    -14.3808 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay | -14.8919 |   -15.5312 |    -14.1800 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay | -14.8320 |   -15.4129 |    -14.1039 |       32 |        32 |
| severe   | delayed      | crowd_vector   | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay | -14.7764 |   -15.2823 |    -14.2097 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_minus_no_relay                            | mean_effective_map_delay | -15.0378 |   -15.5114 |    -14.4809 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay | -15.1155 |   -15.5550 |    -14.5686 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay | -14.8535 |   -15.3069 |    -14.3699 |       32 |        32 |
| severe   | scout_belief | autonomous     | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay | -14.4171 |   -15.0005 |    -13.8153 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_minus_no_relay                            | mean_effective_map_delay | -14.4707 |   -15.1901 |    -13.6587 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_scout_matched_minus_no_relay              | mean_effective_map_delay | -14.4444 |   -15.0871 |    -13.7279 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_scout_matched_standardized_minus_no_relay | mean_effective_map_delay | -14.2152 |   -14.9506 |    -13.3761 |       32 |        32 |
| severe   | scout_belief | crowd_vector   | relay_rich_standardized_minus_no_relay               | mean_effective_map_delay | -15.3989 |   -15.7907 |    -14.9110 |       32 |        32 |

## Operational-Score Weight Sensitivity

| layer           | stress   |   weight_sets |   harm_rate |   mean_delta |
|:----------------|:---------|--------------:|------------:|-------------:|
| q2_confirmatory | degraded |             6 |       1.000 |       -1.545 |
| q2_confirmatory | severe   |             6 |       1.000 |       -1.625 |
| relay_sweep     | degraded |             6 |       1.000 |       -1.222 |
| relay_sweep     | severe   |             6 |       1.000 |       -1.565 |

## Interpretation

- If scout-matched relay-rich variants still reduce delay while harming severe safe delivery, the result is harder to dismiss as a missing-scout artifact.
- If standardized relay dynamics weaken but do not reverse the trade-off, the mechanism is partly embodied-role cost but not merely a slow-relay artifact.
- If alternative operational-score weights mostly keep relay-rich deltas negative, the relay conclusion does not depend on one arbitrary composite score.
- These analyses should be treated as auxiliary material unless they are promoted into a revised manuscript with compact reporting.
