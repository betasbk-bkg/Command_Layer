# Statistical Stress Tests

## Webots Permutation Tests

Permutation tests use the existing 192 Webots trials and avoid normality assumptions. They are intended as small-sample safeguards, not as a replacement for larger Webots reruns.

| layer   | stress   | contrast_type   | contrast                         | metric                   |     delta |   p_permutation_two_sided |   n_left |   n_right |   q_bh_within_webots_family |
|:--------|:---------|:----------------|:---------------------------------|:-------------------------|----------:|--------------------------:|---------:|----------:|----------------------------:|
| webots  | degraded | command         | consensus_gated_minus_autonomous | operational_score        |   7.05809 |                   0.06005 |       32 |        32 |                     0.09341 |
| webots  | degraded | command         | crowd_vector_minus_autonomous    | safe_delivery_success    |   0.15625 |                   0.31988 |       32 |        32 |                     0.40713 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | attrition_rate           |   0.11458 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | degraded_outcome         |   0.33333 |                   0.03910 |       24 |        24 |                     0.06842 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | mean_effective_map_delay | -12.00000 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | payload_delivery_rate    |   0.08333 |                   0.40908 |       24 |        24 |                     0.47726 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | safe_delivery_success    |  -0.33333 |                   0.03885 |       24 |        24 |                     0.06842 |
| webots  | severe   | command         | consensus_gated_minus_autonomous | operational_score        |   3.85529 |                   0.25259 |       32 |        32 |                     0.35362 |
| webots  | severe   | command         | crowd_vector_minus_autonomous    | safe_delivery_success    |   0.00000 |                   1.00000 |       32 |        32 |                     1.00000 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | attrition_rate           |   0.14931 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | degraded_outcome         |   0.62500 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | mean_effective_map_delay | -22.00000 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | payload_delivery_rate    |  -0.04167 |                   0.76871 |       24 |        24 |                     0.82784 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | safe_delivery_success    |  -0.62500 |                   0.00005 |       24 |        24 |                     0.00012 |

## Webots Relay-Only Readout

| layer   | stress   | contrast_type   | contrast                         | metric                   |     delta |   p_permutation_two_sided |   n_left |   n_right |   q_bh_within_webots_family |
|:--------|:---------|:----------------|:---------------------------------|:-------------------------|----------:|--------------------------:|---------:|----------:|----------------------------:|
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | attrition_rate           |   0.11458 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | degraded_outcome         |   0.33333 |                   0.03910 |       24 |        24 |                     0.06842 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | mean_effective_map_delay | -12.00000 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | payload_delivery_rate    |   0.08333 |                   0.40908 |       24 |        24 |                     0.47726 |
| webots  | degraded | team            | relay_rich_minus_no_relay_hetero | safe_delivery_success    |  -0.33333 |                   0.03885 |       24 |        24 |                     0.06842 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | attrition_rate           |   0.14931 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | degraded_outcome         |   0.62500 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | mean_effective_map_delay | -22.00000 |                   0.00005 |       24 |        24 |                     0.00012 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | payload_delivery_rate    |  -0.04167 |                   0.76871 |       24 |        24 |                     0.82784 |
| webots  | severe   | team            | relay_rich_minus_no_relay_hetero | safe_delivery_success    |  -0.62500 |                   0.00005 |       24 |        24 |                     0.00012 |

## Matched-Composition Sign Stability

The table reports four deterministic seed-fold checks for matched-composition relay variants against no-relay.

| stress   | metric                   |   folds |   direction_match_rate |   mean_delta |
|:---------|:-------------------------|--------:|-----------------------:|-------------:|
| degraded | attrition_rate           |      64 |                  0.672 |        0.009 |
| degraded | degraded_outcome         |      64 |                  0.625 |        0.068 |
| degraded | mean_effective_map_delay |      64 |                  1.000 |       -7.775 |
| degraded | safe_delivery_success    |      64 |                  0.609 |       -0.068 |
| severe   | attrition_rate           |      64 |                  0.812 |        0.038 |
| severe   | degraded_outcome         |      64 |                  0.625 |        0.174 |
| severe   | mean_effective_map_delay |      64 |                  1.000 |      -14.813 |
| severe   | safe_delivery_success    |      64 |                  0.672 |       -0.188 |

## Interpretation

- Webots relay safe-delivery, degraded-outcome, attrition, and delay effects should survive small-sample scrutiny if permutation p-values are low.
- Command-layer Webots effects should remain secondary if permutation evidence is weaker.
- Matched-composition sign stability is a check against the possibility that one lucky seed block produced the relay result.
