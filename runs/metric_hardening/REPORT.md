# Metric Hardening Diagnostic Report

## Verdict

Decision: `confirmed_motion_free_contact_difficulty`.

This is a diagnostic recompute from existing artifacts only. No new data collection, benchmark, encoding, model training, or model calls were run.

Main gate components:

- Part 1 all-controls Cube matched-phase excess: `False`
- Cube normalized true-dynamics local expansion: `True`
- Cube normalized model sensitivity: `True`
- Fetch convergent support: `True`

## Cube Part 1

| test | scale | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | median_episode_delta | fraction_episode_positive | wilcoxon_p_greater | n_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase_decile_adjusted | whitened | excess_whitened_recomputed | 3.26291 | -3.48736 | 10.3015 | 0.552409 | nan | nan | nan | nan |
| matched_position | whitened | excess_whitened_recomputed | nan | nan | nan | nan | 2.10222 | 0.6 | 0.204549 | 76 |
| phase_persistence_magnitude_double_match | whitened | excess_whitened_recomputed | 4.44697 | -0.0187752 | 12.9123 | 0.522903 | 1.60394 | 0.705882 | 0.0319138 | 41 |
| phase_decile_adjusted | targetnorm | excess_targetnorm | 0.017869 | 0.00579828 | 0.0298139 | 0.630426 | nan | nan | nan | nan |
| matched_position | targetnorm | excess_targetnorm | nan | nan | nan | nan | 0.030706 | 0.85 | 0.000353813 | 76 |
| phase_persistence_magnitude_double_match | targetnorm | excess_targetnorm | 0.0114085 | 0.00207173 | 0.020466 | 0.537455 | 0.0184982 | 0.8125 | 0.0288391 | 29 |

## Cube Sensitivity Contrasts

| scale | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | neighbor_radius | -0.635397 | -1.48413 | 0.439071 | 0.412211 | 371 | 358 |
| raw | local_expansion | 0.0880551 | 0.0651844 | 0.111421 | 0.643203 | 371 | 358 |
| raw | model_sensitivity | 0.251791 | 0.21919 | 0.287837 | 0.78082 | 371 | 358 |
| raw | residual_norm | 2.09435 | 1.70537 | 2.51987 | 0.705612 | 371 | 358 |
| whitened | neighbor_radius | -32.901 | -60.8226 | -9.83553 | 0.434196 | 371 | 358 |
| whitened | local_expansion | 0.0978887 | 0.0615101 | 0.137706 | 0.560933 | 371 | 358 |
| whitened | model_sensitivity | 0.287567 | 0.202617 | 0.363238 | 0.702345 | 371 | 358 |
| whitened | residual_norm | 26.7156 | 16.2817 | 36.8366 | 0.63249 | 371 | 358 |
| targetnorm | neighbor_radius | -0.619062 | -1.46327 | 0.443093 | 0.417571 | 371 | 358 |
| targetnorm | local_expansion | 0.0891481 | 0.0665303 | 0.111854 | 0.645214 | 371 | 358 |
| targetnorm | model_sensitivity | 0.261558 | 0.229746 | 0.296426 | 0.79109 | 371 | 358 |
| targetnorm | residual_norm | 2.1849 | 1.82883 | 2.62985 | 0.705379 | 371 | 358 |

## Cube Density Summary

| scale | regime | n | n_episodes | median | trimmed_mean | trimmed_mean_ci_low | trimmed_mean_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | transport_free | 358 | 20 | 15.948 | 15.0723 | 12.8901 | 16.5276 |
| raw | interaction | 371 | 20 | 14.6672 | 14.4369 | 13.2691 | 15.3232 |
| raw | static | 31 | 19 | 17.502 | 16.0592 | 13.206 | 17.5372 |
| whitened | transport_free | 358 | 20 | 188.258 | 223.69 | 164.544 | 293.612 |
| whitened | interaction | 371 | 20 | 167.151 | 190.789 | 146.755 | 240.706 |
| whitened | static | 31 | 19 | 167.951 | 233.176 | 151.81 | 331.266 |
| targetnorm | transport_free | 358 | 20 | 16.4471 | 15.581 | 13.49 | 17.1432 |
| targetnorm | interaction | 371 | 20 | 15.1527 | 14.9619 | 13.7952 | 15.9148 |
| targetnorm | static | 31 | 19 | 18.2915 | 16.6536 | 13.9355 | 18.1455 |

## Fetch Sensitivity Contrasts

| scale | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_fullstate | neighbor_radius | 2.65236 | 2.25393 | 3.03037 | 0.75342 | 1069 | 6197 |
| raw_fullstate | local_expansion | -0.0433191 | -0.0565512 | -0.0292362 | 0.381147 | 1069 | 6197 |
| raw_fullstate | model_sensitivity | 0.028077 | 0.0215554 | 0.0364355 | 0.715959 | 1069 | 6197 |
| raw_fullstate | residual_norm | 0.232948 | 0.167427 | 0.3235 | 0.810707 | 1069 | 6197 |
| normalized_fullstate | neighbor_radius | 3.6568 | 2.91899 | 4.37569 | 0.756311 | 1069 | 6197 |
| normalized_fullstate | local_expansion | 0.0960756 | 0.0780251 | 0.115404 | 0.645332 | 1069 | 6197 |
| normalized_fullstate | model_sensitivity | 0.183647 | 0.170465 | 0.197997 | 0.86732 | 1069 | 6197 |
| normalized_fullstate | residual_norm | 1.60126 | 1.41087 | 1.7895 | 0.869713 | 1069 | 6197 |

## Fetch Density Summary

| scale | regime | n | n_episodes | median | trimmed_mean | trimmed_mean_ci_low | trimmed_mean_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw_fullstate | free | 6197 | 200 | 1.81983 | 2.62248 | 2.3523 | 2.90595 |
| raw_fullstate | interaction | 1069 | 187 | 5.12525 | 5.27485 | 4.83208 | 5.70322 |
| raw_fullstate | sustained_contact_dynamics | 1321 | 187 | 2.1532 | 2.57628 | 2.37499 | 2.81563 |
| raw_fullstate | post_impact_response | 682 | 145 | 3.85264 | 3.98107 | 3.71371 | 4.35442 |
| raw_fullstate | impact_onset | 131 | 88 | 3.0237 | 3.65451 | 2.96799 | 5.02516 |
| normalized_fullstate | free | 6197 | 200 | 5.00826 | 5.48712 | 5.13063 | 5.85623 |
| normalized_fullstate | interaction | 1069 | 187 | 8.43249 | 9.14392 | 8.38488 | 9.84618 |
| normalized_fullstate | sustained_contact_dynamics | 1321 | 187 | 5.14192 | 5.34871 | 5.1003 | 5.60246 |
| normalized_fullstate | post_impact_response | 682 | 145 | 6.59913 | 6.83897 | 6.50799 | 7.23912 |
| normalized_fullstate | impact_onset | 131 | 88 | 6.15223 | 7.02734 | 6.00851 | 8.63195 |

## Assumptions And Failure Modes

- Neighbor sets are cross-episode only, which reduces temporal leakage but does not guarantee identical hidden state or action intent.
- Cube sensitivity is latent-state based because the saved Cube artifacts do not include the full action-conditioned model input.
- Local expansion can reflect multimodality, unobserved variables, sparse neighborhoods, or metric geometry; it is not causal proof by itself.
- Fetch is a state-space proxy cross-check, not a latent LeWM result and not sufficient for the main confirm-data gate.
- Density differences should be read alongside expansion/sensitivity; larger interaction neighbor radii weaken the interpretation.

## Provenance

- Cube rows: `1140` total, `760` heldout
- Cube whitening recompute max absolute difference: `7.74913e-05`
- Fetch validation rows: `9400`
- Bootstrap samples: `1000`
- Neighbor k/min: `10` / `5`

## Output Files

- `runs/metric_hardening/cube/part1_metric_contrasts.csv`
- `runs/metric_hardening/cube/double_matched_pairs.csv`
- `runs/metric_hardening/cube/sensitivity_regime_summary.csv`
- `runs/metric_hardening/cube/sensitivity_contrasts.csv`
- `runs/metric_hardening/fetch/sensitivity_regime_summary.csv`
- `runs/metric_hardening/fetch/sensitivity_contrasts.csv`
- `runs/metric_hardening/decision.json`
