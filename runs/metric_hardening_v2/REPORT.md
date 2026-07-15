# Metric Hardening V2 Report

## Verdict

```json
{
  "cube_main_status": "mixed",
  "extra_checks": {
    "cube_aliasing_discriminator": "failed",
    "cube_knn_conditional_mean_floor": "failed",
    "cube_residual_predictability_probe": "failed"
  },
  "fetch_support_status": "supported",
  "holes_closed": {
    "action_conditioning": "closed",
    "cube_whitening_conditioning": "closed",
    "fetch_density_mismatch": "closed",
    "single_k_dependence": "closed"
  },
  "r_item_statuses": {
    "cube_knn_floor": "fail",
    "cube_r1": "pass",
    "cube_r2": "pass",
    "cube_r3": "fail",
    "cube_r4": "pass",
    "cube_residual_predictability": "fail",
    "fetch_r1": "pass",
    "fetch_r4": "pass",
    "fetch_r5": "pass"
  }
}
```

| item | status | detail |
| --- | --- | --- |
| R1 Cube action-matched | PASS | pass |
| R1 Fetch action-matched | PASS | pass |
| R2 Cube whitening robustness | PASS | pass |
| R3 Cube aliasing discriminator | FAIL | fail |
| R4 Cube k-sweep | PASS | pass |
| R4 Fetch k-sweep | PASS | pass |
| R5 Fetch density matching | PASS | pass |
| Cube residual-predictability probe | FAIL | fail |
| Cube kNN conditional-mean floor | FAIL | fail |

This is a recompute-only pass from existing artifacts. No training, pixel encoding, model calls, or new data collection were run.
Unit tests: `uv run --with numpy --with pandas --with scipy --with matplotlib --with h5py python -m unittest runs/metric_hardening_v2/test_metric_hardening_v2.py` passed (`13` tests).

## R1 Action-Matched Neighbors

Cube: `PASS`. Fetch: `PASS`.

Cube contrasts:

| metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n | left_episodes | right_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neighbor_radius | -2.38492 | -4.2866 | -0.623249 | 0.432901 | 371 | 358 | 20 | 20 |
| action_neighbor_distance | -0.216426 | -0.277032 | -0.15341 | 0.344554 | 371 | 358 | 20 | 20 |
| local_expansion | 1.19151 | 0.704485 | 1.66828 | 0.570472 | 371 | 358 | 20 | 20 |
| model_sensitivity | 3.85259 | 2.93247 | 4.63084 | 0.721747 | 371 | 358 | 20 | 20 |

Cube surviving-neighbor counts:

| regime | n_anchors | n_usable | n_below_5_neighbors | n_episodes | median | trimmed_mean |
| --- | --- | --- | --- | --- | --- | --- |
| transport_free | 358 | 358 | 0 | 20 | 13.6387 | 16.2454 |
| interaction | 371 | 371 | 0 | 20 | 12.1167 | 13.8605 |
| static | 31 | 31 | 0 | 19 | 12.193 | 16.9359 |

Fetch contrasts:

| metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n | left_episodes | right_episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neighbor_radius | 0.27745 | 0.228778 | 0.321969 | 0.777348 | 1069 | 6197 | 187 | 200 |
| action_neighbor_distance | 0.0479142 | 0.037539 | 0.0579095 | 0.679721 | 1069 | 6197 | 187 | 200 |
| local_expansion | 0.772492 | 0.613852 | 0.943601 | 0.642167 | 1069 | 6197 | 187 | 200 |
| model_sensitivity | 2.42562 | 2.27804 | 2.57582 | 0.873095 | 1069 | 6197 | 187 | 200 |

Fetch surviving-neighbor counts:

| regime | n_anchors | n_usable | n_below_5_neighbors | n_episodes | median | trimmed_mean |
| --- | --- | --- | --- | --- | --- | --- |
| free | 6197 | 6197 | 0 | 200 | 0.417297 | 0.454439 |
| interaction | 1069 | 1069 | 0 | 187 | 0.695943 | 0.731889 |
| sustained_contact_dynamics | 1321 | 1321 | 0 | 187 | 0.43727 | 0.448936 |
| post_impact_response | 682 | 682 | 0 | 145 | 0.54653 | 0.557142 |
| impact_onset | 131 | 131 | 0 | 88 | 0.479303 | 0.563641 |

## R2 Whitening Robustness

Status: `PASS`.

| scale | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_whitened | neighbor_radius | -32.901 | -60.9985 | -8.33812 | 0.434196 | 371 | 358 |
| full_whitened | local_expansion | 0.0978887 | 0.0596803 | 0.136336 | 0.560933 | 371 | 358 |
| full_whitened | model_sensitivity | 0.287567 | 0.200694 | 0.358702 | 0.702345 | 371 | 358 |
| floor_1e-3 | neighbor_radius | -5.23639 | -10.3483 | -0.538739 | 0.441198 | 371 | 358 |
| floor_1e-3 | local_expansion | 0.0732934 | 0.0444884 | 0.10594 | 0.568191 | 371 | 358 |
| floor_1e-3 | model_sensitivity | 0.274982 | 0.220958 | 0.326212 | 0.752466 | 371 | 358 |
| floor_1e-2 | neighbor_radius | -1.40418 | -2.96168 | 0.364355 | 0.454042 | 371 | 358 |
| floor_1e-2 | local_expansion | 0.0558446 | 0.0342638 | 0.0811214 | 0.567483 | 371 | 358 |
| floor_1e-2 | model_sensitivity | 0.237653 | 0.191583 | 0.278167 | 0.744861 | 371 | 358 |
| pca90 | neighbor_radius | 0.229666 | 0.00321974 | 0.487647 | 0.549346 | 371 | 358 |
| pca90 | local_expansion | 0.0645105 | 0.0277447 | 0.0990005 | 0.573108 | 371 | 358 |
| pca90 | model_sensitivity | 0.199226 | 0.154394 | 0.236597 | 0.706463 | 371 | 358 |
| pca99 | neighbor_radius | 0.413204 | -0.274708 | 1.20516 | 0.536132 | 371 | 358 |
| pca99 | local_expansion | 0.0724343 | 0.0465418 | 0.0966822 | 0.607433 | 371 | 358 |
| pca99 | model_sensitivity | 0.201908 | 0.171426 | 0.231557 | 0.722289 | 371 | 358 |

## R3v2 Aliasing Discriminator

Status: `FAIL`.

| scale | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r3v2_augmented_kinematic_action_neighbors_targetnorm_latent_quat | neighbor_radius | 0.0890332 | 0.0375452 | 0.146177 | 0.60744 | 371 | 358 |
| r3v2_augmented_kinematic_action_neighbors_targetnorm_latent_quat | action_neighbor_distance | -0.0982705 | -0.129552 | -0.0600133 | 0.46502 | 371 | 358 |
| r3v2_augmented_kinematic_action_neighbors_targetnorm_latent_quat | local_expansion | -3.6553 | -4.82146 | -2.56956 | 0.382922 | 371 | 358 |
| r3v2_augmented_kinematic_action_neighbors_targetnorm_latent_quat | model_sensitivity | 2.31406 | 2.07173 | 2.55489 | 0.71726 | 371 | 358 |

Symmetric diagnostic: latent/action neighbors, kinematic-space divergence.

| scale | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r3v2_latent_action_neighbors_kinematic_divergence | neighbor_radius | -2.38492 | -4.42071 | -0.639813 | 0.432901 | 371 | 358 |
| r3v2_latent_action_neighbors_kinematic_divergence | local_expansion | 0.520269 | 0.325566 | 0.708711 | 0.626933 | 371 | 358 |
| r3v2_latent_action_neighbors_kinematic_divergence | kinematic_divergence | 2.16533 | 1.52429 | 2.76694 | 0.716552 | 371 | 358 |

## Cube Struggle Probes

Residual predictability: `FAIL`. kNN conditional-mean floor: `FAIL`.

| metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | left_n | right_n |
| --- | --- | --- | --- | --- | --- | --- |
| predicted_residual_norm | 2.03977 | -0.649881 | 4.86306 | 0.519756 | 371 | 358 |
| residual_prediction_mse | 29.3723 | 16.9832 | 43.3538 | 0.666973 | 371 | 358 |
| residual_predictability | 0.137175 | -0.0415136 | 0.28202 | 0.54336 | 371 | 358 |
| conditional_mean_floor | -39.0234 | -116.088 | 24.4345 | 0.45929 | 371 | 358 |
| conditional_mean_gain | -0.0332429 | -0.107919 | 0.0347529 | 0.463838 | 371 | 358 |

## R4 k-Sweep

Cube: `PASS`. Fetch: `PASS`.

Cube k-sweep:

| k | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | n_valid_interaction | n_valid_right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | local_expansion | 2.43481 | 1.85837 | 3.15475 | 0.607192 | 371 | 358 |
| 5 | model_sensitivity | 4.56021 | 3.55525 | 5.49507 | 0.731015 | 371 | 358 |
| 10 | local_expansion | 1.19151 | 0.720846 | 1.68007 | 0.570472 | 371 | 358 |
| 10 | model_sensitivity | 3.85259 | 2.97828 | 4.65518 | 0.721747 | 371 | 358 |
| 20 | local_expansion | 0.98111 | 0.635898 | 1.45268 | 0.564728 | 371 | 358 |
| 20 | model_sensitivity | 3.30926 | 2.42516 | 4.10028 | 0.707622 | 371 | 358 |
| 40 | local_expansion | 0.724757 | 0.453523 | 1.15279 | 0.553268 | 371 | 358 |
| 40 | model_sensitivity | 2.98979 | 2.11727 | 3.76747 | 0.702811 | 371 | 358 |

Fetch k-sweep:

| k | metric | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | n_valid_interaction | n_valid_right |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | local_expansion | 0.877089 | 0.690449 | 1.06047 | 0.639861 | 1069 | 6197 |
| 5 | model_sensitivity | 2.71376 | 2.52867 | 2.9077 | 0.873079 | 1069 | 6197 |
| 10 | local_expansion | 0.772492 | 0.611684 | 0.947317 | 0.642167 | 1069 | 6197 |
| 10 | model_sensitivity | 2.42562 | 2.27684 | 2.5928 | 0.873095 | 1069 | 6197 |
| 20 | local_expansion | 0.61155 | 0.448494 | 0.768292 | 0.627786 | 1069 | 6197 |
| 20 | model_sensitivity | 2.15772 | 2.02525 | 2.30057 | 0.870927 | 1069 | 6197 |
| 40 | local_expansion | 0.47533 | 0.341161 | 0.617871 | 0.608842 | 1069 | 6197 |
| 40 | model_sensitivity | 1.90934 | 1.78381 | 2.02366 | 0.869357 | 1069 | 6197 |

## R5 Fetch Density Matching

Status: `PASS`.

| metric | n_pairs | eligible_episodes | match_fraction_interaction | delta_trimmed_mean | delta_trimmed_mean_ci_low | delta_trimmed_mean_ci_high | p_superiority | median_abs_radius_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_expansion | 1069 | 187 | 1 | 0.902176 | 0.740696 | 1.0733 | 0.662182 | 0.000133551 |
| model_sensitivity | 1069 | 187 | 1 | 1.87935 | 1.75228 | 2.0263 | 0.807359 | 0.000133551 |

## Provenance

- Cube rows: `1140` total, `760` heldout.
- Cube H5 action/kinematic join source: `source_h5`; cached Lance exists: `True`.
- R3v2 augmented kinematic dim: `70`; native velocity keys: `prev_qvel, proprio_gripper_vel, proprio_joint_vel, qvel`; gripper keys: `proprio_gripper_contact, proprio_gripper_opening, proprio_gripper_vel`.
- Cube full-whitening condition number: `4.28531e+06`.
- Fetch validation rows: `9400`.
- Bootstrap samples and seed: `1000` / `260727`.

## Output Files

- `runs/metric_hardening_v2/cube/r1_action_matched_summary.csv`
- `runs/metric_hardening_v2/cube/r2_whitening_robustness.csv`
- `runs/metric_hardening_v2/cube/r3_aliasing_discriminator.csv`
- `runs/metric_hardening_v2/cube/r3v2_symmetric_aliasing_diagnostic.csv`
- `runs/metric_hardening_v2/cube/struggle_probe_contrasts.csv`
- `runs/metric_hardening_v2/cube/r4_k_sweep.csv`
- `runs/metric_hardening_v2/fetch/r1_action_matched_summary.csv`
- `runs/metric_hardening_v2/fetch/r4_k_sweep.csv`
- `runs/metric_hardening_v2/fetch/r5_density_matched.csv`
- `runs/metric_hardening_v2/figures/k_sweep_delta_ci.png`
- `runs/metric_hardening_v2/decision.json`
