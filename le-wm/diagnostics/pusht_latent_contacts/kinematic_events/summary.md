# Kinematic Event Diagnostic

- Records: `le-wm/diagnostics/pusht_latent_contacts/prediction_errors.csv`
- State source: `/Users/rishisim/Documents/research/World Models/le-wm/.cache/diagnostic/dataset/pusht_subset_n30_fs5_seed0.h5`
- Input prediction records: 727
- Records retained after requiring t, t-1, t-2 states: 727
- Dropped for insufficient history/state: 0
- Event percentile: p90
- High-MSE threshold: p90 = 0.253634
- Window: [-20, +20] model steps
- Raster: `kinematic_event_raster.png`
- All-event aligned average: `kinematic_event_aligned_average.png`

## Metric Note

`block_velocity_discontinuity` and `block_acceleration_proxy` are the same second-difference norm on the strided block-position series; both columns are written for auditability, but the event is reported once.

## Enrichment And Aligned Lift

| event | threshold_p90 | event_rate_pct | trajectories | p_high_mse_given_event | p_event_given_high_mse | odds_ratio | fisher_p_greater | aligned_lift_t0_over_tminus5 | mean_mse_tplus5 | localized_peak_pass | position_adjusted_odds_ratio | position_adjusted_lr_p | trajectory_mean_high_mse_delta | trajectory_wilcoxon_p | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Block velocity/acceleration discontinuity | 19.03 | 10.04 | 23 | 0.2055 | 0.2055 | 2.658 | 0.003353 | 1.683 | 0.1363 | True | 2.737 | 0.003718 | 0.1411 | 0.005785 | explains |
| Pusher acceleration proxy | 87.73 | 10.04 | 24 | 0.2055 | 0.2055 | 2.658 | 0.003353 | 1.356 | 0.1252 | True | 2.511 | 0.007607 | 0.06581 | 0.4152 | partial/promising |
| Joint acceleration proxy | 88.3 | 10.04 | 24 | 0.2192 | 0.2192 | 2.94 | 0.001131 | 1.396 | 0.1221 | True | 2.761 | 0.002806 | 0.1039 | 0.242 | partial/promising |
| Orientation delta | 0.2562 | 10.04 | 25 | 0.2466 | 0.2466 | 3.564 | 9.804e-05 | 2.081 | 0.1357 | True | 7.484 | 1.333e-07 | 0.1408 | 0.2256 | partial/promising |
| Normalized trajectory position | 0.9091 | 10.59 | 30 | 0.1688 | 0.1781 | 1.997 | 0.03382 | 1.27 | nan | False | 1.056 | 0.8937 | 0.08515 | 0.4795 | no |

## Position And Cluster Controls

The row-level Fisher p-values are retained for continuity but are optimistic because records cluster within trajectories. The `position_adjusted_*` columns fit `high_mse ~ event + normalized_position`; the trajectory columns aggregate within episodes before testing whether event steps have higher high-MSE rates.

## Verification Checks

- block velocity discontinuity equals block acceleration proxy: True
- angle wrapping pi-to-minus-pi smoke value: 0.02
- angle wrapping smoke pass: True
- block_kinematic_discontinuity all-event alignment n@0 equals event count: True
- pusher_acceleration_proxy all-event alignment n@0 equals event count: True
- joint_acceleration_proxy all-event alignment n@0 equals event count: True
- orientation_delta all-event alignment n@0 equals event count: True
- late_trajectory_position all-event alignment n@0 equals event count: True
- normalized_position uses raw_step / (episode_raw_length - 1): True

## Interpretation

Block velocity/acceleration discontinuity best explains the MSE spikes: it passes the event-rate, high-MSE enrichment, aligned-lift, position-control, trajectory-level, and localized-peak criteria.
