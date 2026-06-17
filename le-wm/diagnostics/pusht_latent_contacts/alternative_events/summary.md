# Alternative Event Diagnostic

- Records: `le-wm/diagnostics/pusht_latent_contacts/prediction_errors.csv`
- State source: `/Users/rishisim/Documents/research/World Models/le-wm/.cache/diagnostic/dataset/pusht_subset_n30_fs5_seed0.h5`
- Prediction records: 727
- Trajectories: 30
- Block speed threshold: 0.01
- State-delta percentile threshold: p90 = 225.567
- Proximity threshold: 0.08 (raw pixels / 512)
- Figure: `alternative_event_time_aligned_grid.png`

## Stage 1 Event Rates

| event | event_rate_pct | trajectories | flag |
| --- | --- | --- | --- |
| Event B: block velocity onset | 4.127 | 30 | too sparse |
| Event C: large state change | 10.04 | 22 | ok |
| Event D: proximity threshold | 2.201 | 16 | too sparse |

## Stage 3 Onset Sharpness

| event | onset_sharpness | mean_mse_t_minus_5 | mean_mse_t0 |
| --- | --- | --- | --- |
| Original geometric contact | 1.151 | 0.09323 | 0.1073 |
| Event B: block velocity onset | 1.728 | 0.05805 | 0.1003 |
| Event C: large state change | 1.124 | 0.1438 | 0.1616 |
| Event D: proximity threshold | 1.434 | 0.1035 | 0.1485 |

## Verdict

VERDICT D2 -- Even with task-agnostic dynamic discontinuity as the event definition, no localized prediction error spike appears. The foundational claim does not hold for this task in its current form.

Caveat: Event B and Event D have onset ratios above 1.3, but both are below the 5% event-rate floor and neither forms a localized t=0 peak in the grid. Event B steps up after onset, and Event D is noisy with later peaks larger than t=0.
