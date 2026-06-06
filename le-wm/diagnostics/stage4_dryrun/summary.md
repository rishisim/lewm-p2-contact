# PushT LeWM Latent Contact Diagnostic

- Timestamp: 2026-05-25 21:51:02
- Model repo: `quentinll/lewm-pusht`
- Dataset repo: `quentinll/lewm-pusht`
- Device: `mps`
- Trajectories evaluated: 3
- Trajectories with contact onset: 3
- Prediction records: 87
- Contact source(s): geometry-from-state
- Frameskip: 5
- History size: 3

## Error Summary

- Overall next-latent MSE mean +/- std: 0.125571 +/- 0.10264
- `n_contacts == 0`: 0.111872 +/- 0.0893394 across 69 predictions
- `n_contacts > 0`: 0.178083 +/- 0.132992 across 18 predictions
- Contact/non-contact mean ratio: 1.592

## Files

- `prediction_errors.csv`
- `prediction_error_by_contact_timestep.png`
- `prediction_error_contact_split.png`
