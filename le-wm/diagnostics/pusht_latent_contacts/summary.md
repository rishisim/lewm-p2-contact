# PushT LeWM Latent Contact Diagnostic

- Timestamp: 2026-05-25 21:51:55
- Model repo: `quentinll/lewm-pusht`
- Dataset repo: `quentinll/lewm-pusht`
- Device: `mps`
- Trajectories evaluated: 30
- Trajectories with contact onset: 28
- Prediction records: 727
- Contact source(s): geometry-from-state
- Frameskip: 5
- History size: 3

## Error Summary

- Overall next-latent MSE mean +/- std: 0.118942 +/- 0.114617
- `n_contacts == 0`: 0.108736 +/- 0.108431 across 476 predictions
- `n_contacts > 0`: 0.138295 +/- 0.123439 across 251 predictions
- Contact/non-contact mean ratio: 1.272

## Files

- `prediction_errors.csv`
- `prediction_error_by_contact_timestep.png`
- `prediction_error_contact_split.png`
