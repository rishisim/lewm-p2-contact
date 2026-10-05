# PushT replication pilot notes

- 2026-07-18 — Created the single authorized run root. Prior runs and cached model inputs remain read-only; no HDF5/H5 file will be opened.

- 2026-07-18T19:04:05Z — Excluded compatibility smoke passed on mps; projected full wall time 3872.7s, permanently locking full counts {'fit': 120, 'selection': 40, 'evaluation': 80}.

- 2026-07-18T19:04:46Z — Generated fixed fit role: 120 fresh episodes, 2160 transitions, no exclusions/replacements; raw pixels discarded after bounded encoding.

- 2026-07-18T19:05:02Z — Generated fixed selection role: 40 fresh episodes, 720 transitions, no exclusions/replacements; raw pixels discarded after bounded encoding.

- 2026-07-18T19:05:14Z — First PushT refiner/gain-heterogeneity table completed: signal_present=True, checkpoint=29be92d39a2efba992c8e2dcfd289956d04f057d63d8ed1042de25c305641a6f.

- 2026-07-18T19:05:37Z — Fit all six fixed stage-specific gate candidates on fit episodes only; locked coefficients and thresholds before selection access.

- 2026-07-18T19:05:37Z — Selection opened once and froze stage_dual_r10_q0.75 without refit; evaluation trajectories did not yet exist.

- 2026-07-18T19:06:07Z — Generated fixed evaluation role: 80 fresh episodes, 1440 transitions, no exclusions/replacements; raw pixels discarded after bounded encoding.

- 2026-07-18T19:06:22Z — Opened the fixed evaluation outcomes exactly once and locked proposed label pusht_replication_pilot_not_supported; raw/white benefits 0.003909797/-0.00030990013.

- 2026-07-18T19:07:04Z — Finalized terminal scientific label pusht_replication_pilot_not_supported; independent verification passed=True.
