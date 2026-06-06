# Stage 1 Report: Proposal 1 Preliminary Test

## Stage 0 Timing + Fallback

| Item | Value | Unit/Notes |
| --- | --- | --- |
| Load/device | True | mps |
| Param count | 18,034,478 | 18.03M |
| Encode | 3.344504 | ms/frame |
| Predictor fwd | 3.159125 | ms |
| Predictor fwd+bwd+step | 18.467694 | ms |

MPS fallback status: Check terminal log for PyTorch MPS fallback warnings emitted outside Python.

## 1a Extraction Summary

- Episodes: 30 total; 20 train trajectories, 10 test trajectories.
- Pairs: 505 train, 255 test, 760 total.
- History latents: `(760, 3, 192)`; actions: `(760, 3, 10)`; targets: `(760, 192)`.
- Latent stats: mean(mean_dim)=0.000454, mean(std_dim)=0.993174, any_nan=False.
- Contact labels: geometry-from-state with 238 contact pairs and 522 non-contact pairs.

## 1b Three-Way Comparison

History-only probe: 576->512->192 ReLU MLP, 2000 epochs, final train MSE 0.000634, final test MSE 1.155759.

| Method | Abs MSE | Normalized MSE | Fraction of Persistence |
| --- | --- | --- | --- |
| Persistence | 0.234855 | 0.236829 | 1.000000 |
| History-only MLP | 1.155759 | 1.165471 | 4.921154 |
| Full LeWM | 0.007732 | 0.007797 | 0.032924 |

- Action information value: `0.993310`
- Predictive headroom over persistence: `-3.921154`

## 1c Contact Split

| Subset | n | Method | Abs MSE | Normalized MSE | Fraction of Persistence | Action Info Value |
| --- | --- | --- | --- | --- | --- | --- |
| Non-contact | 176 | Persistence | 0.218496 | 0.220374 | 1.000000 |  |
| Non-contact | 176 | History-only MLP | 1.125003 | 1.134675 | 5.148855 | 0.993488 |
| Non-contact | 176 | Full LeWM | 0.007327 | 0.007390 | 0.033532 |  |
| Contact | 79 | Persistence | 0.271302 | 0.292126 | 1.000000 |  |
| Contact | 79 | History-only MLP | 1.224278 | 1.318252 | 4.512610 | 0.992946 |
| Contact | 79 | Full LeWM | 0.008636 | 0.009299 | 0.031833 |  |

## Plain-Language Reading

The history-only probe is worse than persistence on the held-out trajectories, despite fitting the probe-train pairs very closely. That makes the formal action information value hard to interpret by itself: it is large because the frozen full LeWM model is strong and the trained history-only probe overfits/fails to generalize, not because this run cleanly isolates only action signal. The frozen action-conditioned LeWM lowers held-out MSE by 99.3% relative to the history-only probe.

## Divergences, Caveats, Confounds

- No discrepancy for the loaded object checkpoint: AutoCostModel('pusht/lewm') resolves to jepa.JEPA from le-wm/jepa.py.
- Note: stable-worldmodel-readonly also contains stable_worldmodel.wm.lewm.lewm.LeWM with the same one-step predict signature but a different rollout implementation; it is not the loaded object checkpoint class in this environment.
- The HDF5 dataset has no n_contacts column; contact/non-contact uses the prior geometric fallback from 7D state.
- Known confound: the history-only probe is trained on latents from an already action-conditioned trained encoder, so this is a proxy and likely a lower bound on action-free pretraining plausibility.
- No explicit MPS CPU fallback warnings appeared in the captured Stage 0 terminal log.

Stage 2 was not run.
