# Dataset Verification

- Dataset: `reacher_eval.lance`
- Format: `LanceDataset`
- Rows: `2010000`
- Episodes: `10000`
- Required columns passed: `True`
- Reference: `reacher.h5`
- Reference rows: `2010000`
- Reference episodes: `10000`

## Sampled Parity
- `action`: passed (max_abs_diff=1.947395400492269e-08)
- `observation`: passed (max_abs_diff=0.0)
- `qpos`: passed (max_abs_diff=9.836690884057475e-08)
- `qvel`: passed (max_abs_diff=1.0880927003853458e-07)
- `target_pos`: passed (max_abs_diff=3.592307409872042e-09)
- `reward`: passed (max_abs_diff=0.0)
- `score`: passed (max_abs_diff=2.2351741811588166e-10)
- `success`: passed (max_abs_diff=0.0)
- `terminated`: passed (max_abs_diff=0.0)
- `truncated`: passed (max_abs_diff=0.0)
- `step_idx`: passed (max_abs_diff=0.0)
