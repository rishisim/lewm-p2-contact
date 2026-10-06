# Step-3 fine-tuning results

Completed 2026-10-05 (America/Chicago). Instance **54407575**, **1× NVIDIA H100 SXM 80 GB**, offer 50189137; **destroyed**. Hourly price including 80 GB storage: **$1.844444/h** ($1.80 GPU + $0.044444 storage). Total instance wall time: **67 min 11.5 s** (2026-10-06T01:06:08+00:00 to 2026-10-06T02:13:19+00:00). **Total cost: $2.14**, from the post-destruction Vast charge ledger, including GPU, disk, and network transfer; credit deduction was $2.139339. The $5.00 cap was respected; no payment methods or account settings were changed.

Four complete runs, seeds 0 and 1, in the required priority order. Seed 2 was omitted under the authorized time-budget fallback: six runs at the observed ~12.45 minutes/run plus setup and retrieval would exceed the 85-minute target. Every completed run retained the full 2,000-step budget.

All runs: bf16 CUDA autocast, batch size 128, eight data-loader workers, AdamW lr 5e-5 / weight decay 1e-3, unchanged upstream objective and scheduler, validation and atomic checkpoints every 250 steps, per-step loss logs. Runs were sequential, launched with nohup; observed peak GPU memory was 14,781 MiB (14.44 GiB).

| Run | Steps | Final train_pred_loss | Final val_pred_loss | Mean GPU s/step | Wall s/step* | Training wall (s)* |
|---|---:|---:|---:|---:|---:|---:|
| ft_block_s0 | 2000 | 0.04098266 | 0.04389345 | 0.13786 | 0.36664 | 733.28 |
| ft_mixed_s0 | 2000 | 0.02245078 | 0.02583770 | 0.13895 | 0.36610 | 732.20 |
| ft_block_s1 | 2000 | 0.03789269 | 0.04419118 | 0.13756 | 0.36541 | 730.81 |
| ft_mixed_s1 | 2000 | 0.01957759 | 0.02234315 | 0.13853 | 0.36668 | 733.35 |

*Wall timing from train_summary.json includes data loading, validation, and checkpoint saves, but excludes model/dataset initialization. GPU s/step is the mean logged optimizer-step timing. The separate 50-step throughput check took 46.85 seconds including its final full validation; logged GPU training steps were about 0.14 seconds. Sustained training including data loading was 0.23845 seconds/step before validation pauses.

**val_pred_loss is the primary validation metric.** Validation SIGReg is inflated by ordered validation windows and is diagnostic only. Final val_sigreg_loss values, in table order: 28.412327, 44.270525, 29.790199, 47.438905. These training results do not establish planning/control performance.

Validation and preservation: all seven uploaded dataset/JSON/pretrained files matched local SHA-256; CUDA and bf16 support were confirmed; `uv run pytest -q tests/test_finetune.py` passed (21 tests); all `lewm gates --device cuda` gates passed. The SIGKILL test stopped ft_resume_check at step 45, verified its atomic trainer state at step 40, and resumed from 40 to finish at 60. Its log covered steps 1–60, with five replayed entries for 41–45. The background sync copied its checkpoint locally and `load_lewm` loaded that copy; the temporary check and throughput checkpoints were then deleted.

A local background sync used `rsync -az --partial --delay-updates` with serialized transfers every five minutes. Final retrieval verified SHA-256 of all 24 checkpoint artifacts. All four local models loaded on CPU; all trainer states contained version-2 resume metadata, model/optimizer/scheduler/counters/RNG and step 2,000. Local checkpoints are `~/lewm-work/stable-worldmodel/checkpoints/{ft_block_s0,ft_mixed_s0,ft_block_s1,ft_mixed_s1}/`, each containing weights.pt, config.json, normalization.json, train_log.jsonl, train_summary.json, and trainer_state.pth. Runtime logs are retained outside Git in `~/lewm-work/logs/vast_54407575/`; sync log: `~/lewm-work/logs/gpu_sync.log`.

Only the frozen `scratchpad/gpu_snapshot/` was transferred to `/workspace/repo`; no .venv was copied and no code was edited on the instance. Its locked environment was installed with `uv sync --locked`: torch 2.14.1+cu130, CUDA/bf16 verified, on the requested pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel image. Missing OS graphics libraries were installed to support the existing OpenCV dependency. Fine-tuning source SHA-256: `9e33086af3607dc90c277ede326bdba6ecb1d8b2f4cd5b6128aec2d5622708f7`.

To resume an interrupted run on another CUDA GPU: install the same frozen code with `uv sync --locked`; copy the identical datasets and lewm-pusht initialization/normalization to its work root. Copy the entire synced checkpoint directory from `~/lewm-work/stable-worldmodel/checkpoints_remote_sync/ft_<arm>_s<seed>/` into that machine’s `$LEWM_WORK_ROOT/stable-worldmodel/checkpoints/` with the same directory name. Completed runs have already been moved to the canonical local checkpoints directory. Rerun the identical command, for example:

```sh
cd /workspace/repo
export LEWM_WORK_ROOT=/workspace/lewm-work
export OMP_NUM_THREADS=8
uv run lewm finetune --dataset peg_blockpolicy.h5 --name ft_block_s0 \
  --init lewm-pusht --max-steps 2000 --batch-size 128 --seed 0 \
  --device cuda --precision bf16 --workers 8 \
  --val-interval 250 --checkpoint-interval 250
```

Select the original dataset/name/seed for another run. Resume is automatic by name from trainer_state.pth; keep the seed, batch size, 2,000-step scheduler budget, and input file content identical. Portable content fingerprints reject changed inputs. Uncheckpointed steps are replayed and may appear twice in the append-only log; use the last entry for each step when analyzing an interrupted run.

Local watchdog PID 95214 was started immediately after creation with T=8783 seconds and cancelled after destruction. The supervisor watchdog was left untouched. Temporary SSH keys/configuration were removed. Post-destruction verification: `vastai show instances --raw` returned `[]`; human output said “Total: 0 instances” and “No instances found.” No commits or pushes were made.
