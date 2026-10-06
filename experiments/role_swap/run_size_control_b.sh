#!/usr/bin/env bash
# Size-control Phase B (SIZE_CONTROL.md): radius-45 G evaluation of ft45 checkpoints.
# Construction and the reference lane start at once; learned arms wait for checkpoints.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export LEWM_WORK_ROOT="${LEWM_WORK_ROOT:-$HOME/lewm-work}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
root="$LEWM_WORK_ROOT/runs/size-control/phaseB"; mkdir -p "$root/logs"
n=100; master=310000000; arms=(ft45_block_s0 ft45_mixed_s0)
normalization="$LEWM_WORK_ROOT/runs/w4d-calibration/normalization.json"
run() { local label="$1"; shift; echo "$(date -u '+%FT%TZ') start $label"
        uv run lewm "$@" >> "$root/logs/$label.log" 2>&1; echo "$(date -u '+%FT%TZ') finished $label"; }
[[ -f "$root/construction/bases.json" ]] || run construction probe-bases --n "$n" --seed "$master" --near-path-distance 55 --peg-radius 45 --run-dir "$root/construction"
bases="$root/construction/bases.json"
eval_args=(--bases "$bases" --conditions move_peg,move_T_matched --n "$n" --budget 50 --population 300 --iterations 30 --topk 30 --normalization "$normalization")
(
  run reference-F probe-eval "${eval_args[@]}" --arm reference --seed 101 --workers 5 --approach-weight .1 --device cpu --with-target --run-dir "$root/reference-F"
  run reference-E probe-eval "${eval_args[@]}" --arm reference --seed 202 --workers 5 --approach-weight .1 --device cpu --with-target --run-dir "$root/reference-E" --feasibility-run "$root/reference-F"
) &
reference_pid=$!
for arm in "${arms[@]}"; do
  until [[ -f "$LEWM_WORK_ROOT/stable-worldmodel/checkpoints/$arm/weights.pt" && -f "$LEWM_WORK_ROOT/stable-worldmodel/checkpoints/$arm/train_summary.json" ]]; do sleep 60; done
done
report_args=(--runs "$root/reference-E")
for arm in "${arms[@]}"; do
  run "$arm" probe-eval "${eval_args[@]}" --arm "$arm" --seed 42 --workers 1 --device mps --no-with-target --run-dir "$root/$arm"
  report_args+=(--runs "$root/$arm")
done
run readout probe-size-control --checkpoints "$(IFS=,; echo "${arms[*]}")" --radii 15,45 --device mps --run-dir "$root/readout"
wait "$reference_pid"
run report probe-report "${report_args[@]}" --feasibility-run "$root/reference-F" --seed 42 --run-dir "$root/report"
date -u '+%FT%TZ' > "$root/DONE"
