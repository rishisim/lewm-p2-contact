#!/usr/bin/env bash
# Frozen W6 evaluation. Rerun this command to resume completed episodes/stages.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export LEWM_WORK_ROOT="${LEWM_WORK_ROOT:-$HOME/lewm-work}"
export PYTHONUNBUFFERED=1
# Keep CPU numerical libraries from oversubscribing the reference workers.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

mode="${1:-main}"
case "$mode" in
  main) n=400; master=300000000; root="$LEWM_WORK_ROOT/runs/main" ;;
  --dry-run) n=3; master=290000000; root="$LEWM_WORK_ROOT/runs/main-dry-run" ;;
  *) echo "Usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
root="$(uv run python -c 'import sys; from lewm_research.probe.rollout_eval import prepare_output_root; print(prepare_output_root(sys.argv[1]))' "$root")"
# External work roots cannot be tracked by this repository; publication roots
# are explicitly ignored. Fail before generating any output if that changes.
git check-ignore -q experiments/role_swap/results/RESULTS.md
git check-ignore -q experiments/role_swap/results/results.json
mkdir -p "$root/logs"
if [[ -f "$root/DONE" ]]; then echo "Already complete: $root"; exit 0; fi
if ! mkdir "$root/.lock" 2>/dev/null; then
  owner="$(cat "$root/.lock/pid" 2>/dev/null || true)"
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
    echo "Main driver already running (PID $owner)" >&2; exit 1
  fi
  rmdir "$root/.lock" 2>/dev/null || { rm -f "$root/.lock/pid"; rmdir "$root/.lock"; }
  mkdir "$root/.lock"
fi
echo "$$" > "$root/.lock/pid"
echo "$$" > "$root/driver.pid"
rm -f "$root/FAILED"
children=()
stop_tree() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" || true); do stop_tree "$child"; done
  kill "$pid" 2>/dev/null || true
}
finish() {
  local status=$?
  trap - EXIT TERM INT
  # Some Bash 3.2 expansion errors reach EXIT with status zero.
  [[ -f "$root/DONE" ]] || { (( status != 0 )) || status=1; }
  if (( status != 0 )); then
    date -u '+%FT%TZ' > "$root/FAILED"
    for pid in ${children[@]+"${children[@]}"}; do stop_tree "$pid"; done
  fi
  rm -f "$root/.lock/pid"
  rmdir "$root/.lock"
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run() {
  local label="$1"; shift
  echo "$(date -u '+%FT%TZ') start $label"
  uv run lewm "$@" >> "$root/logs/$label.log" 2>&1 &
  local pid=$!
  local index=${#children[@]}
  children+=("$pid")
  echo "$pid" > "$root/$label.pid"
  wait "$pid"
  unset 'children[index]'
  echo "$(date -u '+%FT%TZ') finished $label"
}

normalization="$LEWM_WORK_ROOT/runs/w4d-calibration/normalization.json"
uv run python -c 'import sys; from lewm_research.main import check_normalization; check_normalization(sys.argv[1])' "$normalization" >> "$root/logs/preflight.log" 2>&1
run construction probe-bases --n "$n" --seed "$master" --near-path-distance 55 --run-dir "$root/construction"
bases="$root/construction/bases.json"
conditions=off_path,near_path,move_peg,move_T_matched
checkpoints=lewm-pusht,ft_block_s0,ft_block_s1,ft_mixed_s0,ft_mixed_s1
eval_args=(--bases "$bases" --conditions "$conditions" --n "$n" --budget 50 --population 300 --iterations 30 --topk 30 --normalization "$normalization")

driver_pid=$$
(
  # A failure in the independent CPU lane immediately stops the MPS lane.
  trap 'date -u "+%FT%TZ" > "$root/FAILED"; kill -TERM "$driver_pid"' ERR
  run reference-F probe-eval "${eval_args[@]}" --arm reference --seed 101 --workers 5 --approach-weight .1 --device cpu --with-target --run-dir "$root/reference-F"
  run reference-E probe-eval "${eval_args[@]}" --arm reference --seed 202 --workers 5 --approach-weight .1 --device cpu --with-target --run-dir "$root/reference-E" --feasibility-run "$root/reference-F"
) >> "$root/logs/reference-lane.log" 2>&1 &
reference_pid=$!
children+=("$reference_pid")
echo "$reference_pid" > "$root/reference-lane.pid"

report_args=(--runs "$root/reference-E")
for arm in lewm-pusht ft_block_s0 ft_block_s1 ft_mixed_s0 ft_mixed_s1; do
  rendering=--no-with-target
  [[ "$arm" != lewm-pusht ]] || rendering=--with-target
  run "$arm" probe-eval "${eval_args[@]}" --arm "$arm" --seed 42 --workers 1 --device mps "$rendering" --run-dir "$root/$arm"
  report_args+=(--runs "$root/$arm")
done
wait "$reference_pid"
children=()

readout_args=()
abc_n=50
analysis_args=()
publish_args=(--publish --coverage-underpowered)
if [[ "$mode" == --dry-run ]]; then
  # Exercise all checkpoints/stages with a small common pool; CEM stays frozen.
  readout_args=(--frames-per-dataset 30 --outer-folds 3 --inner-folds 2)
  abc_n=3
  analysis_args=(--bootstrap-samples 20)
  publish_args=()
fi
run readout probe-readout --bases "$bases" --checkpoints "$checkpoints" --seed 42 --device mps --run-dir "$root/readout" ${readout_args[@]+"${readout_args[@]}"} ${analysis_args[@]+"${analysis_args[@]}"}
for family in D G; do
  pair=off_path,near_path
  [[ "$family" != G ]] || pair=move_peg,move_T_matched
  run "abc-$family" probe-abc --bases "$bases" --checkpoints "$checkpoints" --conditions "$pair" --n "$abc_n" --seed 42 --device mps --workers 5 --population 300 --iterations 30 --topk 30 --approach-weight .1 --feasibility-run "$root/reference-F" --run-dir "$root/abc-$family" ${analysis_args[@]+"${analysis_args[@]}"}
  report_args+=(--runs "$root/abc-$family")
done
run report probe-report "${report_args[@]}" --runs "$root/readout" --feasibility-run "$root/reference-F" --seed 42 --run-dir "$root/report" ${analysis_args[@]+"${analysis_args[@]}"} ${publish_args[@]+"${publish_args[@]}"}
date -u '+%FT%TZ' > "$root/DONE"
echo "Complete: $root"
