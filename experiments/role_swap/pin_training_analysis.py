"""Compare training arms on one reference-F-feasible, paired G base set."""

import argparse
import json
from pathlib import Path

from lewm_research.probe.analysis import contrasts, fingerprint, write_json
from lewm_research.probe.conditions import load_bases
from lewm_research.probe.report import _contrast, compute_report


MAIN = Path("~/lewm-work/runs/main")
SEED = 42
SAMPLES = 2000


def read_rows(root, selected, conditions):
    path = root / "episodes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["base_id"] in selected and r["condition"] in conditions]


def percentages(contrast, hard, control):
    def scale(metric):
        return {key: (100 * value if key in ("mean", "difference") and value is not None
                      else [100 * v for v in value] if key == "ci" and value is not None
                      else value) for key, value in metric.items()}
    return {"success_percent": {hard: scale(contrast["hard"]),
                                control: scale(contrast["control"])},
            "G_gap_percentage_points": scale(contrast["gap_control_minus_hard"])}


def analyze(bases_path, n, feasibility_root, reference_root, arms):
    bases = load_bases(bases_path)
    if n < 1 or n > len(bases):
        raise ValueError("n must be between 1 and the number of bases")
    selected_ids = [b.id for b in bases[:n]]
    if len(set(selected_ids)) != n:
        raise ValueError("selected base IDs must be unique")
    selected = set(selected_ids)
    hard, control = contrasts()["G"]
    conditions = (hard, control)
    feasibility = read_rows(feasibility_root, selected, conditions)
    reference = read_rows(reference_root, selected, conditions)
    rows = {name: read_rows(root, selected, conditions) for name, root in arms.items()}
    # Reuse the report's exact F definition: both G conditions must succeed in F.
    report = compute_report(feasibility, feasibility, reference, seed=SEED, samples=SAMPLES)
    feasible = selected & set(report["feasibility"]["G"]["included_base_ids"])
    common = feasible.copy()
    available = {}
    for name, records in {**rows, "reference_E": reference}.items():
        paired = _contrast(records, hard, control, feasible, SEED, SAMPLES)
        available[name] = set(paired["base_ids"])
        common &= available[name]
    if not common:
        raise ValueError("no common paired reference-F-feasible bases")
    reference_metrics = percentages(_contrast(reference, hard, control, common, SEED, SAMPLES),
                                    hard, control)
    source_paths = [bases_path, feasibility_root / "episodes.jsonl",
                    reference_root / "episodes.jsonl",
                    *(root / "episodes.jsonl" for root in arms.values())]
    result = {"n_requested": n, "selected_base_ids": selected_ids,
              "reference_F_feasible_base_ids": sorted(feasible),
              "common_base_ids": sorted(common), "n_common_feasible": len(common),
              "missing_paired_feasible_base_ids": {k: sorted(feasible - v) for k, v in available.items()},
              "conditions": list(conditions), "seed": SEED, "bootstrap_samples": SAMPLES,
              "sources_sha256": {str(p): fingerprint(p) for p in source_paths}, "arms": {}}
    for name, records in rows.items():
        result["arms"][name] = {
            "run_dir": str(arms[name]), "n_common_feasible": len(common),
            **percentages(_contrast(records, hard, control, common, SEED, SAMPLES), hard, control),
            "reference_E": reference_metrics}
    return result


def markdown(result):
    hard, control = result["conditions"]
    def number(metric):
        value = metric.get("mean", metric.get("difference"))
        low, high = metric["ci"]
        return f"{value:.1f} [{low:.1f}, {high:.1f}]"
    lines = [f"| Arm | Feasible n | {hard} % (95% CI) | {control} % (95% CI) | G gap pp (95% CI) |",
             "| --- | --- | --- | --- | --- |"]
    for name, arm in result["arms"].items():
        lines.append(f"| {name} | {arm['n_common_feasible']} | "
                     f"{number(arm['success_percent'][hard])} | "
                     f"{number(arm['success_percent'][control])} | "
                     f"{number(arm['G_gap_percentage_points'])} |")
    reference = next(iter(result["arms"].values()))["reference_E"]
    lines.append(f"| reference-E | {result['n_common_feasible']} | "
                 f"{number(reference['success_percent'][hard])} | "
                 f"{number(reference['success_percent'][control])} | "
                 f"{number(reference['G_gap_percentage_points'])} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bases", type=Path, default=MAIN / "construction/bases.json")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--feasibility-run", type=Path, default=MAIN / "reference-F")
    parser.add_argument("--reference-run", type=Path, default=MAIN / "reference-E")
    parser.add_argument("--arm-runs", action="append", metavar="NAME=DIR",
                        help="Repeat for each probe-eval arm; defaults to ft_mixed_s0.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    arms = {}
    for spec in args.arm_runs or [f"ft_mixed_s0={MAIN / 'ft_mixed_s0'}"]:
        name, separator, directory = spec.partition("=")
        if not separator or not name or not directory or name in arms or name == "reference_E":
            parser.error("--arm-runs requires unique NAME=DIR entries (reference_E is reserved)")
        arms[name] = Path(directory).expanduser().resolve()
    result = analyze(args.bases.expanduser().resolve(), args.n,
                     args.feasibility_run.expanduser().resolve(),
                     args.reference_run.expanduser().resolve(), arms)
    write_json(args.out.expanduser(), result)
    print(markdown(result))


if __name__ == "__main__":
    main()
