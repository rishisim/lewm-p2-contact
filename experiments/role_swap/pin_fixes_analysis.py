"""Paired G-only analysis of the first N main-run bases and optional pin arms."""

import argparse
import json
from pathlib import Path

from lewm_research.probe.analysis import write_json
from lewm_research.probe.report import (_contrast, _number, _paired, _readout_summary,
                                       _success_map, _table)

HARD = "move_peg"
CONTROL = "move_T_matched"


def _episodes(root, ids):
    path = root / "episodes.jsonl"
    if not path.exists():
        return None
    with path.open() as file:
        return [row for line in file if (row := json.loads(line))["base_id"] in ids
                and row["condition"] in (HARD, CONTROL)]


def analyze(main_root, arm_dirs, readout_root, n=100):
    if n < 1:
        raise ValueError("n must be positive")
    bases = json.loads((main_root / "construction/bases.json").read_text())
    ordered_ids = [b["id"] for b in bases[:n]]
    ids = set(ordered_ids)
    main = _episodes(main_root / "ft_mixed_s0", ids)
    feasibility = _episodes(main_root / "reference-F", ids)
    reference = _episodes(main_root / "reference-E", ids)
    if main is None or feasibility is None or reference is None:
        raise ValueError("main ft_mixed_s0, reference-F, and reference-E episodes required")
    fscore = _success_map(feasibility)
    feasible = {b for b in ids if fscore.get((b, HARD)) == 1 and fscore.get((b, CONTROL)) == 1}
    seeds = {(r["base_id"], r["condition"]): r["seed"] for r in main}
    if len(seeds) != len(main):
        raise ValueError("duplicate main evaluation episodes")
    result = {"base_ids": ordered_ids, "requested_n": n, "seed": 42, "bootstrap_samples": 2000,
              "feasibility": {"included_base_ids": sorted(feasible),
                  "missing_base_ids": sorted(b for b in ids if any((b, c) not in fscore for c in (HARD, CONTROL)))},
              "arms": {}, "missing_arms": []}
    baseline = _success_map(main)
    for arm, root in {"ft_mixed_s0": main_root / "ft_mixed_s0", **arm_dirs}.items():
        rows = main if arm == "ft_mixed_s0" else _episodes(root, ids)
        if rows is None:
            result["missing_arms"].append(arm)
            continue
        seen = set()
        for row in rows:
            key = (row["base_id"], row["condition"])
            if key in seen:
                raise ValueError(f"duplicate evaluation episode: {arm}, {key}")
            seen.add(key)
            if key not in seeds or row["seed"] != seeds[key]:
                raise ValueError(f"episode seed differs from main ft_mixed_s0: {arm}, {key}")
        scores = _success_map(rows)
        entry = {"run_dir": str(root), "readout": None}
        for mode, allowed in (("common_feasible", feasible), ("unconditional", ids)):
            learned = _contrast(rows, HARD, CONTROL, allowed, 42, 2000)
            # Reference E and differences use exactly the arm's paired bases.
            paired_ids = learned["base_ids"]
            ref = _contrast(reference, HARD, CONTROL, set(paired_ids), 42, 2000)
            shared = [b for b in paired_ids if all((b, c) in baseline for c in (HARD, CONTROL))]
            gap = {b: scores[(b, CONTROL)] - scores[(b, HARD)] for b in shared}
            main_gap = {b: baseline[(b, CONTROL)] - baseline[(b, HARD)] for b in shared}
            entry[mode] = {"learned": learned, "reference_E": ref,
                "difference_of_gaps_vs_ft_mixed_s0": _paired(gap, main_gap, shared, 42, 2000),
                "difference_base_ids": shared}
        readout_path = (main_root / "readout" if arm == "ft_mixed_s0" else readout_root) / f"{arm}.json"
        if readout_path.exists():
            readout = json.loads(readout_path.read_text())
            # Pool readouts may omit the optional precomputed pixel baseline.
            readout["features"].setdefault("pixel", {"episode_errors": {}})
            entry["readout"] = _readout_summary([readout], 42, 2000)
            for kind in ("cls", "projected"):
                metrics = entry["readout"][kind]
                if "peg_over_T" not in metrics:
                    denominator = metrics["T"]["mean"]
                    metrics["peg_over_T"] = {"mean": metrics["peg"]["mean"] / denominator if denominator else None}
        result["arms"][arm] = entry
    return result


def markdown(result):
    success, readout = [], []
    for arm, entry in result["arms"].items():
        for mode in ("common_feasible", "unconditional"):
            e = entry[mode]
            learned = e["learned"]
            success.append([arm, mode, len(learned["base_ids"]), _number(learned["hard"]),
                _number(learned["control"]), _number(learned["gap_control_minus_hard"]),
                _number(e["reference_E"]["gap_control_minus_hard"]),
                _number(e["difference_of_gaps_vs_ft_mixed_s0"])])
        for kind, metrics in (entry["readout"] or {}).items():
            if kind in ("cls", "projected", "mean"):
                readout.append([arm, kind, _number(metrics["peg"]), _number(metrics["T"]),
                                _number(metrics.get("peg_over_T"))])
    tables = [_table(["Arm", "Set", "Bases", "Hard success", "Control success",
        "Gap control-hard (95% CI)", "Reference-E gap", "Gap minus ft_mixed_s0"], success),
        _table(["Arm", "Readout", "Peg error px (95% CI)", "T error px (95% CI)", "Peg/T ratio"], readout)]
    return "\n\n".join(tables) + "\n\nMissing arms: " + (", ".join(result["missing_arms"]) or "none") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=Path("~/lewm-work/runs/main"))
    parser.add_argument("--idm-run", type=Path, default=Path("~/lewm-work/runs/pin-fixes/eval/pin3_idm"))
    parser.add_argument("--pegsup-run", type=Path, default=Path("~/lewm-work/runs/pin-fixes/eval/pin3_pegsup"))
    parser.add_argument("--readout-root", type=Path, default=Path("~/lewm-work/runs/pin-fixes/readout"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.main_root.expanduser(),
        {"pin3_idm": args.idm_run.expanduser(), "pin3_pegsup": args.pegsup_run.expanduser()},
        args.readout_root.expanduser(), args.n)
    for path in (args.out_json, args.out_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_json, result)
    text = markdown(result)
    args.out_md.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
