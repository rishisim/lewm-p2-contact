"""Pin check 2 linear peg/T readout on the main run's frame pool.

The pin2 arms train on peg_mixedpolicy.h5 with seed 0, i.e. the same episode split as
ft_mixed_s0, so adding them to the split list leaves the held-out pool unchanged. The pool
identity is checked against the main readout before anything is reported.
"""

import argparse
import json
from pathlib import Path

from lewm_research.probe.readout import SPLIT_CHECKPOINTS, run_readout

MAIN = Path("~/lewm-work/runs/main").expanduser()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", default="pin2_ft_mixed_long,pin2_scratch_mixed")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    names = args.checkpoints.split(",")
    result = run_readout(str(MAIN / "construction/bases.json"), names, seed=42,
                         run_dir=args.run_dir, device=args.device,
                         split_checkpoints=tuple(SPLIT_CHECKPOINTS) + tuple(names))
    main_result = json.loads((MAIN / "readout/readout.json").read_text())
    ours = json.loads((Path(args.run_dir) / "readout.json").read_text())
    same = ours["pool_identity"] == main_result["pool_identity"]
    print(json.dumps({"pool_identity_matches_main": same, "pool_identity": ours["pool_identity"]}))
    if not same:
        raise SystemExit("frame pool differs from the main readout")
    rows = []
    for name in names:
        features = ours["checkpoints"][name]["features"]
        for feature in ("projected", "cls"):
            ridge, mean = features[feature]["ridge"], features[feature]["mean"]
            rows.append({"checkpoint": name, "feature": feature,
                         "peg": ridge["peg"]["mean"], "peg_ci": ridge["peg"]["ci"],
                         "T": ridge["T"]["mean"], "T_ci": ridge["T"]["ci"],
                         "mean_peg": mean["peg"]["mean"], "mean_T": mean["T"]["mean"]})
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
