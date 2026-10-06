"""Mechanical application of the frozen Step-4 decision bar."""

from collections import defaultdict
from pathlib import Path
import json
import re
import shutil

import numpy as np

from .analysis import cluster_mean, contrasts, fingerprint, open_run, write_json
from .stats import paired_cluster_bootstrap


def gap_label(gap, reference_difference, complete=True):
    if not complete or gap is None or reference_difference is None:
        return "unavailable"
    return ("gap present" if gap["difference"] >= .15 - 1e-12 and gap["ci"][0] > 0
            and abs(reference_difference) < .05 - 1e-12 else "gap absent")


def localization_labels(peg_error, t_error, pixel_peg_error, hard, control):
    """Apply every localization clause independently; missing data is explicit."""
    labels, missing = [], []
    readout_valid = all(v is not None and np.isfinite(v) for v in (peg_error,t_error,pixel_peg_error))
    if readout_valid:
        if peg_error >= 2*t_error and peg_error >= 1.5*pixel_peg_error:
            labels.append("linear-readout deficit")
    else:
        missing.append("linear-readout deficit")
    if readout_valid and all(v is not None for v in (hard.get("BC"),control.get("BC"))):
        if control["BC"]-hard["BC"] >= .2-1e-12 and peg_error < 1.25*pixel_peg_error:
            labels.append("observed-state cost mismatch")
    else:
        missing.append("observed-state cost mismatch")
    if all(hard.get(k) is not None and control.get(k) is not None for k in ("BC","AB","AC")):
        if (hard["BC"] >= .5 and control["BC"] >= .5
                and control["AB"]-hard["AB"] >= .2-1e-12
                and control["AC"]-hard["AC"] >= .2-1e-12):
            labels.append("prediction-associated degradation")
    else:
        missing.append("prediction-associated degradation")
    return {"labels": labels, "unavailable_clauses": missing}


def coverage_label(hard_gain, control_gain, block_gap, mixed_gap, complete=True):
    if not complete or hard_gain is None or any(v is None for v in (control_gain,block_gap,mixed_gap)):
        return "unavailable"
    shrink = block_gap-mixed_gap
    if (block_gap > 0 and hard_gain["difference"] >= .1-1e-12 and hard_gain["ci"][0] > 0
            and control_gain > -.05+1e-12 and shrink >= .5*block_gap-1e-12):
        return "sufficient remedy"
    return "reduced disparity" if block_gap > 0 and shrink > 0 else "none"


def _family(checkpoint):
    return re.sub(r"_s\d+$", "", checkpoint)


def _success_map(records, field="success"):
    values = defaultdict(list)
    for row in records:
        values[(row["base_id"],row["condition"])].append(float(row["score"][field]))
    return {key: float(np.mean(v)) for key,v in values.items()}


def _paired(left, right, ids, seed, samples):
    if not ids:
        return None
    return paired_cluster_bootstrap([left[i] for i in ids], [right[i] for i in ids], ids, seed, samples)


def _contrast(records, hard, control, allowed, seed, samples, field="success"):
    scores = _success_map(records, field)
    hard_rows = {b:scores[(b,hard)] for b,c in scores if c==hard and (allowed is None or b in allowed)}
    control_rows = {b:scores[(b,control)] for b,c in scores if c==control and (allowed is None or b in allowed)}
    ids = sorted(hard_rows.keys() & control_rows.keys())
    result = {"base_ids": ids, "missing_base_ids": sorted(set(allowed or [])-set(ids)),
              "hard": cluster_mean(list(hard_rows.values()), list(hard_rows), seed, samples),
              "control": cluster_mean(list(control_rows.values()), list(control_rows), seed, samples),
              "gap_control_minus_hard": _paired(control_rows,hard_rows,ids,seed,samples)}
    if ids:
        result["discordance"] = float(np.mean([hard_rows[i] != control_rows[i] for i in ids]))
    return result


def _abc_comparison(records, hard, control, allowed, seed, samples):
    data = defaultdict(list)
    for row in records:
        if row["condition"] in (hard,control) and (allowed is None or row["base_id"] in allowed):
            data[(row["base_id"],row["condition"])].append(row["metrics"])
    ids = sorted({b for b,c in data if c==hard} & {b for b,c in data if c==control})
    result = {"paired_base_ids": ids, "hard": {}, "control": {}, "control_minus_hard": {}}
    for key in ("BC","AB","AC","A_regret","B_regret"):
        def values(name):
            return {b:float(np.mean([r[key] for r in data[(b,name)]])) for b in ids
                    if all(r[key] is not None for r in data[(b,name)])}
        h, c = values(hard), values(control)
        paired = sorted(h.keys() & c.keys())
        result["hard"][key] = cluster_mean([h[b] for b in paired],paired,seed,samples)
        result["control"][key] = cluster_mean([c[b] for b in paired],paired,seed,samples)
        result["control_minus_hard"][key] = _paired(c,h,paired,seed,samples)
    return result


def _readout_summary(readouts, seed, samples):
    result = {}
    distributions = {}
    for kind in ("cls","projected","pixel","mean"):
        episodes = defaultdict(lambda: defaultdict(lambda: [0.,0]))
        for arm in readouts:
            data = (arm["features"]["cls"].get("mean_episode_errors",{}) if kind=="mean"
                    else arm["features"][kind]["episode_errors"])
            for ep, targets in data.items():
                for target, values in targets.items():
                    episodes[ep][target][0] += values["sum"]
                    episodes[ep][target][1] += values["n"]
        if not episodes:
            continue
        # Frame-weighted means, resample whole episodes (not frames or seeds).
        ids = sorted(episodes)
        draws = np.random.default_rng(seed).integers(len(ids),size=(samples,len(ids)))
        result[kind] = {}
        for target in ("peg","T"):
            sums = np.array([episodes[e][target][0] for e in ids])
            counts = np.array([episodes[e][target][1] for e in ids])
            bootstrap = sums[draws].sum(1)/counts[draws].sum(1)
            result[kind][target] = {"mean": float(sums.sum()/counts.sum()),
                "ci": np.quantile(bootstrap,[.025,.975]).tolist(),"episodes":len(ids)}
            distributions[(kind,target)] = bootstrap
    for kind in ("cls","projected"):
        if kind not in result or "pixel" not in result:
            continue
        for ratio,denominator in (("peg_over_T",(kind,"T")),("peg_over_pixel",("pixel","peg"))):
            den = result[denominator[0]][denominator[1]]["mean"]
            bootden = distributions[denominator]
            valid = bootden>0
            bootstrap = distributions[(kind,"peg")][valid]/bootden[valid]
            result[kind][ratio] = {"mean":result[kind]["peg"]["mean"]/den if den>0 else None,
                "ci":np.quantile(bootstrap,[.025,.975]).tolist() if len(bootstrap) else None}
    return result


def compute_report(episodes, feasibility, reference_e, abc_records=(), readouts=None,
                   seed=42, samples=2000, complete_feasibility=True):
    """Pure reporting core. Inputs are episode rows, not arm-specific summaries."""
    readouts = readouts or {}
    by_checkpoint = defaultdict(list)
    for row in episodes:
        by_checkpoint[row["arm"]].append(row)
    for name in sorted(set(readouts) | {r["checkpoint"] for r in abc_records}):
        by_checkpoint.setdefault(name,[])
    scopes = {name: {"checkpoints":[name],"rows":rows,"pooled":False}
              for name,rows in sorted(by_checkpoint.items())}
    families = defaultdict(list)
    for name in by_checkpoint:
        families[_family(name)].append(name)
    for family,names in families.items():
        if len(names) < 2:
            continue
        scopes[f"{family}:pooled"] = {"checkpoints":sorted(names),
            "rows":[r for name in names for r in by_checkpoint[name]],"pooled":True}
    fscore = _success_map(feasibility)
    universe = {r["base_id"] for r in episodes}
    if not universe:
        universe = {r["base_id"] for r in abc_records}
    reference_e = [r for r in reference_e if r["base_id"] in universe]
    result = {"seed":seed,"bootstrap_samples":samples,"contrasts":contrasts(),
              "feasibility":{},"checkpoints":{},"coverage":{}}
    for label,(hard,control) in contrasts().items():
        common = {b for b in universe if fscore.get((b,hard))==1 and fscore.get((b,control))==1}
        missing_f = {b for b in universe if (b,hard) not in fscore or (b,control) not in fscore}
        result["feasibility"][label] = {"included_base_ids":sorted(common),
            "excluded_base_ids":sorted(universe-common),"missing_base_ids":sorted(missing_f),
            "complete":bool(feasibility) and complete_feasibility and not missing_f}
    for scope, info in scopes.items():
        names, rows = info["checkpoints"], info["rows"]
        abc = [r for r in abc_records if r["checkpoint"] in names]
        readout = _readout_summary([readouts[n] for n in names if n in readouts],seed,samples)
        output = {"checkpoints":names,"pooled":info["pooled"],"readout":readout,
                  "evaluation_seeds":sorted({r.get("evaluation_seed",r["seed"]) for r in rows}),
                  "training_seeds": sorted({int(m.group(1)) for n in names if (m:=re.search(r"_s(\d+)$",n))}),
                  "contrasts":{}}
        for label,(hard,control) in contrasts().items():
            f = result["feasibility"][label]
            common = set(f["included_base_ids"])
            entry = {"hard_condition":hard,"control_condition":control}
            for mode,allowed in (("unconditional",None),("common_feasible",common)):
                learned = _contrast(rows,hard,control,allowed,seed,samples)
                reference = _contrast(reference_e,hard,control,allowed,seed,samples)
                abc_comparison = _abc_comparison(abc,hard,control,allowed,seed,samples)
                entry[mode] = {"learned":learned,"reference_E":reference,"abc":abc_comparison}
                if label=="D":
                    entry[mode]["disturbance_allowed"] = _contrast(rows,hard,control,allowed,seed,samples,"t_success")
                    entry[mode]["reference_E_disturbance_allowed"] = _contrast(reference_e,hard,control,allowed,seed,samples,"t_success")
            measured = entry["common_feasible"]
            ref_gap = measured["reference_E"]["gap_control_minus_hard"]
            complete = (f["complete"] and bool(common) and not measured["learned"]["missing_base_ids"]
                        and not measured["reference_E"]["missing_base_ids"])
            # A pooled arm must include each checkpoint on every F-selected
            # base, rather than changing the seed mixture across scenes.
            complete &= all(not _contrast(by_checkpoint[n],hard,control,common,seed,samples)["missing_base_ids"] for n in names)
            entry["gap_label"] = gap_label(measured["learned"]["gap_control_minus_hard"],
                                           ref_gap["difference"] if ref_gap else None,complete)
            entry["localization"] = {}
            for representation in ("cls","projected"):
                rd = readout.get(representation,{})
                entry["localization"][representation] = localization_labels(
                    rd.get("peg",{}).get("mean"),rd.get("T",{}).get("mean"),
                    readout.get("pixel",{}).get("peg",{}).get("mean"),
                    {k:v["mean"] for k,v in measured["abc"]["hard"].items()},
                    {k:v["mean"] for k,v in measured["abc"]["control"].items()})
                entry["localization"][representation]["gap_present"] = entry["gap_label"]=="gap present"
            output["contrasts"][label] = entry
        result["checkpoints"][scope] = output
    for block,info in scopes.items():
        if any(_family(n)!="ft_block" for n in info["checkpoints"]):
            continue
        mixed = "ft_mixed:pooled" if info["pooled"] else block.replace("ft_block","ft_mixed",1)
        if mixed not in scopes:
            result["coverage"][f"{block}->{mixed}"] = {label:{"label":"unavailable","base_ids":[],
                "hard_gain":None,"control_gain":None,"block_gap":None,"mixed_gap":None,
                "gap_shrink":None,"gap_shrink_fraction":None} for label in contrasts()}
            continue
        coverage = {}
        for label,(hard,control) in contrasts().items():
            ids = set(result["feasibility"][label]["included_base_ids"])
            bs, ms = _success_map(info["rows"]), _success_map(scopes[mixed]["rows"])
            common = sorted(b for b in ids if all((b,c) in table for table in (bs,ms) for c in (hard,control)))
            h = _paired({b:ms[(b,hard)] for b in common},{b:bs[(b,hard)] for b in common},common,seed,samples)
            c = _paired({b:ms[(b,control)] for b in common},{b:bs[(b,control)] for b in common},common,seed,samples)
            bg = _paired({b:bs[(b,control)] for b in common},{b:bs[(b,hard)] for b in common},common,seed,samples)
            mg = _paired({b:ms[(b,control)] for b in common},{b:ms[(b,hard)] for b in common},common,seed,samples)
            shrink = _paired({b:bs[(b,control)]-bs[(b,hard)] for b in common},
                             {b:ms[(b,control)]-ms[(b,hard)] for b in common},common,seed,samples)
            label_value = coverage_label(h,c["difference"] if c else None,bg["difference"] if bg else None,
                mg["difference"] if mg else None,
                result["feasibility"][label]["complete"] and set(common)==ids and bool(ids)
                and all(not _contrast(by_checkpoint[n],hard,control,ids,seed,samples)["missing_base_ids"]
                        for n in info["checkpoints"]+scopes[mixed]["checkpoints"]))
            coverage[label] = {"label":label_value,"base_ids":common,"hard_gain":h,"control_gain":c,
                "block_gap":bg,"mixed_gap":mg,"gap_shrink":shrink,
                "gap_shrink_fraction": shrink["difference"]/bg["difference"] if bg and bg["difference"]>0 else None}
            unconditional_ids = sorted({b for b,condition in bs if condition==hard}
                & {b for b,condition in bs if condition==control}
                & {b for b,condition in ms if condition==hard}
                & {b for b,condition in ms if condition==control})
            coverage[label]["unconditional"] = {
                "base_ids":unconditional_ids,
                "hard_gain":_paired({b:ms[(b,hard)] for b in unconditional_ids},
                                    {b:bs[(b,hard)] for b in unconditional_ids},unconditional_ids,seed,samples),
                "control_gain":_paired({b:ms[(b,control)] for b in unconditional_ids},
                                       {b:bs[(b,control)] for b in unconditional_ids},unconditional_ids,seed,samples),
                "block_gap":_paired({b:bs[(b,control)] for b in unconditional_ids},
                                    {b:bs[(b,hard)] for b in unconditional_ids},unconditional_ids,seed,samples),
                "mixed_gap":_paired({b:ms[(b,control)] for b in unconditional_ids},
                                    {b:ms[(b,hard)] for b in unconditional_ids},unconditional_ids,seed,samples),
                "gap_shrink":_paired({b:bs[(b,control)]-bs[(b,hard)] for b in unconditional_ids},
                    {b:ms[(b,control)]-ms[(b,hard)] for b in unconditional_ids},unconditional_ids,seed,samples)}
        result["coverage"][f"{block}->{mixed}"] = coverage
    return result


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
                     + ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def _number(value):
    if value is None:
        return "unavailable"
    mean = value.get("difference",value.get("mean"))
    if mean is None:
        return "unavailable"
    ci = value.get("ci")
    return f"{mean:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def markdown_report(result):
    tables = [_table(["Contrast","Feasible bases","Excluded","Missing F","Complete"],
        [[k,len(v["included_base_ids"]),len(v["excluded_base_ids"]),len(v["missing_base_ids"]),v["complete"]]
         for k,v in result["feasibility"].items()])]
    success, bars, readout, abc = [], [], [], []
    for scope,arm in result["checkpoints"].items():
        for label,entry in arm["contrasts"].items():
            for mode in ("common_feasible","unconditional"):
                e = entry[mode]
                success.append([scope,label,mode,len(e["learned"]["base_ids"]),_number(e["learned"]["hard"]),
                    _number(e["learned"]["control"]),_number(e["learned"]["gap_control_minus_hard"]),
                    _number(e["reference_E"]["gap_control_minus_hard"])])
                if "disturbance_allowed" in e:
                    success.append([scope,label+" T-only",mode,len(e["disturbance_allowed"]["base_ids"]),
                        _number(e["disturbance_allowed"]["hard"]),_number(e["disturbance_allowed"]["control"]),
                        _number(e["disturbance_allowed"]["gap_control_minus_hard"]),
                        _number(e["reference_E_disturbance_allowed"]["gap_control_minus_hard"])])
                for metric in ("BC","AB","AC","A_regret","B_regret"):
                    v = e["abc"]
                    abc.append([scope,label,mode,metric,_number(v["hard"][metric]),_number(v["control"][metric]),
                                _number(v["control_minus_hard"][metric])])
            for representation,v in entry["localization"].items():
                bars.append([scope,label,entry["gap_label"],representation,", ".join(v["labels"]) or "none",
                             ", ".join(v["unavailable_clauses"]) or "none"])
        for kind,targets in arm["readout"].items():
            readout.append([scope,kind,_number(targets["peg"]),_number(targets["T"]),
                            _number(targets.get("peg_over_T")),_number(targets.get("peg_over_pixel"))])
    tables.extend([_table(["Checkpoint","Contrast","Set","Bases","Hard success","Control success","Gap control-hard (95% CI)","Reference E gap"],success),
                   _table(["Checkpoint","Contrast","Gap label","Readout","Localization labels","Unavailable clauses"],bars),
                   _table(["Checkpoint","Readout","Peg error px (95% CI)","T error px (95% CI)","Peg/T ratio","Peg/pixel ratio"],readout),
                   _table(["Checkpoint","Contrast","Set","ABC metric","Hard (95% CI)","Control (95% CI)","Control-hard (95% CI)"],abc)])
    tables.append(_table(["Comparison","Contrast","Coverage label","Hard gain","Control gain","Gap shrink","Shrink fraction"],
        [[pair,label,v["label"],_number(v["hard_gain"]),_number(v["control_gain"]),_number(v["gap_shrink"]),
          f'{v["gap_shrink_fraction"]:.3f}' if v["gap_shrink_fraction"] is not None else "unavailable"]
         for pair,entries in result["coverage"].items() for label,v in entries.items()]))
    note = ("Amendment 1: N=400 candidate bases. The 10-point cross-arm coverage "
            "comparison is underpowered; frozen decision thresholds remain unchanged.\n\n"
            if result.get("coverage_underpowered") else "")
    return note + "\n\n".join(tables)+"\n"


def run_report(runs, run_dir=None, feasibility_run=None, seed=42, samples=2000, publish=False,
               coverage_underpowered=False):
    roots = [Path(p).expanduser().resolve() for p in runs]
    episodes, abc, readouts, configs, sources = [], [], {}, [], {}
    analysis_configs, readout_pool_ids, abc_diagnostics = [], set(), []
    fpaths = set()
    if feasibility_run:
        fpaths.add(Path(feasibility_run).expanduser().resolve())
    for root in roots:
        config = json.loads((root/"config.json").read_text())
        sources[str(root/"config.json")] = fingerprint(root/"config.json")
        if config.get("feasibility_run"):
            fpaths.add(Path(config["feasibility_run"]).expanduser().resolve())
        if (root/"readout.json").exists():
            data = json.loads((root/"readout.json").read_text())
            sources[str(root/"readout.json")] = fingerprint(root/"readout.json")
            analysis_configs.append(config)
            if not data.get("pool_identity"):
                raise ValueError("readout needs a persisted common pool identity")
            readout_pool_ids.add(data["pool_identity"])
            for name,value in data["checkpoints"].items():
                if name in readouts:
                    raise ValueError(f"duplicate readout checkpoint: {name}")
                readouts[name] = value
        elif (root/"abc.json").exists():
            data = json.loads((root/"abc.json").read_text())
            abc.extend(data["records"])
            analysis_configs.append(config)
            abc_diagnostics.append({"run_dir":str(root),"included_banks":data["included_banks"],
                "excluded_banks":data["excluded_banks"],"banks":data["banks"],
                "feasibility_selected":data["feasibility_selected"]})
            sources[str(root/"abc.json")] = fingerprint(root/"abc.json")
        elif (root/"episodes.jsonl").exists():
            rows = [json.loads(line) for line in (root/"episodes.jsonl").read_text().splitlines()]
            configs.append((root,config,rows))
            sources[str(root/"episodes.jsonl")] = fingerprint(root/"episodes.jsonl")
        else:
            raise ValueError(f"no completed analysis or episodes in {root}")
    if len(fpaths)>1:
        raise ValueError("one common reference feasibility run required")
    if len(readout_pool_ids)>1:
        raise ValueError("all readouts must use the same evaluation pool")
    feasibility, fc = [], None
    if fpaths:
        froot = next(iter(fpaths))
        fc = json.loads((froot/"config.json").read_text())
        if fc["arm"] != "reference":
            raise ValueError("feasibility must be a reference run")
        feasibility = [json.loads(line) for line in (froot/"episodes.jsonl").read_text().splitlines()]
        sources[str(froot/"episodes.jsonl")] = fingerprint(froot/"episodes.jsonl")
        sources[str(froot/"config.json")] = fingerprint(froot/"config.json")
    reference_e = []
    protocol = None
    seen = set()
    for root,config,rows in configs:
        if root in fpaths:
            continue
        identity = tuple(config.get(k) for k in ("bases_sha256","budget","normalization_sha256"))
        if protocol is not None and identity!=protocol:
            raise ValueError("evaluation runs disagree on bases, budget or normalization")
        protocol = identity
        if fc:
            if any(fc.get(k)!=config.get(k) for k in ("bases_sha256","budget","displacement_range","normalization_sha256")):
                raise ValueError("F and evaluation protocols differ")
            if config["seed"]==fc["seed"]:
                raise ValueError("reference F must be independent of evaluation seeds")
            if config["arm"]=="reference" and any(fc.get(k)!=config.get(k) for k in ("population","iterations","topk","approach_weight")):
                raise ValueError("reference E planner must match F")
        for row in rows:
            key = (row["arm"],config["seed"],row["base_id"],row["condition"])
            if key in seen:
                raise ValueError("duplicate evaluation episode")
            seen.add(key)
            row["evaluation_seed"] = config["seed"]
        (reference_e if config["arm"]=="reference" else episodes).extend(rows)
    if protocol is not None and any(c["bases_sha256"]!=protocol[0] for c in analysis_configs):
        raise ValueError("analysis and evaluation runs use different bases")
    abc_seen = set()
    shared_banks = {}
    for row in abc:
        key = (row["checkpoint"],row["base_id"],row["condition"],row["bank_sha256"])
        if key in abc_seen:
            raise ValueError("duplicate ABC checkpoint/bank result")
        abc_seen.add(key)
        bank_key = (row["base_id"],row["condition"])
        if bank_key in shared_banks and shared_banks[bank_key]!=row["bank_sha256"]:
            raise ValueError("checkpoints must share identical physical candidate banks")
        shared_banks[bank_key] = row["bank_sha256"]
    for _,ec,_ in configs:
        arm = ec["arm"]
        for ac in analysis_configs:
            if arm in ac.get("checkpoint_sha256",{}) and ec.get("checkpoint_weights_sha256")!=ac["checkpoint_sha256"][arm]:
                raise ValueError("analysis and evaluation checkpoint weights differ")
        if arm in readouts and ec.get("with_target")!=readouts[arm]["with_target"]:
            raise ValueError("readout and evaluation rendering differ")
        if any(row["with_target"]!=ec.get("with_target") for row in abc if row["checkpoint"]==arm):
            raise ValueError("ABC and evaluation rendering differ")
    config = {"stage":"probe-report","runs":list(map(str,roots)),
              "feasibility_run":str(next(iter(fpaths))) if fpaths else None,"seed":seed,"bootstrap_samples":samples}
    if coverage_underpowered:
        config["coverage_underpowered"] = True
    root = open_run("probe-report",config,run_dir)
    result = compute_report(episodes,feasibility,reference_e,abc,readouts,seed,samples)
    result["coverage_underpowered"] = coverage_underpowered
    # Reports may be rerun after upstream resumes append results. Keep the
    # requested sources/settings fixed, but refresh their content provenance.
    result.update(stage="probe-report",run_dir=str(root),sources={**config,"sources_sha256":sources},
                  abc_diagnostics=abc_diagnostics,
                  readout_pool_identity=next(iter(readout_pool_ids)) if readout_pool_ids else None,
                  reference_E_seeds=sorted({r["evaluation_seed"] for r in reference_e}),
                  feasibility_seed=fc["seed"] if fc else None)
    write_json(root/"results.json",result)
    (root/"RESULTS.md").write_text(markdown_report(result))
    if publish:
        target = Path(__file__).resolve().parents[3]/"experiments/role_swap/results"
        target.mkdir(parents=True,exist_ok=True)
        for filename in ("RESULTS.md","results.json"):
            shutil.copyfile(root/filename,target/filename)
    return result
