"""Verify raw component contrasts and generate the appendix tables from public counts."""
import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

from component_records import CONTROLS, COUNTS, HEADS, PAIR_CONTROLS, STAGES, write_rows
from time_control_lock import ROOT, SEEDS, read_json, sha

ALL = (*STAGES, *PAIR_CONTROLS, *CONTROLS)
LABELS = {
    "baseline":"Strict six-expert baseline", "risk_power":"Regional statistics head, weighted focal and ranking",
    "peaks":"Add peak-shape correction", "independent":"Add temporal pooling and global adjustment",
    "independent_aux":"Add regional auxiliary focal supervision", "collection":"Add within-group historical pair correction",
    "fresh_both":"Refresh positive and stored negative scores", "plain":"Unit weights, focal only",
    "rank":"Unit weights, focal and ranking", "weight":"Adaptive group weights, focal only",
    "full":"Adaptive group weights, focal and ranking", "static_rank":"Static group weights, focal and ranking",
    "scores_only":"Expert and mixture scores only", "no_moments":"Scores and router features",
    "mean_only":"Scores, router features and activation means", "pooled_quantiles":"Pooled groups, risk quantiles",
    "within_quantiles":"Same group, risk quantiles", "core_path":"Six-expert core",
    "simultaneous_risk_power":"Regional statistics correction", "simultaneous_peak_anchor":"Peak-shape correction",
    "simultaneous_allocation_anchor":"Temporal and global adjustments", "output":"Final pair correction"
}


def raw_results(source):
    lock = read_json(source/"lock.json")
    digest = sha(source/"lock.json")
    evaluator = sha(ROOT/"tools/component_replay.py")
    manifest = read_json(ROOT/"results/protocol.json")["manifest"]["csv_sha256"]
    rows = []
    for seed in SEEDS:
        parts = sorted((read_json(p) for p in (source/"raw").glob(f"seed{seed}_shard*.json")),
                       key=lambda x:x["contract"]["shard"])
        if not parts:
            raise ValueError("Raw replay missing")
        offset=0
        for i,part in enumerate(parts):
            c=part["contract"]
            if (c["seed"],c["start"],c["shard"],c["shards"],c["lock"],c["manifest"],c["script"]) != (
                    seed,offset,i,len(parts),digest,manifest,evaluator) or c["stop"]<=offset:
                raise ValueError("Mismatched replay partition")
            if set(part["counts"])!=set(ALL):
                raise ValueError("Missing model output")
            for arm in ALL:
                if sum(part["counts"][arm].values())!=c["stop"]-offset:
                    raise ValueError("Wrong number of decisions")
            offset=c["stop"]
        if offset!=166079:
            raise ValueError("Incomplete raw replay")
        for arm in ALL:
            ck=lock[f"{arm}_{seed}"]
            counts={k:sum(p["counts"][arm][k] for p in parts) for k in COUNTS}
            tp,fp,fn,tn=(counts[k] for k in COUNTS)
            changed=sum(p["changed"][arm] for p in parts)
            rows.append(dict(arm=arm,seed=seed,epoch=ck["epoch"],threshold=ck["threshold"],**counts,
                             f1=2*tp/(2*tp+fp+fn),changed_cached_decisions=changed))
    return rows,dict(source_manifest_sha256=manifest,evaluator_sha256=evaluator,
                    verified_draws_per_fit=400000,verified_steps_per_fit=3140,checkpoints=lock)


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def f1(row):
    tp,fp,fn,tn=(int(row[k]) for k in COUNTS)
    v=2*tp/(2*tp+fp+fn)
    if min(tp,fp,fn,tn)<0 or tp+fn!=3471 or fp+tn!=162608 or not math.isclose(v,float(row["f1"])):
        raise ValueError("F1 or validation population differs")
    return 100*v


def checked_inputs(results):
    history=defaultdict(list)
    for row in read_rows(results/"component_epochs.csv"):
        f1(row)
        history[row["arm"],int(row["seed"])].append(row)
    if set(history)!={(a,s) for a in (*ALL,*HEADS) for s in SEEDS}:
        raise ValueError("Incomplete component histories")
    best={}
    for k,rows in history.items():
        if [int(r["epoch"]) for r in rows]!=list(range(1,21)):
            raise ValueError("Incomplete selection window")
        best[k]=max(rows,key=lambda r:(float(r["f1"]),int(r["epoch"])))
    proof=read_json(results/"component_audit.json")
    if (proof["evaluator_sha256"]!=sha(ROOT/"tools/component_replay.py") or
            proof["source_manifest_sha256"]!=read_json(results/"protocol.json")["manifest"]["csv_sha256"]):
        raise ValueError("Raw replay provenance mismatch")
    scores={}
    for row in read_rows(results/"component_xiehe.csv"):
        arm,seed=row["arm"],int(row["seed"])
        ck=proof["checkpoints"][f"{arm}_{seed}"]
        selected=best[arm,seed]
        if ((arm,seed) in scores or row["epoch"]!=selected["epoch"] or row["threshold"]!=selected["threshold"]
                or int(row["epoch"])!=ck["epoch"] or float(row["threshold"])!=ck["threshold"]
                or any(int(selected[k])!=ck["cached_counts"][k] for k in COUNTS)):
            raise ValueError("Locked operating point mismatch")
        changes=int(row["changed_cached_decisions"])
        net=sum(abs(int(row[k])-int(selected[k])) for k in COUNTS)//2
        if net>changes or (changes-net)%2:
            raise ValueError("Raw counts inconsistent with changed decisions")
        scores[arm,seed]=f1(row)
    if set(scores)!={(a,s) for a in ALL for s in SEEDS}:
        raise ValueError("Raw scores missing")
    for seed in SEEDS:
        ck=proof["checkpoints"]
        if len({ck[f"{a}_{seed}"]["shared_core_initialization"] for a in ALL})!=1:
            raise ValueError("Shared core initialization mismatch")
        if len({(ck[f"{a}_{seed}"]["all_trainable_initialization"],ck[f"{a}_{seed}"]["parameters"])
                for a in CONTROLS})!=1:
            raise ValueError("First-head ablations differ at initialization")
    return scores,history,best


def render(results):
    scores,history,best=checked_inputs(results)
    blocks=defaultdict(list)
    effects=[]
    def score(arm):
        return [scores[arm,s] for s in SEEDS]
    def line(label,values):
        return " & ".join([label,*values])+r"\\"
    def effect(name,values):
        effects.append(dict(contrast=name,seed3=values[0],seed42=values[1],mean=mean(values)))
    def delta(a,b):
        return [scores[a,s]-scores[b,s] for s in SEEDS]
    def formatted(values, signed=False):
        return [format(x,"+.3f" if signed else ".3f") for x in [*values,mean(values)]]
    previous=None
    for a in STAGES:
        differences=["--"]*3
        if previous is not None:
            v=delta(a,previous)
            differences=formatted(v,True)
            effect(f"{a} minus {previous}",v)
        blocks["ComponentStages"].append(line(LABELS[a],formatted(score(a))+differences))
        previous=a
    for a in ("plain","rank","weight","full","static_rank","scores_only","no_moments","mean_only"):
        blocks["ComponentFits"].append(line(LABELS[a],formatted(score(a))))
    for name,a,b in (
        ("Weighting without ranking","weight","plain"),("Weighting with ranking","full","rank"),
        ("Ranking without weighting","rank","plain"),("Ranking with weighting","full","weight"),
        ("Adaptive vs. static weighting, with ranking","full","static_rank"),
        ("Adding router features to scores","no_moments","scores_only"),
        ("Adding means to router features and scores","mean_only","no_moments"),
        ("Adding maxima and SD to the means","full","mean_only"),
        ("Within-group vs. pooled risk quantiles","within_quantiles","pooled_quantiles")):
        v=delta(a,b)
        effect(name,v)
        blocks["ComponentEffects"].append(line(name,formatted(v,True)))
    interaction=[x-y for x,y in zip(delta("full","rank"),delta("weight","plain"))]
    effect("Weighting by ranking interaction",interaction)
    blocks["ComponentEffects"].append(line("Weighting by ranking interaction",formatted(interaction,True)))
    effect("Combined weighting and ranking",delta("full","plain"))
    for label,a,b,c,d in (("Weighting averaged over ranking","weight","plain","full","rank"),
                          ("Ranking averaged over weighting","rank","plain","full","weight")):
        effect(label,[(x+y)/2 for x,y in zip(delta(a,b),delta(c,d))])
    for a in (*HEADS,"fresh_both"):
        chosen=[f1(best[a,s]) for s in SEEDS]
        fixed=[f1(history[a,s][int(best["fresh_both",s]["epoch"])-1]) for s in SEEDS]
        blocks["ComponentHeads"].append(line(LABELS["output" if a=="fresh_both" else a],
                                                  formatted(chosen)+formatted(fixed)))
    macros = []
    for name, lines in blocks.items():
        command = chr(92) + "newcommand{" + chr(92) + name + "}{%"
        macros.append("\n".join([command, *lines, "}"]) + "\n")
    tex = "".join(macros)
    return tex,effects


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private",type=Path)
    args=p.parse_args()
    results=ROOT/"results"
    tex=ROOT/"paper/component_rows.tex"
    if args.private:
        rows,audit=raw_results(args.private)
        shutil.copyfile(args.private/"epochs.csv",results/"component_epochs.csv")
        write_rows(results/"component_xiehe.csv",rows)
        (results/"component_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    generated,effects=render(results)
    if args.private:
        tex.write_text(generated)
        write_rows(results/"component_effects.csv",effects)
    if tex.read_text()!=generated or read_rows(results/"component_effects.csv")!=[
            {k:str(v) for k,v in r.items()} for r in effects]:
        raise ValueError("Published contrasts differ from counts")
    print(generated)


if __name__=="__main__":
    main()
