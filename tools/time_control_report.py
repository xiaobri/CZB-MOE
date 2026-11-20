"""Aggregate shard confusion matrices and typeset the pre-specified paired contrast."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from time_control_lock import ARMS, SEEDS, ROOT, read_json, sha

SITES = {"xiehe": "Xiehe (validation)", "qilu": "Qilu", "tiantan": "Tiantan",
         "chifeng": "Chifeng", "xian": "Xi'an"}
FIELDS = ("tp", "fp", "fn", "tn")


def collect(parts, lockfile):
    lock = read_json(lockfile)
    digest = sha(lockfile)
    original = read_json(ROOT / "results/external_protocol.json")["hospitals"]
    source = read_json(ROOT / "results/protocol.json")["manifest"]
    keys = [f"{arm}_{seed}" for arm in ARMS for seed in SEEDS]
    rows = []
    audits = {}
    for site in SITES:
        shards = sorted((read_json(p) for p in parts.glob(f"{site}_*.json")), key=lambda s: s["contract"]["shard"])
        if not shards:
            raise ValueError(f"Missing hospital {site}")
        n = source["mode_counts"]["Val"] if site == "xiehe" else original[site]["windows"]
        positive = source["label_counts"]["Val/target1"] if site == "xiehe" else original[site]["positives"]
        manifest = source["csv_sha256"] if site == "xiehe" else original[site]["manifest_sha256"]
        offset = 0
        sums = {k: dict.fromkeys(FIELDS, 0) for k in keys}
        for i, shard in enumerate(shards):
            c = shard["contract"]
            if (c["shard"], c["shards"], c["site"], c["start"], c["size"], c["manifest"], c["lock"], c["script"]) != (
                    i, len(shards), site, offset, n, manifest, digest, sha(ROOT / "tools/time_control_eval.py")):
                raise ValueError("Incomplete or mismatched audit")
            if c["stop"] <= offset:
                raise ValueError("Empty shard")
            for k in keys:
                counts = shard["counts"][k]
                if sum(counts.values()) != c["stop"] - offset or min(counts.values()) < 0:
                    raise ValueError("Incorrect number of decisions")
                for name in FIELDS:
                    sums[k][name] += counts[name]
            offset = c["stop"]
        if offset != n:
            raise ValueError("Incomplete dataset")
        audits[site] = dict(windows=n, positives=positive, shards=len(shards), manifest_sha256=manifest,
                            first_batch_max_weight_range={
                                k: max(s["weight_spread"][k] for s in shards) for k in keys})
        if site == "xiehe":
            audits[site]["raw_vs_cache_changed_decisions"] = {
                k: sum(s["changed_cached_decisions"][k] for s in shards) for k in keys}
            audits[site]["raw_vs_cache_max_probability_error"] = {
                k: max(s["max_cached_score_error"][k] for s in shards) for k in keys}
        for key, counts in sums.items():
            arm, seed = key.split("_")
            ck = lock["checkpoints"][key]
            tp, fp, fn, tn = (counts[k] for k in FIELDS)
            if tp + fn != positive:
                raise ValueError("Wrong class distribution")
            rows.append(dict(site=site, arm=arm, seed=seed, epoch=ck["epoch"], threshold=ck["threshold"],
                             **counts, f1=2*tp/(2*tp+fp+fn)))
    return rows, dict(**lock, sites=audits, evaluator_sha256=sha(ROOT / "tools/time_control_eval.py"))


def table(rows):
    values = {}
    for r in rows:
        tp, fp, fn, tn = (int(r[k]) for k in FIELDS)
        f1 = 2*tp/(2*tp+fp+fn)
        key = (r["site"], r["arm"], int(r["seed"]))
        if key in values or not math.isclose(f1, float(r["f1"])):
            raise ValueError("Duplicate record or inconsistent F1")
        values[key] = 100*f1
    expected = {(site, arm, seed) for site in SITES for arm in ARMS for seed in SEEDS}
    if set(values) != expected:
        raise ValueError("Incomplete paired results")
    for arm in ARMS:
        for seed in SEEDS:
            values["external", arm, seed] = mean(values[site, arm, seed] for site in SITES if site != "xiehe")
    lines = []
    for site, name in dict(**SITES, external="Equal-site external mean").items():
        fixed = [values[site, "fixed", seed] for seed in SEEDS]
        dynamic = [values[site, "dynamic", seed] for seed in SEEDS]
        delta = [d-f for d,f in zip(dynamic, fixed)]
        cells = [name, f"{fixed[0]:.2f}", f"{dynamic[0]:.2f}", f"{delta[0]:+.2f}",
                 f"{fixed[1]:.2f}", f"{dynamic[1]:.2f}", f"{delta[1]:+.2f}",
                 f"{mean(fixed):.2f}", f"{mean(dynamic):.2f}", f"{mean(delta):+.2f}"]
        if site == "external":
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r"\\")
    return r"\newcommand{\TemporalControlRows}{%" + "\n" + "\n".join(lines) + "\n}\n"


def verify_public(rows, proof):
    digest = sha(ROOT / "tools/time_control_eval.py")
    if digest != proof["evaluator_sha256"] or not proof["matched_to_original_training_stream"]:
        raise ValueError("Paired evaluation provenance differs")
    for seed in SEEDS:
        locked = [proof["checkpoints"][f"{a}_{seed}"] for a in ARMS]
        if locked[0]["initialization"] != locked[1]["initialization"]:
            raise ValueError("Different trainable initialization within a seed")
    for row in rows:
        site, arm = row["site"], row["arm"]
        key = f"{arm}_{row['seed']}"
        counts = {k: int(row[k]) for k in FIELDS}
        audit = proof["sites"][site]
        ck = proof["checkpoints"][key]
        if ((int(row["epoch"]), float(row["threshold"])) != (ck["epoch"], ck["threshold"])
                or sum(counts.values()) != audit["windows"] or min(counts.values()) < 0
                or counts["tp"] + counts["fn"] != audit["positives"]):
            raise ValueError("Incorrect cohort or operating point")
        movement = audit["first_batch_max_weight_range"][key]
        if (arm == "fixed" and movement > 1e-6) or (arm == "dynamic" and movement <= 0):
            raise ValueError("Observed temporal weights differ from the intervention")
        if site == "xiehe" and audit["raw_vs_cache_changed_decisions"][key] == 0:
            if counts != ck["cached_counts"]:
                raise ValueError("Raw operating point differs from cached counts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    csvfile = ROOT / "results/time_control_sites.csv"
    tex = ROOT / "paper/time_control_rows.tex"
    if args.parts and args.lock:
        rows, proof = collect(args.parts, args.lock)
        with csvfile.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        (ROOT / "results/time_control_protocol.json").write_text(json.dumps(proof, indent=2) + "\n")
        tex.write_text(table(rows))
    elif args.parts or args.lock:
        parser.error("Supply both --parts and --lock")
    with csvfile.open() as f:
        rows = list(csv.DictReader(f))
    verify_public(rows, read_json(ROOT / "results/time_control_protocol.json"))
    if table(rows) != tex.read_text():
        raise ValueError("Published table and counts disagree")
    print(tex.read_text())


if __name__ == "__main__":
    main()
