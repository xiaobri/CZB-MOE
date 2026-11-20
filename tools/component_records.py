"""Check the training records and reconstruct source selections for component attribution."""
import argparse
import csv
import json
import math
from pathlib import Path

from time_control_lock import ROOT, SEEDS, read_json, sha

STAGES = {
    "baseline": ("strict_baseline_20261115", "baseline"),
    "risk_power": ("fair_complement_20261118", "risk_power"),
    "peaks": ("fair_peakshape_20261119", "peaks"),
    "independent": ("fair_expert_allocation_20261120", "independent"),
    "independent_aux": ("fair_expert_allocation_20261120", "independent_aux"),
    "collection": ("fair_refreshed_pairs_20261120", "collection"),
    "fresh_both": ("fair_refreshed_pairs_20261120", "fresh_both"),
}
PAIR_CONTROLS = {arm: ("fair_risk_pairs_20261120", arm) for arm in ("pooled_quantiles", "within_quantiles")}
CONTROLS = ("plain", "rank", "weight", "full", "static_rank", "scores_only", "no_moments", "mean_only")
HEADS = ("core_path", "simultaneous_risk_power", "simultaneous_peak_anchor", "simultaneous_allocation_anchor")
COUNTS = ("tp", "fp", "fn", "tn")


def location(original, controls, arm, seed):
    if arm in CONTROLS:
        return controls / arm / f"seed{seed}"
    date, method = {**STAGES, **PAIR_CONTROLS}[arm]
    return original / f"I2MOE_test333_{date}" / "formal" / method / f"seed{seed}"


def write_rows(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def epoch_row(arm, seed, epoch, validation):
    m = validation["best_f1_on_validation_grid"]
    tp, fp, fn, tn = (m[k] for k in COUNTS)
    f1 = 2*tp/(2*tp+fp+fn)
    if ((tp + fn, tn + fp) != (3471, 162608) or not math.isclose(f1, m["f1"])
            or not 1 <= round(100*m["threshold"]) <= 99):
        raise ValueError(f"Source metric differs for {arm}")
    return dict(arm=arm, seed=seed, epoch=epoch, threshold=m["threshold"],
                **{k: m[k] for k in COUNTS}, f1=f1)


def collect(original, controls, output):
    with (ROOT / "results/sampling.csv").open() as f:
        sampler = {(int(r["seed"]), int(r["epoch"])): r for r in csv.DictReader(f) if r["method"] == "baseline"}
    protocol = read_json(ROOT / "results/protocol.json")
    rows, locks = [], {}
    for seed in SEEDS:
        shared = set()
        paired = set()
        for arm in (*STAGES, *PAIR_CONTROLS, *CONTROLS):
            folder = location(original, controls, arm, seed)
            if not (folder / "completed.json").is_file():
                raise ValueError(f"Incomplete {arm}, seed {seed}")
            check = read_json(folder / "configuration_check.json")
            args = read_json(folder / "args.json")
            if not check["passed"] or not check["formal"] or check["differences"]:
                raise ValueError(f"Configuration mismatch in {arm}")
            for name, expected in protocol["shared_arguments"].items():
                if name != "memory_cap_gib" and args[name] != expected:
                    raise ValueError(f"Protocol differs at {arm}/{name}")
            if read_json(folder / "manifest_summary.json")["csv_sha256"] != protocol["manifest"]["csv_sha256"]:
                raise ValueError("Different source manifest")
            if arm in CONTROLS:
                init = read_json(folder / "paired_initialization.json")
                if init["parameters"] != 1775702 or check["arm"] != arm:
                    raise ValueError("Control arm does not match its record")
                paired.add(init["all_trainable"])
                core, count = init["core"], init["parameters"]
            else:
                init = read_json(folder / "initialization.json")
                core, count = init["original_trainable_state_sha256"], init["parameters_trainable"]
            shared.add(core)
            records = [json.loads(l) for l in (folder / "metrics.jsonl").read_text().splitlines()]
            if [r["epoch"] for r in records] != list(range(1, 21)):
                raise ValueError("Missing or duplicated training epoch")
            own = []
            for record in records:
                ep = record["epoch"]
                sample = read_json(folder / f"sampling_epoch{ep:02d}.json")
                ref = sampler[seed,ep]
                if (any(str(sample[k]) != ref[k] for k in ref if k not in ("seed", "epoch", "method"))
                        or record["train"]["samples"] != 20000 or record["train"]["steps"] != 157):
                    raise ValueError(f"Sampling mismatch at {arm} / {seed} / {ep}")
                own.append(epoch_row(arm, seed, ep, record["validation"]))
                if arm == "fresh_both":
                    for head in HEADS:
                        rows.append(epoch_row(head, seed, ep, record["validation"][head]))
            rows.extend(own)
            best = max(own, key=lambda r: (r["f1"], r["epoch"]))
            locks[f"{arm}_{seed}"] = dict(epoch=best["epoch"], threshold=best["threshold"], cached_f1=best["f1"],
                    cached_counts={k: best[k] for k in COUNTS}, best_checkpoint_sha256=sha(folder/"best_f1.pt"),
                    shared_core_initialization=core, parameters=count, allocator_cap_gib=args["memory_cap_gib"])
            if arm in CONTROLS:
                locks[f"{arm}_{seed}"].update(all_trainable_initialization=init["all_trainable"])
        if len(shared) != 1 or len(paired) != 1:
            raise ValueError(f"Initial parameters differ within seed {seed}")
    output.mkdir(exist_ok=True, parents=True)
    write_rows(output/"epochs.csv", rows)
    (output/"lock.json").write_text(json.dumps(locks, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "controls", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    collect(args.original,args.controls,args.output)
