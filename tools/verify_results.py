"""Verify released metrics, sampler pairing and source digests without EEG or PyTorch."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from benchmark_table import render as render_benchmark
from time_control_report import table as temporal_table, verify_public as verify_temporal_control

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METHODS = ("baseline", "collection", "fresh_cases", "fresh_controls", "fresh_both")
SEEDS = (3, 42)


def csv_rows(name):
    with (RESULTS / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_operating_points(selected):
    records = json.loads((RESULTS / "selected_operating_points.json").read_text())
    check([r["seed"] for r in records] == list(SEEDS), "Missing grid comparison")
    grid = [0.01 + i * 0.01 for i in range(99)]
    for record in records:
        seed = record["seed"]
        check(record["thresholds"] == grid, "Wrong threshold grid")
        overlaps = record["decision_overlap"]
        for label, count in (("positive", 3471), ("negative", 162608)):
            check(sum(overlaps[label].values()) == count, "Decision overlap count differs")
        for method, only in (("baseline", "baseline_only"), ("fresh_both", "full_only")):
            row = selected[method, seed]
            index = grid.index(float(row["threshold"]))
            for field, label in (("tp", "positive"), ("fp", "negative")):
                counts = record["models"][method][field]
                check(len(counts) == len(grid) and counts == sorted(counts, reverse=True),
                      "Counts should decrease as the threshold rises")
                expected = overlaps[label]["both_positive"] + overlaps[label][only]
                check(counts[index] == int(row[field]) == expected, "Decision counts disagree")
        limit = int(selected["baseline", seed]["fp"])
        full = record["models"]["fresh_both"]
        feasible = next(i for i, fp in enumerate(full["fp"]) if fp <= limit)
        check(limit == record["baseline_fp_budget"] and feasible == record["lowest_feasible_full_grid_index"],
              "Inconsistent false-positive budget")
        print(f"Seed {seed} at FP budget {limit}: threshold {grid[feasible]:.2f}, "
              f"TP {full['tp'][feasible]}, FP {full['fp'][feasible]}")


def verify_recording_budget():
    runs = json.loads((RESULTS / "recording_budget.json").read_text())
    ref = json.loads((RESULTS / "fresh_both_recording_selection.json").read_text())["seeds"]
    check(len(runs) == 10, "Expected two sets of five held folds")
    for seed in SEEDS:
        folds = sorted((r for r in runs if r["seed"] == seed), key=lambda r:r["fold"])
        check([r["fold"] for r in folds] == list(range(1,6)), "Incomplete held folds")
        check(sum(r["held_recordings"] for r in folds) == 72, "Wrong recording coverage")
        for r in folds:
            n = r["selection_fp_budget"]
            check(r["selection_full_fp"] <= n and
                  (r["selection_previous_full_fp"] is None or r["selection_previous_full_fp"] > n),
                  "Candidate did not use the first feasible grid point")
            b, f = r["held_baseline"], r["held_full"]
            check(b["tp"] + b["fn"] == f["tp"] + f["fn"] and b["fp"] + b["tn"] == f["fp"] + f["tn"],
                  "Fold denominators differ")
            prev = ref[str(seed)]
            i = r["fold"] - 1
            check([b[k] for k in ("tp","fp","fn")] == prev["baseline"]["folds"][i]["held_counts"] and
                  r["full_epoch"] == prev["candidate"]["folds"][i]["selected_epoch"], "Fold selection differs")


def verify_external(selected):
    sites = ("qilu", "tiantan", "chifeng", "xian")
    external = csv_rows("external_sites.csv")
    protocol = json.loads((RESULTS / "external_protocol.json").read_text())
    metrics = {}
    expected = {(site, method, seed) for site in sites for method in ("baseline", "fresh_both") for seed in SEEDS}
    for r in external:
        key = r["site"], r["method"], int(r["seed"])
        check(key not in metrics, "Duplicate external result")
        ref = selected[key[1:]]
        check(int(r["epoch"]) == int(ref["epoch"]) and float(r["threshold"]) == float(ref["threshold"]),
              "External selection differs from the source")
        tp, fp, fn, tn = (int(r[c]) for c in ("tp", "fp", "fn", "tn"))
        n = tp + fp + fn + tn
        site = protocol["hospitals"][key[0]]
        check(n == int(r["windows"]) == site["windows"] and tp+fn == int(r["positives"]) == site["positives"],
              "External coverage mismatch")
        check(site["direct_check_windows"] > 0 and site["maximum_direct_output_difference"] == 0,
              "Native forward comparison missing")
        for name, value in (("f1",2*tp/(2*tp+fp+fn)), ("precision",tp/max(1,tp+fp)), ("recall",tp/(tp+fn)),
                            ("specificity",tn/(tn+fp)), ("accuracy",(tp+tn)/n)):
            check(math.isclose(float(r[name]), value, abs_tol=1e-12), f"External {name} mismatch")
        metrics[key] = r
    check(set(metrics) == expected, "Missing hospital, method or seed")

    activity = defaultdict(dict)
    for r in csv_rows("expert_activity.csv"):
        key = r["site"], r["method"], int(r["seed"]), int(r["label"])
        check(r["expert"] not in activity[key], "Duplicate expert activity")
        activity[key][r["expert"]] = r
    all_sites = (*sites, "xiehe")
    check(set(activity) == {(site,m,s,y) for site in all_sites for m in ("baseline","fresh_both")
                            for s in SEEDS for y in (0,1)}, "Incomplete activity groups")
    for (site, method, seed, y), group in activity.items():
        check(set(group) == {"frontal","left_temporal","right_temporal","centro_parietal","occipital","redundancy"},
              "Incomplete expert panel")
        val = selected[method,seed] if site == "xiehe" else metrics[site,method,seed]
        n = int(val["tp"]) + int(val["fn"]) if y else int(val["tn"]) + int(val["fp"])
        check(all(int(v["windows"]) == n for v in group.values()), "Wrong activity denominator")
        check(sum(int(v["top_weight_windows"]) for v in group.values()) == n, "Highest-weight counts differ")
        check(abs(sum(float(v["mean_weight"]) for v in group.values()) - 1) < 1e-5, "Invalid routing mass")
        for v in group.values():
            check(0 <= float(v["weight_q25"]) <= float(v["weight_median"]) <= float(v["weight_q75"]) <= 1,
                  "Invalid routing quantiles")
    print("Verified four external hospitals, 16 operating points and 240 class-conditional expert summaries.")


def main():
    expected = {(method, seed) for method in METHODS for seed in SEEDS}
    histories = defaultdict(list)
    for row in csv_rows("epochs.csv"):
        tp, fp, fn, tn = (int(row[key]) for key in ("tp", "fp", "fn", "tn"))
        f1 = 2 * tp / (2 * tp + fp + fn)
        check(math.isclose(f1, float(row["f1"]), abs_tol=1e-12), "F1 mismatch")
        check(tp + fn == 3471 and tn + fp == 162608, "Validation totals differ")
        histories[row["method"], int(row["seed"])].append(row)
    check(set(histories) == expected, "Incomplete experiment")
    selected = {}
    for row in csv_rows("selected.csv"):
        key = row["method"], int(row["seed"])
        check(key not in selected, "Duplicate selected run")
        epochs = histories[key]
        check([int(v["epoch"]) for v in epochs] == list(range(1, 21)), "Missing epochs")
        check(max(epochs, key=lambda x: float(x["f1"])) == row, "Checkpoint selection mismatch")
        selected[key] = row
    check(set(selected) == expected, "Missing selected results")

    orders = {}
    for row in csv_rows("sampling.csv"):
        key = row["method"], int(row["seed"]), int(row["epoch"])
        check(key not in orders, "Duplicate sampler record")
        check([int(row[k]) for k in ("class_0", "class_1", "samples", "steps")]
              == [19355, 645, 20000, 157], "Draw or step budget differs")
        orders[key] = row["sampled_indices_sha256"], row["actual_batch_order_sha256"]
    check(len(orders) == 200, "Missing sampler records")
    for method, seed in expected:
        for epoch in range(1, 21):
            check(orders[method, seed, epoch] == orders["baseline", seed, epoch], "Sampler mismatch")

    init = {(r["method"], int(r["seed"])): r for r in csv_rows("initialization.csv")}
    for method, seed in expected:
        field = "original_trainable_state_sha256"
        check(init[method, seed][field] == init["baseline", seed][field], "Shared initialization differs")
    checks = json.loads((RESULTS / "configuration_checks.json").read_text())
    check({(r["method"], r["seed"]) for r in checks} == expected, "Missing config audit")
    check(all(r["passed"] and r["formal"] and not r["differences"] for r in checks), "Formal config drift")
    for proof in json.loads((RESULTS / "raw_replay.json").read_text()):
        row = selected["fresh_both", proof["seed"]]
        check(proof["rows"] == 166079 and proof["changed_threshold_decisions"] == 0, "Replay differs")
        check(proof["full_original_six_experts_recomputed"] and proof["exact_f1"], "Raw replay incomplete")
        for k in ("tp", "fp", "tn", "fn"):
            check(proof["raw_metric"][k] == int(row[k]), "Raw confusion matrix differs")

    check(render_benchmark() == (ROOT / "paper/benchmark_rows.tex").read_text(), "Table 1 differs from its counts")
    temporal = csv_rows("time_control_sites.csv")
    verify_temporal_control(temporal, json.loads((RESULTS / "time_control_protocol.json").read_text()))
    check(temporal_table(temporal) == (ROOT / "paper/time_control_rows.tex").read_text(),
          "Temporal control table differs from its counts")
    verify_recording_budget()
    verify_external(selected)
    verify_operating_points(selected)
    for row in csv_rows("source_hashes.csv"):
        actual = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
        check(actual == row["published_sha256"], f"Unexpected source digest for {row['path']}")

    for method in METHODS:
        scores = [100 * float(selected[method, seed]["f1"]) for seed in SEEDS]
        print(f"{method:16s} seed 3 {scores[0]:.3f}  seed 42 {scores[1]:.3f}  mean {mean(scores):.3f}")
    print("Verified 10 selected checkpoints, 200 epochs and their paired sampling records.")


if __name__ == "__main__":
    main()
