"""Verify released metrics, sampler pairing and source digests without EEG or PyTorch."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

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
