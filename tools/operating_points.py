"""Summarize archived raw validation predictions at the prespecified epochs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_scores(run: Path, epoch: int):
    with np.load(run / f"predictions_epoch{epoch:02}.npz", allow_pickle=False) as saved:
        labels = saved["labels"]
        scores = saved["probabilities"].astype(np.float64)
    if scores.shape != (166079,) or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Unexpected validation arrays")
    return scores, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Formal strict baseline directory containing seed folders")
    parser.add_argument("--candidate", type=Path, required=True, help="Formal fresh_both directory containing seed folders")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "results/selected.csv").open() as stream:
        selected = {(r["method"], int(r["seed"])): r for r in csv.DictReader(stream)}
    grid = np.asarray([0.01 + i * 0.01 for i in range(99)])
    output = []
    for seed in (3, 42):
        models = {}
        reference_labels = None
        decisions = {}
        for method, path in (("baseline", args.baseline / f"seed{seed}"),
                             ("fresh_both", args.candidate / f"seed{seed}" / "raw_replay")):
            row = selected[method, seed]
            scores, labels = load_scores(path, int(row["epoch"]))
            if reference_labels is None:
                reference_labels = labels
            np.testing.assert_array_equal(labels, reference_labels)
            np.testing.assert_array_equal(np.bincount(labels), [162608, 3471])
            pos, neg = np.sort(scores[labels == 1]), np.sort(scores[labels == 0])
            tp = len(pos) - np.searchsorted(pos, grid, side="right")
            fp = len(neg) - np.searchsorted(neg, grid, side="right")
            cutoff = float(row["threshold"])
            decisions[method] = scores > cutoff
            idx = np.flatnonzero(grid == cutoff).item()
            if (int(tp[idx]), int(fp[idx])) != (int(row["tp"]), int(row["fp"])):
                raise ValueError("Archived predictions differ from the published selection")
            models[method] = {"tp": tp.tolist(), "fp": fp.tolist()}
        switches = {}
        for label, name in ((1, "positive"), (0, "negative")):
            group = reference_labels == label
            first, second = decisions["baseline"], decisions["fresh_both"]
            switches[name] = {
                "baseline_only": int(np.sum(group & first & ~second)),
                "full_only": int(np.sum(group & second & ~first)),
                "both_positive": int(np.sum(group & first & second)),
                "both_negative": int(np.sum(group & ~first & ~second)),
            }
        budget = int(selected["baseline", seed]["fp"])
        first_feasible = next(i for i, n in enumerate(models["fresh_both"]["fp"]) if n <= budget)
        output.append({"seed": seed, "thresholds": grid.tolist(), "models": models,
                       "decision_overlap": switches, "baseline_fp_budget": budget,
                       "lowest_feasible_full_grid_index": first_feasible})
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
