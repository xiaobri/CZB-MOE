"""Transfer a baseline-derived false-positive budget across five recording folds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from operating_points import ROOT, load_scores


def counts(scores, labels, mask, threshold):
    predicted = scores[mask] > threshold
    y = labels[mask]
    return dict(tp=int(np.sum(predicted & (y == 1))), fp=int(np.sum(predicted & (y == 0))),
                fn=int(np.sum(~predicted & (y == 1))), tn=int(np.sum(~predicted & (y == 0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "results/protocol.json").read_text())
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != protocol["manifest"]["csv_sha256"]:
        raise ValueError("Input manifest differs from the fixed study split")
    frame = pd.read_csv(args.manifest, usecols=["mode", "file_path", "target"])
    val = frame[frame["mode"] == "Val"]
    labels = val.target.to_numpy()
    folders = val.file_path.str.rsplit("/", n=1).str[0].to_numpy()
    splits = list(StratifiedGroupKFold(5).split(labels[:, None], labels, folders))
    reference = json.loads((ROOT / "results/fresh_both_recording_selection.json").read_text())["seeds"]
    results = []
    grid = np.array([0.01 + i * 0.01 for i in range(99)])
    for seed in (3, 42):
        previous = reference[str(seed)]
        for index, (selection, held) in enumerate(splits):
            score = {}
            for key, root in (("baseline", args.baseline), ("candidate", args.candidate)):
                point = previous[key]["folds"][index]
                values, y = load_scores(root / f"seed{seed}", point["selected_epoch"])
                np.testing.assert_array_equal(labels, y)
                reproduced = counts(values, y, held, point["threshold"])
                assert [reproduced[k] for k in ("tp", "fp", "fn")] == point["held_counts"]
                score[key] = values
            base, full = (previous[key]["folds"][index] for key in ("baseline", "candidate"))
            budget = counts(score["baseline"], labels, selection, base["threshold"])["fp"]
            neg = np.sort(score["candidate"][selection[labels[selection] == 0]])
            fp = len(neg) - np.searchsorted(neg, grid, side="right")
            feasible = np.flatnonzero(fp <= budget)
            if not len(feasible):
                raise ValueError("The original grid has no feasible candidate threshold")
            j = int(feasible[0])
            results.append(dict(seed=seed, fold=index + 1, held_recordings=len(np.unique(folders[held])),
                                baseline_epoch=base["selected_epoch"], full_epoch=full["selected_epoch"],
                                baseline_threshold=base["threshold"], full_threshold=float(grid[j]),
                                selection_fp_budget=budget, selection_full_fp=int(fp[j]),
                                selection_previous_full_fp=int(fp[j-1]) if j else None,
                                held_baseline=counts(score["baseline"], labels, held, base["threshold"]),
                                held_full=counts(score["candidate"], labels, held, grid[j])))
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
