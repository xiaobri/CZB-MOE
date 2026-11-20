"""Regenerate the manuscript figure from published epoch-level counts."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "results/epochs.csv").open() as stream:
    rows = list(csv.DictReader(stream))
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(3.25, 2.55))
for seed, axis in zip(("3", "42"), axes):
    for method, label, color, dash in (
        ("baseline", "Baseline", "#555555", "--"),
        ("fresh_both", "Full method", "#166a80", "-"),
    ):
        run = [r for r in rows if r["seed"] == seed and r["method"] == method]
        xs = [int(r["epoch"]) for r in run]
        ys = [100 * float(r["f1"]) for r in run]
        axis.plot(xs, ys, dash, color=color, lw=1.15, label=label)
        best = ys.index(max(ys))
        axis.plot(xs[best], ys[best], "o", color=color, ms=3)
    axis.set_ylabel("F1 (%)", fontsize=8)
    axis.text(0.98, 0.07, f"Seed {seed}", ha="right", transform=axis.transAxes, fontsize=7)
    axis.tick_params(labelsize=7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
axes[0].legend(loc="lower left", fontsize=6, frameon=False, ncol=2)
axes[1].set(xlabel="Training epoch", xticks=[1, 5, 10, 15, 20], xlim=(0.5, 20.5))
fig.tight_layout(pad=0.35)
fig.savefig(ROOT / "paper/epoch_f1.pdf", metadata={"CreationDate": None, "ModDate": None})
