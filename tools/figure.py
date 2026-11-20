"""Regenerate the manuscript figure from published epoch-level counts."""
import csv
import json
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

# Show attainable integer counts on the original threshold grid near both operating points.

audits = json.loads((ROOT / "results/selected_operating_points.json").read_text())
with (ROOT / "results/selected.csv").open() as stream:
    selected = {(row["method"], int(row["seed"])): row for row in csv.DictReader(stream)}
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.8, 2.5))
for axis, audit in zip(axes, audits):
    seed = audit["seed"]
    for method, label, color, line in (("baseline", "Baseline", "#555555", "--"),
                                      ("fresh_both", "Full model", "#166a80", "-")):
        metrics = audit["models"][method]
        axis.plot(metrics["fp"], metrics["tp"], line, lw=1.15, color=color, label=label)
        visible = [(x, y) for x, y in zip(metrics["fp"], metrics["tp"]) if 350 <= x <= 1400]
        axis.scatter(*zip(*visible), s=7, color=color)
        row = selected[method, seed]
        axis.scatter(int(row["fp"]), int(row["tp"]), s=48, marker="s", facecolors="white",
                     edgecolors=color, zorder=4)
    limit = audit["baseline_fp_budget"]
    axis.axvline(limit, lw=0.8, linestyle=":", color="#777777")
    j = audit["lowest_feasible_full_grid_index"]
    full = audit["models"]["fresh_both"]
    axis.scatter(full["fp"][j], full["tp"][j], marker="D", s=44, color="#166a80", zorder=4)
    axis.set(xlim=(350, 1400), ylim=(1650, 2800), title=f"Seed {seed}", xlabel="False-positive windows")
    axis.tick_params(labelsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("True-positive windows")
axes[1].legend(loc="lower right", frameon=False, fontsize=8)
fig.tight_layout(pad=0.45)
fig.savefig(ROOT / "paper/operating_points.pdf", metadata={"CreationDate": None, "ModDate": None})
