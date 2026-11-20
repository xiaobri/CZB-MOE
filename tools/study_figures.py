"""Draw the model diagram and the six-expert appendix panels from published aggregates."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
META = {"CreationDate": None, "ModDate": None}
EXPERTS = ("frontal", "left_temporal", "right_temporal", "centro_parietal", "occipital", "redundancy")
NAMES = ("Frontal", "Left\ntemporal", "Right\ntemporal", "Centro-\nparietal", "Occipital", "Global")
SITES = ("xiehe", "qilu", "tiantan", "chifeng", "xian")
SITE_NAMES = ("Xiehe (val.)", "Qilu", "Tiantan", "Chifeng", "Xi’an")


def architecture():
    fig, ax = plt.subplots(figsize=(7, 2.35))
    ax.set(xlim=(0, 12.2), ylim=(0, 4))
    ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.12",
                                    facecolor=color, edgecolor="#596975", lw=0.8))
        ax.text(x+w/2, y+h*0.65, title, ha="center", va="center", fontsize=8, weight="semibold")
        ax.text(x+w/2, y+h*0.30, sub, ha="center", va="center", fontsize=6.8)

    def arrow(points, dashed=False):
        xs, ys = zip(*points)
        options = {"color": "#475867", "lw": 1.0, "ls": "--" if dashed else "-"}
        if len(points) > 2:
            ax.plot(xs[:-1], ys[:-1], **options)
        ax.annotate("", xy=points[-1], xytext=points[-2],
                    arrowprops=dict(arrowstyle="-|>", shrinkA=0, shrinkB=0, mutation_scale=9, **options))

    box(.08, 1.9, 1.65, .96, "EEG window", "19 × 2,000", "#e6f0f4")
    box(2.12, 1.78, 2.05, 1.20, "Regional experts", "Five frozen CNNs", "#e6f0f4")
    box(4.77, 2.86, 2.6, .97, "Global expert + router", "Six logits and routing weights", "#e6f0f4")
    box(4.77, 1.55, 2.6, .97, "Temporal readout", "Segment and peak evidence", "#e6f0f4")
    box(8.0, 2.26, 1.75, 1.02, "Parent margin", "Six-expert fusion", "#e6f0f4")
    box(8.0, .62, 1.75, 1.02, "Pair correction", "Fitted projection", "#eef1e5")
    box(10.40, 1.65, 1.72, 1.02, "Final score", "Parent + correction", "#d7ebdf")
    ax.text(2.98, .94, "Training pairs\nfrom one acquisition", ha="center", va="center",
            fontsize=7.6, color="#525b32")
    arrow([(1.76,2.38),(2.08,2.38)])
    arrow([(4.20,2.58),(4.5,2.58),(4.5,3.34),(4.72,3.34)])
    arrow([(4.20,2.15),(4.72,2.08)])
    arrow([(7.40,3.34),(7.68,3.34),(7.68,2.93),(7.95,2.93)])
    arrow([(7.40,2.03),(7.68,2.03),(7.68,2.54),(7.95,2.54)])
    arrow([(6.12,1.52),(6.12,1.30),(7.95,1.30)])
    arrow([(4.11,.87),(7.95,.87)], dashed=True)
    arrow([(9.79,2.78),(10.02,2.78),(10.02,2.40),(10.35,2.40)])
    arrow([(9.79,1.14),(10.02,1.14),(10.02,1.90),(10.35,1.90)])
    ax.text(.10,.18,"Dashed arrow denotes training supervision. The output probability is the sigmoid of the summed margins.",
            va="center", fontsize=7, color="#374650")
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
    fig.savefig(ROOT / "paper/architecture.pdf", metadata=META)
    plt.close(fig)


def activity_panels():
    with (ROOT / "results/expert_activity.csv").open() as stream:
        data = {(r["site"],r["method"],int(r["seed"]),int(r["label"]),r["expert"]):r for r in csv.DictReader(stream)}
    for field, filename, heading in (("mean_weight","expert_routing","Mean routing weight (%)"),
                                     ("mean_expert_score","expert_scores","Mean positive-class expert score (%)")):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5), sharex=True, sharey=True)
        for row, seed in enumerate((3,42)):
            for col,label in enumerate((0,1)):
                ax = axes[row,col]
                values = np.array([[100*float(data[site,"fresh_both",seed,label,e][field])
                                    for e in EXPERTS] for site in SITES])
                ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
                for (y,x),v in np.ndenumerate(values):
                    ax.text(x,y,f"{v:.1f}",ha="center",va="center",fontsize=7.4,
                            color="white" if v>57 else "#172737")
                ax.set_title(f"Seed {seed} · {'Positive' if label else 'Negative'} windows",fontsize=9)
                ax.set_xticks(range(6),NAMES,fontsize=7)
                ax.set_yticks(range(5),SITE_NAMES,fontsize=8)
                ax.tick_params(length=0,pad=3)
                ax.spines[:].set_visible(False)
        fig.suptitle(heading,fontsize=11,y=.99)
        fig.subplots_adjust(left=.12,right=.99,top=.88,bottom=.09,hspace=.28,wspace=.075)
        fig.savefig(ROOT / f"paper/{filename}.pdf",metadata=META)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.45), sharex=True, sharey=True)
    for idx, (expert, ax) in enumerate(zip(EXPERTS, axes.flat)):
        for seed, offset, color, marker in ((3, -0.13, "#087675", "o"), (42, 0.13, "#99501f", "s")):
            rows = [data[site, "fresh_both", seed, 1, expert] for site in SITES]
            quartiles = np.array([[float(r[k])*100 for r in rows] for k in ("weight_q25", "weight_median", "weight_q75")])
            low, mid, high = quartiles
            ax.errorbar(mid, np.arange(5)+offset, xerr=np.stack((mid-low, high-mid)), fmt=marker,
                        color=color, markersize=3, lw=1, capsize=2, label=f"Seed {seed}")
        ax.set_title(NAMES[idx].replace("\n", " ").replace("- ", "-"), fontsize=9)
        ax.set_yticks(range(5), SITE_NAMES, fontsize=7)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=7)
        ax.set_xlim(-2,102)
        ax.set_ylim(4.5,-.5)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["right","top"]].set_visible(False)
    fig.supxlabel("Routing weight on positive windows (%)", fontsize=9, y=.025)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=.12, right=.985, top=.86, bottom=.13, hspace=.32, wspace=.12)
    fig.savefig(ROOT / "paper/expert_distributions.pdf", metadata=META)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-only", action="store_true")
    args = parser.parse_args()
    architecture()
    if not args.architecture_only:
        activity_panels()
