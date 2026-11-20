"""Rebuild Table 1 from public, aggregate window counts (no model weights needed)."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from math import isclose
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3, 42)
SITES = {"xiehe": (166079, 3471), "qilu": (106983, 4946), "tiantan": (325755, 4283),
         "chifeng": (46053, 440), "xian": (219711, 504)}
MODELS = {
    "eegnet": "EEGNet", "tcnet_eeg": "TCNet", "actnet_s": "ACTNet-S",
    "transformer_eeg": "EEG Transformer", "reve_base": "REVE Base",
    "resnet50": "ResNet-50", "densenet121": "DenseNet-121",
    "mobilenet_v3_large": "MobileNetV3-L", "efficientnet_v2_s": "EfficientNetV2-S",
    "convnext_large": "ConvNeXt-L", "vit_b_16_eeg": "ViT-B/16",
    "efficientvit_m4": "EfficientViT-M4", "efficientvim_m1": "EfficientViM-M1",
    "i2moe9e": "I2MOE9E", "baseline": "Six-expert baseline",
    "fresh_both": r"\textbf{Ours}",
}
ARCHIVED = ("resnet50", "densenet121", "mobilenet_v3_large", "efficientnet_v2_s", "convnext_large")
PAIRED = ("baseline", "fresh_both")


def read_csv(name: str) -> list[dict]:
    with (ROOT / "results" / name).open() as stream:
        return list(csv.DictReader(stream))


def checked_f1(row: dict, site: str) -> float:
    tp, fp, fn, tn = (int(row[k]) for k in ("tp", "fp", "fn", "tn"))
    if min(tp, fp, fn, tn) < 0 or (tp + fp + fn + tn, tp + fn) != SITES[site]:
        raise ValueError(f"Inconsistent cohort for {site}: {row}")
    f1 = 2 * tp / (2 * tp + fp + fn)
    if not isclose(f1, float(row["f1"]), rel_tol=1e-12):
        raise ValueError(f"Inconsistent F1: {row}")
    return 100 * f1


def table_row(method: str, values: list[float]) -> str:
    cells = [f"{v:.2f}" for v in values]
    if method == "fresh_both":
        cells = [rf"\textbf{{{v}}}" if i == len(cells) - 1 else v for i, v in enumerate(cells)]
    return " & ".join([MODELS[method], *cells]) + r"\\"


def render() -> str:
    histories = defaultdict(list)
    for row in read_csv("benchmark_validation.csv"):
        checked_f1(row, "xiehe")
        threshold = float(row["threshold"])
        if not any(isclose(threshold, i / 20) for i in range(1, 20)):
            raise ValueError(f"Unexpected source threshold: {row}")
        histories[row["protocol"], row["model"], int(row["seed"])].append(row)

    selected = {}
    protocols = {"source_f1_19": MODELS, "external_ap19": ARCHIVED}
    expected = {(p, m, s) for p, models in protocols.items() for m in models for s in SEEDS}
    if histories.keys() != expected:
        raise ValueError("Unexpected source histories")
    for (protocol, model, seed), rows in histories.items():
        if [int(r["epoch"]) for r in rows] != list(range(1, 21)):
            raise ValueError(f"Incomplete epoch history for {model}, seed {seed}")
        metric = "auprc" if protocol == "external_ap19" else "f1"
        selected[protocol, model, seed] = max(rows, key=lambda r: float(r[metric]))

    source = []
    for model in MODELS:
        if model == "i2moe9e":
            source.append(r"\midrule")
        scores = [checked_f1(selected["source_f1_19", model, s], "xiehe") for s in SEEDS]
        source.append(table_row(model, [*scores, mean(scores)]))

    operating_points = {(m, s): selected["external_ap19", m, s] for m in ARCHIVED for s in SEEDS}
    for row in read_csv("selected.csv"):
        if row["method"] in PAIRED:
            operating_points[row["method"], int(row["seed"])] = row

    transfer = {}
    for name, field in (("benchmark_external.csv", "model"), ("external_sites.csv", "method")):
        for row in read_csv(name):
            method, seed, site = row[field], int(row["seed"]), row["site"]
            source_row = operating_points[method, seed]
            if (int(row["epoch"]) != int(source_row["epoch"]) or
                    not isclose(float(row["threshold"]), float(source_row["threshold"]))):
                raise ValueError(f"External operating point differs from Xiehe: {row}")
            key = (method, seed, site)
            if key in transfer or site == "xiehe":
                raise ValueError(f"Unexpected external entry: {key}")
            transfer[key] = checked_f1(row, site)

    def hospital_rows(models: tuple[str, ...]) -> list[str]:
        lines = []
        for model in models:
            xiehe = mean(checked_f1(operating_points[model, s], "xiehe") for s in SEEDS)
            hospitals = [mean(transfer[model, s, site] for s in SEEDS)
                         for site in SITES if site != "xiehe"]
            lines.append(table_row(model, [xiehe, *hospitals, mean(hospitals)]))
        return lines

    blocks = {"BenchmarkSource": source, "BenchmarkArchived": hospital_rows(ARCHIVED),
              "BenchmarkPaired": hospital_rows(PAIRED)}
    return "".join("\\newcommand{\\" + name + "}{%\n" + "\n".join(rows) + "\n}\n"
                   for name, rows in blocks.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update the LaTeX rows")
    args = parser.parse_args()
    rows = render()
    path = ROOT / "paper/benchmark_rows.tex"
    if args.write:
        path.write_text(rows)
    elif rows != path.read_text():
        raise SystemExit("Table 1 differs from its aggregate counts; run with --write")
    print("Verified Table 1 across 16 source models, 2 seeds and 4 external hospitals.")


if __name__ == "__main__":
    main()
