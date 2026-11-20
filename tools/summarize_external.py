"""Publish aggregate fixed-threshold performance and class-conditional expert activity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SITES = ("qilu", "tiantan", "chifeng", "xian")
EXPERTS = ("frontal", "left_temporal", "right_temporal", "centro_parietal", "occipital", "redundancy")
KEYS = [(m, s) for m in ("baseline", "fresh_both") for s in (3, 42)]


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def activities(site, method, seed, y, weights, expert_scores):
    rows = []
    if weights.shape != expert_scores.shape or weights.shape != (len(y), 6):
        raise ValueError("Unexpected expert layout")
    if not np.isfinite(expert_scores).all() or not np.isfinite(weights).all():
        raise ValueError("Invalid expert outputs")
    np.testing.assert_allclose(weights.sum(1), 1, rtol=0, atol=1e-5)
    for label in (0, 1):
        mask = y == label
        w, p = weights[mask], expert_scores[mask]
        wins = np.bincount(w.argmax(1), minlength=6)
        q25, median, q75 = np.quantile(w, [0.25, 0.5, 0.75], axis=0)
        for i, expert in enumerate(EXPERTS):
            rows.append(dict(site=site, method=method, seed=seed, label=label, expert=expert,
                             windows=len(w), mean_weight=w[:, i].mean(dtype=np.float64),
                             mean_expert_score=p[:, i].mean(dtype=np.float64), top_weight_windows=int(wins[i]),
                             weight_q25=float(q25[i]), weight_median=float(median[i]), weight_q75=float(q75[i])))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("private", "baseline", "candidate", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "results/selected.csv").open() as stream:
        selection = {(r["method"], int(r["seed"])): r for r in csv.DictReader(stream)}
    summary, activity, provenance = [], [], {}
    identities = None
    for site in SITES:
        done = [json.loads(p.read_text()) for p in (args.private / site).glob("shard*/complete.json")]
        done.sort(key=lambda d: d["contract"]["shard"])
        if not done:
            raise ValueError(f"No completed partitions for {site}")
        first = done[0]["contract"]
        if len(done) != first["shards"]:
            raise ValueError(f"Incomplete {site}")
        if identities is not None and identities != first["checkpoints"]:
            raise ValueError("Checkpoint identity changed between hospitals")
        identities = first["checkpoints"]
        scores, labels, offset, parity_rows = [], [], 0, 0
        for i, item in enumerate(done):
            c = item["contract"]
            if (c["shard"] != i or c["start"] != offset or c["hospital"] != site or c["shards"] != len(done)
                    or c["manifest_sha256"] != first["manifest_sha256"] or c["checkpoints"] != identities
                    or c["script_sha256"] != first["script_sha256"] or c["total"] != first["total"]
                    or c["models"] != [f"{m}_{s}" for m, s in KEYS]
                    or c["experts"] != ["unique_" + name for name in EXPERTS[:-1]] + ["redundancy"]):
                raise ValueError(f"Inconsistent partition {site} / {i}")
            path = args.private / site / f"shard{i}" / "scores.npz"
            if item["sha256"] != digest(path):
                raise ValueError("Scores changed after completion")
            with np.load(path, allow_pickle=False) as data:
                y, s = data["labels"], data["scores"]
                if s.shape != (c["stop"] - offset, 4, 13) or y.shape != (len(s),):
                    raise ValueError("Incomplete partition")
                scores.append(s)
                labels.append(y)
            offset = c["stop"]
            if not item["direct_checks"] or any(any(e != 0 for e in a["maximum_direct_difference"].values())
                                                 for a in item["direct_checks"]):
                raise ValueError("Shared-feature parity check failed")
            parity_rows += sum(a["rows"] for a in item["direct_checks"])
        if offset != first["total"]:
            raise ValueError("Coverage does not reach end of manifest")
        y = np.concatenate(labels)
        arr = np.concatenate(scores)
        provenance[site] = dict(manifest_sha256=first["manifest_sha256"], windows=len(y),
                                positives=int(y.sum()), partitions=len(done), native_batch_size=first["batch_size"],
                                direct_check_windows=parity_rows, maximum_direct_output_difference=0,
                                inference_script_sha256=first["script_sha256"])
        for i, (method, seed) in enumerate(KEYS):
            row = selection[method, seed]
            threshold = float(row["threshold"])
            ck = identities[f"{method}_{seed}"]
            if ck["epoch"] != int(row["epoch"]) or ck["threshold"] != threshold:
                raise ValueError("Locked selection mismatch")
            p = arr[:, i, 0].astype(np.float64)
            predicted = p > threshold
            tp = int(np.sum(predicted & (y == 1)))
            fp = int(np.sum(predicted & (y == 0)))
            fn = int(y.sum()) - tp
            tn = len(y) - tp - fp - fn
            summary.append(dict(site=site, method=method, seed=seed, epoch=ck["epoch"], threshold=threshold,
                                windows=len(y), positives=tp+fn, tp=tp, fp=fp, fn=fn, tn=tn,
                                f1=2*tp/(2*tp+fp+fn), precision=tp/max(1,tp+fp), recall=tp/(tp+fn),
                                specificity=tn/(tn+fp), accuracy=(tp+tn)/len(y),
                                auprc=average_precision_score(y,p), auroc=roc_auc_score(y,p)))
            activity.extend(activities(site, method, seed, y, arr[:,i,1:7], arr[:,i,7:]))
    for method, seed in KEYS:
        epoch = int(selection[method, seed]["epoch"])
        run = (args.baseline if method == "baseline" else args.candidate) / f"seed{seed}"
        folder = run if method == "baseline" else run / "raw_replay"
        with np.load(folder / f"predictions_epoch{epoch:02}.npz", allow_pickle=False) as saved:
            activity.extend(activities("xiehe", method, seed, saved["labels"], saved["weights"],
                                       saved["expert_probabilities"]))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "external_sites.csv", summary)
    write_csv(args.output / "expert_activity.csv", activity)
    (args.output / "external_protocol.json").write_text(json.dumps(
        dict(source_site="xiehe", selection="Source validation best F1 checkpoint and threshold, fixed before external inference.",
             checkpoints=identities, hospitals=provenance), indent=2) + "\n")


if __name__ == "__main__":
    main()
