"""Check all paired training records and lock the operating points before testing."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("dynamic", "fixed")
SEEDS = (3, 42)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def lock(runs):
    with (ROOT / "results/sampling.csv").open() as stream:
        historical = {(int(r["seed"]), int(r["epoch"])): r for r in csv.DictReader(stream)
                      if r["method"] == "baseline"}
    selected = {}
    for seed in SEEDS:
        compare = []
        for arm in ARMS:
            folder = runs / arm / f"seed{seed}"
            if not (folder / "completed.json").is_file():
                raise ValueError(f"Training incomplete: {arm}, seed {seed}")
            check = read_json(folder / "configuration_check.json")
            contract = read_json(folder / "strict_contract.json")
            init = read_json(folder / "paired_initialization.json")
            args = read_json(folder / "args.json")
            args.pop("output_dir")
            if not (check["formal"] and check["passed"] and not check["differences"]
                    and contract["method_loss_and_fusion"]["temporal_scorer_input"] == arm
                    and args["seed"] == init["seed"] == seed and init["arm"] == arm
                    and init["parameters"] == 2826520):
                raise ValueError("Formal paired configuration differs")
            compare.append((args, init["all_trainable_sha256"], contract["cached_execution"]))
            epochs = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
            if [r["epoch"] for r in epochs] != list(range(1, 21)):
                raise ValueError("Expected all 20 validation evaluations")
            for record in epochs:
                ep = record["epoch"]
                sample = read_json(folder / f"sampling_epoch{ep:02d}.json")
                expected = historical[seed, ep]
                if any(str(v) != expected[k] for k, v in sample.items()):
                    raise ValueError(f"Train stream differs at epoch {ep}")
                if (record["validation"]["samples"], record["validation"]["positives"],
                        record["train"]["samples"], record["train"]["steps"]) != (166079, 3471, 20000, 157):
                    raise ValueError("Draws, optimizer steps, or source validation differ")
            best = max(epochs, key=lambda r: (r["validation"]["best_f1_on_validation_grid"]["f1"], r["epoch"]))
            metric = best["validation"]["best_f1_on_validation_grid"]
            selected[f"{arm}_{seed}"] = dict(epoch=best["epoch"], threshold=metric["threshold"],
                    sha256=sha(folder / "best_f1.pt"), initialization=init["all_trainable_sha256"],
                    cached_counts={k: metric[k] for k in ("tp", "fp", "fn", "tn")})
        if compare[0] != compare[1]:
            raise ValueError(f"Initialization, training arguments or cache differ for seed {seed}")
    return dict(parameters_per_arm=2826520, epochs=20, samples_per_epoch=20000, steps_per_epoch=157,
                matched_to_original_training_stream=True, checkpoints=selected)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = lock(args.runs)
    if args.output.exists() and read_json(args.output) != payload:
        raise SystemExit("Existing lock refers to a different experiment")
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
