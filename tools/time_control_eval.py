"""Raw EEG replay at Xiehe and four hospitals, with source-locked temporal controls."""
import argparse
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import external_eval as external
from time_control_lock import ARMS, SEEDS, read_json, sha
sys.path.insert(0, str(external.ROOT / "source/I2MOE/test333/fair_static_time_control"))
from time_control import configure

KEYS = tuple(f"{arm}_{seed}" for arm in ARMS for seed in SEEDS)
SITES = ("xiehe", *external.HOSPITALS)


def load_models(runs, locks, device):
    models, archives, digests = {}, {}, {}
    source_args = None
    for key in KEYS:
        arm, seed = key.split("_")
        root = runs / arm / f"seed{seed}"
        path = root / "best_f1.pt"
        locked = locks[key]
        if sha(path) != locked["sha256"]:
            raise ValueError("Locked checkpoint changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metric = payload["validation"]["best_f1_on_validation_grid"]
        spec = payload["config"]["strict_extension"]["method_loss_and_fusion"]
        if (payload["epoch"] != locked["epoch"] or payload["seed"] != int(seed)
                or spec["temporal_scorer_input"] != arm or metric["threshold"] != locked["threshold"]
                or any(metric[k] != v for k, v in locked["cached_counts"].items())):
            raise ValueError("Checkpoint and locked operating point disagree")
        args = argparse.Namespace(**payload["args"])
        for name, path in external.trainer.frozen_checkpoint_paths(args).items():
            if path not in digests:
                digests[path] = sha(Path(path))
            if digests[path] != payload["config"]["frozen_checkpoint_identity"][name]["sha256"]:
                raise ValueError("Frozen encoder identity changed")
        external.set_seed(int(seed), args.deterministic)
        with patch.object(external.trainer, "SixExpertF1",
                          lambda **kw: configure(external.RefreshedPairMoE("fresh_both", **kw), arm)):
            model = external.trainer.build_model(args)
        model.load_trainable_state_dict(payload["model"])
        models[key] = model.to(device).eval()
        with np.load(root / f"predictions_epoch{locked['epoch']:02d}.npz", allow_pickle=False) as a:
            archives[key] = (a["labels"], a["probabilities"])
        source_args = args
    if len(digests) != 5:
        raise ValueError("Expected five frozen encoders")
    return models, archives, source_args


def evaluate(args):
    if torch.cuda.device_count() != 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Expected one visible CUDA device and a valid native-batch shard")
    torch.set_num_threads(2)
    locks = read_json(args.lock)["checkpoints"]
    device = torch.device("cuda:0")
    models, archives, source = load_models(args.runs, locks, device)
    historical = read_json(external.ROOT / "results/external_protocol.json")["hospitals"]
    for site in SITES:
        if site == "xiehe":
            manifest = Path(source.csv_path)
            frame = pd.read_csv(manifest, usecols=["file_path", "mode", "target"])
            dataset = external.trainer.EEG19NpyDataset(frame, "Val", source.stored_prefix, source.data_root,
                            target_samples=source.target_samples, eeg_scale_divisor=source.eeg_scale_divisor)
            manifest_hash = sha(manifest)
            expected_hash = read_json(external.ROOT / "results/protocol.json")["manifest"]["csv_sha256"]
        else:
            dataset, manifest_hash = external.dataset_for(site, args.storage)
            expected_hash = historical[site]["manifest_sha256"]
        if manifest_hash != expected_hash:
            raise ValueError(f"Manifest digest changed for {site}")
        n = len(dataset)
        nb = math.ceil(n / external.BATCH)
        lo = min(n, nb * args.shard // args.shards * external.BATCH)
        hi = min(n, nb * (args.shard + 1) // args.shards * external.BATCH)
        file = args.output / f"{site}_{args.shard}.json"
        contract = dict(site=site, shard=args.shard, shards=args.shards, start=lo, stop=hi, size=n,
                        manifest=manifest_hash, lock=sha(args.lock), script=sha(Path(__file__)))
        if file.exists():
            if read_json(file)["contract"] != contract:
                raise ValueError("Existing partition differs")
            continue
        args.output.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(Subset(dataset, range(lo, hi)), batch_size=external.BATCH, num_workers=2,
                            shuffle=False, pin_memory=True)
        offset = lo
        counts = {k: np.zeros(4, dtype=np.int64) for k in KEYS}
        changed = dict.fromkeys(KEYS, 0)
        deviation = dict.fromkeys(KEYS, 0.)
        weight_spread = {}
        for eeg, y in loader:
            labels = y.numpy()
            end = offset + len(labels)
            np.testing.assert_array_equal(dataset.labels[offset:end], labels)
            check = offset == lo or len(eeg) != external.BATCH
            with ExitStack() as hooks:
                if offset == lo:
                    for key, model in models.items():
                        def observe(module, inputs, results, name=key):
                            weights = results[2].detach()
                            weight_spread[name] = float((weights.max(0).values - weights.min(0).values).max())
                        hooks.enter_context(model.correction.allocation.register_forward_hook(observe))
                with patch.object(external, "KEYS", KEYS):
                    values, _ = external.predict(models, eeg.to(device, non_blocking=True), check)
            for i, key in enumerate(KEYS):
                p = values[:, i, 0].astype(np.float64)
                decision = p > locks[key]["threshold"]
                counts[key] += np.bincount(2 * labels + decision, minlength=4)
                if site == "xiehe":
                    truth, reference = archives[key]
                    np.testing.assert_array_equal(labels, truth[offset:end])
                    ref = reference[offset:end].astype(np.float64)
                    changed[key] += int(np.sum(decision != (ref > locks[key]["threshold"])))
                    deviation[key] = max(deviation[key], float(np.max(abs(p - ref))))
            offset = end
        if offset != hi or any(weight_spread[k] > 1e-6 for k in KEYS if k.startswith("fixed")):
            raise RuntimeError("Incomplete partition or window-dependent fixed weights")
        result = dict(contract=contract, weight_spread=weight_spread,
                      counts={k: dict(zip(("tn", "fp", "fn", "tp"), map(int, v))) for k, v in counts.items()})
        if site == "xiehe":
            result.update(changed_cached_decisions=changed, max_cached_score_error=deviation)
        external.atomic_write_json(file, result)
        print(f"{site} shard {args.shard} complete, {hi-lo} windows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runs", "storage", "output", "lock"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    evaluate(parser.parse_args())
