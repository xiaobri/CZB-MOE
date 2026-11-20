"""Evaluate every source-selected component model on the same raw Xiehe windows."""
import argparse
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import external_eval as external
from component_records import CONTROLS, COUNTS, PAIR_CONTROLS, STAGES, location
from time_control_lock import ROOT, read_json, sha

sys.path.insert(0,str(ROOT / "source/I2MOE/test333/fair_component_attribution"))
from components import configure
from allocation_models import ExpertAllocationMoE
from complement_models import ComplementMoE
from pair_models import RiskPairMoE
from peak_models import PeakShapeMoE
from refresh_models import RefreshedPairMoE

ARMS = (*STAGES, *PAIR_CONTROLS, *CONTROLS)
BASE = external.trainer.SixExpertF1


def construct(arm, **kw):
    if arm in CONTROLS:
        return configure(ComplementMoE("risk_power", **kw), arm)
    if arm == "baseline":
        return BASE(**kw)
    if arm == "risk_power":
        return ComplementMoE(arm, **kw)
    if arm == "peaks":
        return PeakShapeMoE(arm, **kw)
    if arm in ("independent", "independent_aux"):
        return ExpertAllocationMoE(arm, **kw)
    if arm in PAIR_CONTROLS:
        return RiskPairMoE(arm, **kw)
    return RefreshedPairMoE(arm, **kw)


def load_models(args, lock, device):
    models, cache, seen = {}, {}, {}
    flags = None
    for arm in ARMS:
        folder = location(args.original,args.controls,arm,args.seed)
        ck = lock[f"{arm}_{args.seed}"]
        path = folder/"best_f1.pt"
        if sha(path) != ck["best_checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after source selection")
        payload = torch.load(path,map_location="cpu",weights_only=False)
        val = payload["validation"]["best_f1_on_validation_grid"]
        config = payload["config"]
        method = config["component_control"]["arm"] if arm in CONTROLS else config["strict_extension"]["method"]
        if (method != arm or payload["epoch"] != ck["epoch"] or payload["seed"] != args.seed
                or val["threshold"] != ck["threshold"] or any(val[k] != ck["cached_counts"][k] for k in COUNTS)):
            raise ValueError(f"Inconsistent selected snapshot for {arm}")
        flags = argparse.Namespace(**payload["args"])
        for name, source in external.trainer.frozen_checkpoint_paths(flags).items():
            if source not in seen:
                seen[source] = sha(Path(source))
            if seen[source] != config["frozen_checkpoint_identity"][name]["sha256"]:
                raise ValueError("Different frozen encoders")
        external.set_seed(args.seed, flags.deterministic)
        with patch.object(external.trainer, "SixExpertF1", lambda **kw: construct(arm,**kw)):
            model = external.trainer.build_model(flags)
        model.load_trainable_state_dict(payload["model"])
        models[arm] = model.to(device).eval()
        with np.load(folder / f"predictions_epoch{ck['epoch']:02d}.npz",allow_pickle=False) as a:
            cache[arm] = a["labels"], a["probabilities"]
    if len(seen) != 5:
        raise ValueError("Expected five frozen sources")
    return models, cache, flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original","controls","lock","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--seed",type=int,choices=(3,42),required=True)
    parser.add_argument("--shard",type=int,required=True)
    parser.add_argument("--shards",type=int,default=4)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or not 0 <= args.shard < args.shards:
        raise ValueError("Invalid worker")
    torch.set_num_threads(2)
    lock = read_json(args.lock)
    device = torch.device("cuda:0")
    models,cache,flags = load_models(args,lock,device)
    frame = pd.read_csv(flags.csv_path, usecols=["target","mode","file_path"])
    data = external.trainer.EEG19NpyDataset(frame,"Val",flags.stored_prefix,flags.data_root,
                    target_samples=flags.target_samples,eeg_scale_divisor=flags.eeg_scale_divisor)
    manifest_hash = sha(Path(flags.csv_path))
    if manifest_hash != read_json(ROOT/"results/protocol.json")["manifest"]["csv_sha256"] or len(data)!=166079:
        raise ValueError("Unexpected Xiehe manifest")
    batch = external.BATCH
    batches = math.ceil(len(data)/batch)
    lo = min(len(data),batch*(batches*args.shard//args.shards))
    hi = min(len(data),batch*(batches*(args.shard+1)//args.shards))
    contract = dict(seed=args.seed,shard=args.shard,shards=args.shards,start=lo,stop=hi,
                    manifest=manifest_hash,lock=sha(args.lock),script=sha(Path(__file__)))
    out = args.output/f"seed{args.seed}_shard{args.shard}.json"
    if out.exists():
        if read_json(out)["contract"] != contract:
            raise ValueError("Existing replay differs")
        return
    args.output.mkdir(exist_ok=True,parents=True)
    loader = DataLoader(Subset(data,range(lo,hi)),batch_size=batch,shuffle=False,num_workers=2,pin_memory=True)
    counts = {arm:np.zeros(4,dtype=np.int64) for arm in ARMS}
    changed = dict.fromkeys(ARMS,0)
    offset = lo
    with patch.object(external,"KEYS",ARMS):
        for eeg,targets in loader:
            y = targets.numpy()
            end = offset+len(y)
            np.testing.assert_array_equal(y,data.labels[offset:end])
            scores,_ = external.predict(models,eeg.to(device,non_blocking=True),offset==lo or len(y)!=batch)
            for i,arm in enumerate(ARMS):
                threshold=lock[f"{arm}_{args.seed}"]["threshold"]
                pred=scores[:,i,0].astype(np.float64)>threshold
                counts[arm] += np.bincount(2*y+pred,minlength=4)
                labels,reference=cache[arm]
                np.testing.assert_array_equal(labels[offset:end],y)
                changed[arm] += int(np.sum(pred != (reference[offset:end].astype(np.float64)>threshold)))
            offset=end
            if (offset-lo)%(batch*100)==0:
                print(f"{offset-lo}/{hi-lo}",flush=True)
    if offset!=hi:
        raise ValueError("Incomplete raw replay")
    external.atomic_write_json(out,dict(contract=contract,changed=changed,
                        counts={arm:dict(zip(("tn","fp","fn","tp"),map(int,c))) for arm,c in counts.items()}))


if __name__=="__main__":
    main()
