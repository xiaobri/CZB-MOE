# Within-Acquisition Pair Ranking for Six-Expert EEG Classification

[Read the paper (PDF)](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Selected results](results/selected.csv) · [Reproduction guide](REPRODUCING.md)

This study asks whether five frozen regional EEG experts and a global redundancy expert can recover more positive windows under the original training schedule. The full model learns to pool regional evidence over eight temporal segments and compares positive windows with previously observed negative windows from the same training acquisition group. At inference all six experts contribute, and the classifier only needs the EEG window.

### What improved?

Positive-class **validation F1**, in percent, on 166,079 windows. The baseline is the strict raw-input rerun. Each model selects its epoch and threshold with the same validation grid.

| Training method | Seed 3 | Seed 42 | Two-seed mean |
| --- | ---: | ---: | ---: |
| Strict baseline | 69.390 | 70.184 | 69.787 |
| Collection-time scores | 71.869 | 72.150 | 72.009 |
| Refresh positive scores | 71.849 | 72.068 | 71.959 |
| Refresh negative scores | 71.939 | 72.099 | 72.019 |
| Refresh both scores | **72.096** | **71.967** | **72.032** |

The full model gains **2.245 percentage points** on average. At its selected operating points it recovers 232 and 141 additional positives, at the cost of 168 and 87 additional false positives. Refreshing both scores adds only 0.023 points over their collection-time counterparts on average, and the sign differs by seed. The absolute gain covers the entire extended model, including its larger temporal readout and auxiliary supervision.

### How comparable are the runs?

The runs share the manifest, five frozen checkpoints, shared-parameter initialization, per-seed sample order, 20 epochs, 400,000 draws, 3,140 AdamW steps and threshold grid. The extended model uses training acquisition metadata and additional computation. Trainable parameter counts are 365,576 for the baseline and 2,826,520 for the full model. Replaying each selected full-model checkpoint from raw EEG changed zero decisions relative to cached execution. [Protocol](results/protocol.json) · [Raw replay](results/raw_replay.json)

The validation split was repeatedly used to develop methods. Three recording-directory basenames appear in both manifest splits. When their windows are excluded, the full model scores 70.296 and 70.127, compared with 66.995 and 67.981 for baseline. Patient-disjoint generalization remains unverified. [Overlap sensitivity](results/fresh_both_overlap_sensitivity.json) · [Recording-fold sensitivity](results/fresh_both_recording_selection.json)

### Code and data

- [Baseline trainer and six-expert model](source/I2MOE6_F1), [strict baseline rerun](source/I2MOE/test333/strict_baseline_20261115), [full method and three refresh controls](source/I2MOE/test333/fair_refreshed_pairs_20261120).
- [Pair estimator](source/I2MOE/test333/fair_risk_pairs_20261120/pair_models.py), [temporal pooling](source/I2MOE/test333/fair_expert_allocation_20261120/allocation_models.py) and [raw EEG replay](source/I2MOE/test333/fair_refreshed_pairs_20261120/replay_raw.py).
- [All 200 epoch results](results/epochs.csv), [sampling hashes](results/sampling.csv), [initialization hashes](results/initialization.csv) and [conditional recording-bootstrap intervals](results/paired_bootstrap.json).
- [Original and released source digests](results/source_hashes.csv). The release preserves the experiment implementations and maps workstation-specific locations to configurable directories. The EEG files, per-window predictions, recordings, private manifest and checkpoint weights are held separately.

Run `python3 tools/verify_results.py` from a clone to reproduce the headline numbers and check the paired protocol using only the Python standard library. The manuscript uses the official ICML 2027 preprint layout. This repository is a research release. [ICML author instructions](https://icml.cc/Conferences/2027/AuthorInstructions)
