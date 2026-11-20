# Within-Acquisition Pair Ranking for Six-Expert EEG Classification

[Read the paper (PDF)](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Selected results](results/selected.csv) · [Reproduction guide](REPRODUCING.md)

This study asks whether five frozen regional EEG experts and a global redundancy expert can recover more positive windows under the original training schedule. The full model learns to pool regional evidence over eight temporal segments and compares positive windows with previously observed negative windows from the same training acquisition group. At inference all six experts contribute, and the classifier only needs the EEG window.

### Main benchmark

Table 1 of the paper compares **15 baselines and the proposed model** on the same 166,079 Xiehe validation windows. The table gives each seed and its two-seed mean, plus the four external hospitals wherever there are matching archived tests. [Table 1 in the paper](paper/main.pdf#page=5) · [Validation counts by epoch](results/benchmark_validation.csv) · [Backbone external counts](results/benchmark_external.csv)

| Xiehe validation method | Seed 3 | Seed 42 | Mean F1 (%) |
| --- | ---: | ---: | ---: |
| EEGNet | 33.47 | 33.68 | 33.58 |
| TCNet | 30.87 | 25.88 | 28.37 |
| ACTNet-S | 6.87 | 4.76 | 5.81 |
| EEG Transformer | 14.48 | 13.76 | 14.12 |
| REVE Base | 25.61 | 24.10 | 24.85 |
| ResNet-50 | 63.47 | 66.46 | 64.96 |
| DenseNet-121 | 62.79 | 63.96 | 63.37 |
| MobileNetV3-L | 61.67 | 62.46 | 62.07 |
| EfficientNetV2-S | 64.13 | 65.22 | 64.68 |
| ConvNeXt-L | 4.10 | 4.09 | 4.10 |
| ViT-B/16 | 5.56 | 21.72 | 13.64 |
| EfficientViT-M4 | 48.53 | 50.48 | 49.50 |
| EfficientViM-M1 | 55.70 | 55.27 | 55.48 |
| I2MOE9E | 69.80 | 70.47 | 70.13 |
| Six-expert baseline | 69.37 | 69.91 | 69.64 |
| **Proposed model** | 72.10 | 71.97 | **72.03** |

Every row above uses epochs 1–20, batch 128, 20,000 draws per epoch, and the same 19 Xiehe thresholds. The primary paired experiment below uses its original 99 thresholds. This changes the strict baseline mean from 69.64 to 69.79. The proposed model selects the same operating points on both grids. Independent architectures use their own preprocessing and initialization.

### What improved within the six-expert model?

Positive-class **validation F1**, in percent, on 166,079 windows. The baseline is the strict raw-input rerun. Each model selects its epoch and threshold with the same validation grid.

| Training method | Seed 3 | Seed 42 | Two-seed mean |
| --- | ---: | ---: | ---: |
| Strict baseline | 69.390 | 70.184 | 69.787 |
| Collection-time scores | 71.869 | 72.150 | 72.009 |
| Refresh positive scores | 71.849 | 72.068 | 71.959 |
| Refresh negative scores | 71.939 | 72.099 | 72.019 |
| Refresh both scores | **72.096** | **71.967** | **72.032** |

The full model gains **2.245 percentage points** on average. At its selected operating points it recovers 232 and 141 additional positives, at the cost of 168 and 87 additional false positives. Refreshing both scores adds only 0.023 points over their collection-time counterparts on average, and the sign differs by seed. Under a false-positive budget equal to the baseline count, the full model still detects 28 and 17 more positive windows for the two seeds on the original validation grid. [Threshold sweep and decision overlap](results/selected_operating_points.json)

The absolute gain covers the entire extended model, including its larger temporal readout and auxiliary supervision.

### What transferred to four other hospitals?

All 698,502 external windows were evaluated at the Xiehe-selected checkpoints and thresholds. Each cell averages the two seed-specific F1 scores, in percent.

| Hospital | Windows (positive) | Strict baseline | Full model | Difference |
| --- | ---: | ---: | ---: | ---: |
| Qilu | 106,983 (4,946) | 61.89 | 63.77 | +1.88 |
| Tiantan | 325,755 (4,283) | 42.37 | 41.57 | −0.80 |
| Chifeng | 46,053 (440) | 42.22 | 45.73 | +3.51 |
| Xi’an | 219,711 (504) | 24.19 | 31.44 | +7.25 |
| Equal-site mean | | 42.67 | 45.63 | +2.96 |

At Tiantan both seeds recover more positives and generate more false positives. Xi’an shows the largest F1 gain, but mean recall of the full model there is 21.13%. The appendix reports each seed and separates dense routing weights from expert positive-class scores. [All 16 external results](results/external_sites.csv) · [Six-expert routing heatmaps](paper/expert_routing.pdf) · [Expert scores](paper/expert_scores.pdf) · [Per-expert routing distributions](paper/expert_distributions.pdf) · [Aggregate data](results/expert_activity.csv)

### How comparable are the runs?

The runs share the manifest, five frozen checkpoints, shared-parameter initialization, per-seed sample order, 20 epochs, 400,000 draws, 3,140 AdamW steps and threshold grid. The extended model uses training acquisition metadata and additional computation. Trainable parameter counts are 365,576 for the baseline and 2,826,520 for the full model. Replaying each selected full-model checkpoint from raw EEG changed zero decisions relative to cached execution. [Protocol](results/protocol.json) · [Raw replay](results/raw_replay.json)

The validation split was repeatedly used to develop methods. Three recording-directory basenames appear in both manifest splits. When their windows are excluded, the full model scores 70.296 and 70.127, compared with 66.995 and 67.981 for baseline. Patient-disjoint generalization remains unverified. [Overlap sensitivity](results/fresh_both_overlap_sensitivity.json) · [Recording-fold sensitivity](results/fresh_both_recording_selection.json)

### Code and data

- [Baseline trainer and six-expert model](source/I2MOE6_F1), [strict baseline rerun](source/I2MOE/test333/strict_baseline_20261115), [full method and three refresh controls](source/I2MOE/test333/fair_refreshed_pairs_20261120).
- [Pair estimator](source/I2MOE/test333/fair_risk_pairs_20261120/pair_models.py), [temporal pooling](source/I2MOE/test333/fair_expert_allocation_20261120/allocation_models.py) and [raw EEG replay](source/I2MOE/test333/fair_refreshed_pairs_20261120/replay_raw.py).
- [All 200 epoch results](results/epochs.csv), [sampling hashes](results/sampling.csv), [initialization hashes](results/initialization.csv) and [conditional recording-bootstrap intervals](results/paired_bootstrap.json).
- [Original and released source digests](results/source_hashes.csv). The release preserves the experiment implementations and maps workstation-specific locations to configurable directories. The EEG files, per-window predictions, recordings, private manifest and checkpoint weights are held separately.

Run `python3 tools/verify_results.py` from a clone to reproduce the headline numbers and check the paired protocol using only the Python standard library. The manuscript has eight pages of main text in the official ICML 2026 preprint layout. The impact statement, references and appendix follow. This repository is a research release. [ICML author instructions](https://icml.cc/Conferences/2026/AuthorInstructions)
