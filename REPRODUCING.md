# Reproducing the comparison

## Verify the numbers from a clean checkout

```bash
python3 tools/verify_results.py
```

This recalculates F1 from the four confusion counts, verifies checkpoint selection over all 20 epochs, checks the sample-order hashes, compares initialization of shared parameters and checks the released source hashes. It uses only aggregate results. Scores in the CSVs are fractions. The manuscript reports percentages, and differences are in percentage points. AUPRC in the CSVs is scikit-learn average precision. Paired bootstrap intervals condition on the selected epochs and thresholds.

## Run the code tests

The experiment used CPython 3.10.20, PyTorch 2.4.0 and torchvision 0.19.0 with CUDA 12.1 on Linux. The tests below use CPU-sized inputs. The `round3` and `round4*` directories contain original imports required by the strict baseline wrapper. Launch different experiment entry points in separate processes because each selects its own `bootstrap` module.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHON_BIN=python ./tools/test.sh
```

## Refit with the study inputs

Training requires access to the original manifest, EEG arrays and five distilled regional source checkpoints. The manifest provides `file_path`, binary `target`, and `mode` (`Train` or `Val`). Place the waveform tree and its original CSV under `$CZB_STORAGE/finetune-xiehe-dataset`. The loader's expected CSV filename is in [data.py](source/I2MOE6_F1/i2moe_eeg/data.py). The five relative checkpoint paths and expected SHA-256 digests are in [protocol.json](results/protocol.json). Keep the manifest's directory structure because it defines training acquisition groups and recording proxies.

Run these commands from the repository root. Set `CZB_STORED_PREFIX` to the common input path prefix already recorded in your authorized copy of the CSV. The storage and run roots can be any writable absolute directories with room for the private artifacts. The default multi-GPU cache preparation scripts use the study's original GPU assignments.

```bash
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
export CZB_STORAGE="$PWD/private-data"
export CZB_RUNS="$PWD/private-runs"
export CZB_STORED_PREFIX="/replace/with/prefix/in/manifest"
mkdir -p "$CZB_STORAGE" "$CZB_RUNS"
BASE=source/I2MOE6_F1
EXP=source/I2MOE/test333
ORIGINAL="$CZB_STORAGE/I2MOE6_F1/screen_20261112/baseline"
CONTROL="$CZB_RUNS/I2MOE_test333_strict_baseline_20261115"
STUDY="$CZB_RUNS/I2MOE_test333_fair_refreshed_pairs_20261120"
```

First generate the local reference config and checkpoint used for the trainer's formal configuration guard. The wrappers read the reference config from `best_f1.pt`. The shared trainable parameters of the paired runs start afresh. Pass the threshold options explicitly because this experiment used a finer grid than the trainer's CLI defaults.

```bash
for SEED in 3 42; do
  CUDA_VISIBLE_DEVICES=0 python "$BASE/train_eeg19_i2moe.py" \
    --variant baseline --seed "$SEED" --ratio 30 --batch-size 128 --eval-batch-size 128 \
    --epochs 20 --samples-per-epoch 20000 --lr 0.0005 --num-workers 2 --memory-cap-gib 4 \
    --threshold-start 0.01 --threshold-end 0.99 --threshold-step 0.01 \
    --output-dir "$ORIGINAL/seed$SEED"
  CUDA_VISIBLE_DEVICES=0 python "$EXP/strict_baseline_20261115/train_strict.py" \
    --method baseline --seed "$SEED" --mode formal --output-dir "$CONTROL/formal/baseline/seed$SEED"
done
```

Prepare the original FP32 execution caches. These include separate packets for native partial batches. The original block-cache coordinator waits for the depth-cache job to complete. Existing cache plans contain absolute provenance paths and source hashes, so creating caches from this checkout gives them consistent local identities.

```bash
python "$EXP/fair_cache_20261115/prepare_cache.py"
python "$EXP/fair_moment_cache_20261115/prepare_cache.py"
python "$EXP/fair_raw_cache_20261115/prepare_cache.py"
python "$EXP/fair_time_scores_20261119/prepare.py"
python "$EXP/fair_depth_cache_20261120/prepare.py"
python "$EXP/fair_temporal_blocks_20261120/prepare.py"
```

Run the four learning variants and replay the selected full-model checkpoints from native EEG. Each training process expects one visible CUDA device. The loop shows a serial execution order and its jobs can be assigned to separate GPUs.

```bash
for SEED in 3 42; do
  for METHOD in collection fresh_cases fresh_controls fresh_both; do
    CUDA_VISIBLE_DEVICES=0 python "$EXP/fair_refreshed_pairs_20261120/train_strict.py" \
      --method "$METHOD" --seed "$SEED" --mode formal --output-dir "$STUDY/formal/$METHOD/seed$SEED"
  done
  CUDA_VISIBLE_DEVICES=0 python "$EXP/fair_refreshed_pairs_20261120/replay_raw.py" \
    --run "$STUDY/formal/fresh_both/seed$SEED"
done
python "$EXP/fair_analysis_20261115/check_overlap_sensitivity.py" "$STUDY" fresh_both
python "$EXP/fair_analysis_20261115/check_recording_selection.py" "$STUDY" fresh_both
```

The original comparison used nondeterministic GPU kernels. Shared initialization and identical sampled indices and batch order give a paired training comparison. A new fit may yield different probabilities and F1. The raw replay records describe the archived selected models. The private training inputs and weights are required to repeat that check.

## Recompute the operating-point analysis

The public [threshold counts](results/selected_operating_points.json) contain the TP and FP counts at all 99 original grid thresholds for the selected baseline and full-model checkpoints. The same file gives the number of windows flagged by only one model or both, broken down by true label. For the budget comparison use the first full-model grid threshold with no more false positives than the baseline at its F1-selected threshold. To regenerate the file from authorized prediction archives after the runs above, use the following command. Its output contains only aggregate counts.

```bash
python tools/operating_points.py --baseline "$CONTROL/formal/baseline" \
  --candidate "$STUDY/formal/fresh_both" --output reproduced_operating_points.json
```

## Build the paper

The paper uses the official ICML 2026 LaTeX style in preprint mode. The compiled PDF contains eight pages of main text. Both figures and their aggregate source data are included. Build with a TeX Live installation and BibTeX, or with Tectonic.

```bash
cd paper
latexmk -pdf main.tex
# Alternative
tectonic main.tex
```

To regenerate both figures from the epoch CSV and operating-point JSON, install matplotlib 3.10.9 in a plotting environment and run `python tools/figure.py` from the repository root. The conference supplies the style, bibliography style and companion files. [Official ICML 2026 style archive](https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip)
