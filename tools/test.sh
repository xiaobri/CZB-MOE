#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
for VARIANT in I2MOE6_F1 I2MOE/test333/{strict_baseline_20261115,fair_risk_pairs_20261120,fair_refreshed_pairs_20261120}; do
    (cd "$ROOT/source/$VARIANT" && "${PYTHON_BIN:-python3}" -m unittest discover -s tests -p 'test_*.py' -v)
done
