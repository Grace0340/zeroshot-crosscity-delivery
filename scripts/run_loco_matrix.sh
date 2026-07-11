#!/bin/bash
# Full leave-one-city-out matrix with the adopted training recipe
# (lr 5e-4, gradient clipping 1.0, seeds 0-4), followed by every evaluation
# used in the paper. Run from the repository root after preparing the data
# (see data/README.md) and setting paths in configs/default.yaml.
#
# The _v3 tag identifies the adopted recipe; the shipped results/ JSONs and
# tools/make_{tables,figures}.py use the same tag.
set -eu
PY=${PYTHON:-python}
TAG=_v3

# ---- 0) build the hourly demand tensors from raw LaDe CSVs ----
$PY src/data/build_dataset.py --config configs/default.yaml
$PY scripts/check_align.py

# ---- 1) train the backbone: 4 targets x 5 seeds ----
for SEED in 0 1 2 3 4; do
  for TGT in sh hz cq yt; do
    $PY src/train_zeroshot.py --config configs/default.yaml \
       --target "$TGT" --seed "$SEED" --lr 0.0005 --clip 1.0 --tag "$TAG"
  done
done

# ---- 2) zero-shot baselines (climatology, kNN profiles) ----
$PY src/baselines.py

# ---- 3) evaluations ----
$PY src/aggregate_seeds.py --tag "$TAG" --seeds 0 1 2 3 4   # multi-seed summary
$PY src/eval_staged.py --tag "$TAG" --seeds 0 1 2 3 4       # staged replay + ensemble
$PY src/decision_replay.py --tag "$TAG"                     # newsvendor replay
$PY src/eval_interval_efficiency.py --tag "$TAG"            # interval sharpness
$PY src/aci_replay.py --tag "$TAG"                          # ACI vs online CP
$PY src/analyze_calibration.py --tag "$TAG"                 # weighted-conformal grid

echo "LOCO matrix complete; JSON summaries are in results/"
