#!/bin/bash
# Supplementary experiments:
#   1) WR-CP-style interval baseline (Sinkhorn OT weights over the
#      regularization grid) on the existing calibration/prediction artifacts.
#   2) IMPEL partial-observation LOCO transfers (protocol upper bound),
#      using IMPEL's released code and pretrained checkpoints.
# Run from the repository root. IMPEL_DIR must point to a clone of
# https://github.com/tongnie/IMPEL with pretrained source models.
set -eu
PY=${PYTHON:-python}
IMPEL_DIR=${IMPEL_DIR:-../IMPEL}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"

# ---- 1) WR-CP grid (requires: pip install POT) ----
$PY src/eval_wrcp.py --tag _v3 --regs 0.02 0.05 0.1 0.2

# ---- 2) IMPEL LOCO: all source->target partial-observation transfers ----
CITIES=(SH HZ CQ YT)
for TGT in "${CITIES[@]}"; do
  for SRC in "${CITIES[@]}"; do
    if [ "$SRC" = "$TGT" ]; then continue; fi
    LOGF="$LOG_DIR/impel_loco_${SRC}_${TGT}.log"
    if grep -q "Average Test MAE" "$LOGF" 2>/dev/null; then
      echo "skip IMPEL $SRC->$TGT (already done)"
      continue
    fi
    (cd "$IMPEL_DIR" && $PY experiments/impel/transfer_partial.py \
      --source_data "Delivery_${SRC}" \
      --target_data "Delivery_${TGT}" \
      --num_unknown_nodes 10 \
      --log_dir_pretrained "./logs/Delivery_${SRC}/impel/impel-32-1.0") \
      > "$LOGF" 2>&1
    echo "IMPEL $SRC->$TGT done"
  done
done

# ---- 3) summarize IMPEL LOCO logs into JSON ----
$PY src/summarize_impel_loco.py --log_dir "$LOG_DIR" --out results/impel_loco.json

echo "Supplementary experiments complete."
