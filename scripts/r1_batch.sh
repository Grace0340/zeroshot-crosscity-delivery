#!/bin/bash
# R1 supplement batch: 5-seed staged rerun + EMA sensitivity + Chronos baseline
set -x
cd /root/autodl-tmp/zeroshot/ours
LOG=/root/autodl-tmp/zeroshot/logs

python src/eval_staged.py --config configs/default.yaml --seeds 0 1 2 3 4 \
    --tag _v3 > $LOG/r1_staged5_v3.log 2>&1
echo "staged 5-seed done"

python src/eval_staged.py --config configs/default.yaml --seeds 0 1 2 3 4 \
    --ema_beta 0.5 --tag _v3 --out_tag _v3ema05 > $LOG/r1_staged_ema05.log 2>&1
python src/eval_staged.py --config configs/default.yaml --seeds 0 1 2 3 4 \
    --ema_beta 0.9 --tag _v3 --out_tag _v3ema09 > $LOG/r1_staged_ema09.log 2>&1
echo "ema sensitivity done"

pip install -q chronos-forecasting > $LOG/r1_chronos_install.log 2>&1
export HF_ENDPOINT=https://hf-mirror.com
python src/eval_chronos.py --targets sh hz cq yt --tag _v3 > $LOG/r1_chronos.log 2>&1
echo "chronos done"

touch $LOG/R1_BATCH_DONE
