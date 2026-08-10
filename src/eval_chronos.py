# -*- coding: utf-8 -*-
"""Foundation-model baseline (R1f): Chronos-Bolt zero-shot univariate
forecasts on the staged deployment timeline.

For every non-overlapping 24h window from day 1 onward, each region's
observed history up to the window start is fed to a pretrained Chronos-Bolt
model, which predicts the next 24 hours (median quantile). Day 0 is skipped:
with zero target observations a time-series foundation model has no context,
which is exactly the regime our zero-shot backbone addresses. Daily MAE
curves are aligned with staged_{tgt}{tag}.npz.

Usage:
  python src/eval_chronos.py --targets sh hz cq yt --tag _v3
Writes results/chronos_summary{tag}.json + results/chronos_{tgt}{tag}.npz
"""
import argparse
import json

import numpy as np
import torch
import yaml

from chronos import BaseChronosPipeline

MAX_CTX = 2048  # Chronos-Bolt context limit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--targets", nargs="+", default=["sh", "hz", "cq", "yt"])
    ap.add_argument("--model", default="amazon/chronos-bolt-base")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    in_len, out_len = cfg["data"]["input_len"], cfg["data"]["output_len"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = BaseChronosPipeline.from_pretrained(
        args.model, device_map=dev,
        torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)

    seg = {"day1-3": slice(1, 3), "day4-7": slice(3, 7), "day8+": slice(7, None),
           "overall_d1+": slice(1, None)}
    all_res = {}
    for tgt in args.targets:
        d = np.load(f"{cfg['data']['processed_dir']}/{tgt}.npz", allow_pickle=True)
        demand = d["demand"]                                   # [T, N]
        T, N = demand.shape
        widx = np.arange(0, T - in_len - out_len + 1)[::out_len]
        W = len(widx)
        mae = np.full(W, np.nan)
        # native Chronos-Bolt quantile heads; the 10-90% band is later
        # recalibrated to any nominal level by the online conformal layer
        q_levels = [0.1, 0.5, 0.9]
        P = np.full((W, N, out_len, len(q_levels)), np.nan, dtype=np.float32)
        Y = np.full((W, N, out_len), np.nan, dtype=np.float32)
        for k, i in enumerate(widx):
            start = i + in_len
            Y[k] = demand[start: start + out_len].T
            if k == 0:
                continue                                        # no target history yet
            ctx = demand[max(0, start - MAX_CTX): start]        # [T_ctx, N]
            batch = [torch.tensor(ctx[:, j], dtype=torch.float32) for j in range(N)]
            with torch.no_grad():
                q, _ = pipe.predict_quantiles(
                    context=batch, prediction_length=out_len, quantile_levels=q_levels)
            P[k] = q.float().cpu().numpy()                      # [N, H, Q]
            mae[k] = np.abs(P[k, :, :, 1] - Y[k]).mean()
            if k % 50 == 0:
                print(f"  {tgt} window {k}/{W} mae={mae[k]:.3f}", flush=True)
        res = {sk: round(float(np.nanmean(mae[sv])), 3) for sk, sv in seg.items()}
        all_res[tgt] = res
        np.savez_compressed(f"results/chronos_{tgt}{args.tag}.npz", mae=mae,
                            pred_quantiles=P, y_true=Y,
                            quantiles=np.array(q_levels))
        print(f"== {tgt} == {res}", flush=True)

    json.dump(all_res, open(f"results/chronos_summary{args.tag}.json", "w"), indent=2)
    print("saved", f"results/chronos_summary{args.tag}.json")


if __name__ == "__main__":
    main()
