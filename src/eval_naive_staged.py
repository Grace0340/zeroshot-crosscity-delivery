# -*- coding: utf-8 -*-
"""Classical history-based baselines for the staged deployment phase (R1e).

persistence     : previous day's observed demand, same hours (available day 1+)
seasonal-naive  : observed demand 7 days earlier, same hour-of-week
                  (falls back to persistence before a full week exists)

Both use only already-observed target data over the same non-overlapping
24-hour windows as the staged replay, so their daily MAE curves are directly
comparable with staged_{tgt}_v3.npz.

Reads  results/npz/preds_{tgt}_v3.npz (y_true) and staged_{tgt}_v3.npz.
Writes results/naive_baselines_v3.json and results/npz/naive_{tgt}_v3.npz.
"""
import json

import numpy as np

CITIES = ["sh", "hz", "cq", "yt"]
TAG = "_v3"
RES = "results"
NPZ = "results/npz"
SEG = {"day1-3": slice(1, 3), "day4-7": slice(3, 7), "day8+": slice(7, None),
       "overall_d1+": slice(1, None)}


def main():
    out = {}
    for tgt in CITIES:
        Y = np.load(f"{NPZ}/preds_{tgt}{TAG}.npz")["y_true"]      # [W,N,H]
        W = Y.shape[0]
        mae_p, mae_s = np.full(W, np.nan), np.full(W, np.nan)
        for t in range(1, W):
            mae_p[t] = np.abs(Y[t - 1] - Y[t]).mean()
            ref = Y[t - 7] if t >= 7 else Y[t - 1]
            mae_s[t] = np.abs(ref - Y[t]).mean()

        staged = np.load(f"{NPZ}/staged_{tgt}{TAG}.npz")
        res = {}
        rows = [("persistence", mae_p), ("seasonal-naive", mae_s),
                ("model-staged", staged["model-staged"]),
                ("ha-target", staged["ha-target"]),
                ("ensemble", staged["ensemble"])]
        if "ensemble3" in staged:
            rows.append(("ensemble3", staged["ensemble3"]))
        for name, arr in rows:
            res[name] = {k: round(float(np.nanmean(arr[s])), 3) for k, s in SEG.items()}
        out[tgt] = res
        np.savez_compressed(f"{NPZ}/naive_{tgt}{TAG}.npz",
                            persistence=mae_p, seasonal_naive=mae_s)
        print(f"== {tgt} ==", flush=True)
        for name, v in res.items():
            print(f"  {name:14s} " + " ".join(f"{k}:{x}" for k, x in v.items()), flush=True)

    json.dump(out, open(f"{RES}/naive_baselines{TAG}.json", "w"), indent=2)
    print("saved", f"{RES}/naive_baselines{TAG}.json")


if __name__ == "__main__":
    main()
