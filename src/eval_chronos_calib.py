# -*- coding: utf-8 -*-
"""Model-agnostic calibration demo: apply the online conformal layer to
Chronos-Bolt's native quantile band (10-90%).

Shows that the paper's calibration stage is not tied to our backbone: the
same online CP recomputation restores nominal 90% coverage on top of a
pretrained foundation forecaster. Day 0 has no Chronos forecast (no target
history); the replay starts at day 1 with q0 = 0 (raw band) and switches to
the online conformal quantile once one day of scores is observed.

Reads  results/npz/chronos_{tgt}_v3.npz (pred_quantiles, y_true).
Writes results/chronos_calib_v3.json.
"""
import json

import numpy as np

CITIES = ["sh", "hz", "cq", "yt"]
TAG = "_v3"
RES = "results"
NPZ = "results/npz"
ALPHA = 0.1


def stage_stats(cov_t, wid_t):
    seg = {"day1-3": slice(1, 3), "day4-7": slice(3, 7), "day8+": slice(7, None),
           "overall_d1+": slice(1, None)}
    return {k: {"cov": round(float(np.nanmean(cov_t[s])), 4),
                "width": round(float(np.nanmean(wid_t[s])), 2)} for k, s in seg.items()}


def main():
    out = {}
    for tgt in CITIES:
        z = np.load(f"{NPZ}/chronos_{tgt}{TAG}.npz")
        P, Y = z["pred_quantiles"], z["y_true"]      # [W,N,H,3], [W,N,H]; k=0 is nan
        W = P.shape[0]
        cov_raw, wid_raw = np.full(W, np.nan), np.full(W, np.nan)
        cov_cp, wid_cp = np.full(W, np.nan), np.full(W, np.nan)
        obs, q = [], 0.0
        for t in range(1, W):
            lo_r, hi_r = P[t, :, :, 0], P[t, :, :, 2]
            cov_raw[t] = ((Y[t] >= lo_r) & (Y[t] <= hi_r)).mean()
            wid_raw[t] = (hi_r - lo_r).mean()
            lo, hi = lo_r - q, hi_r + q
            cov_cp[t] = ((Y[t] >= lo) & (Y[t] <= hi)).mean()
            wid_cp[t] = (hi - lo).mean()
            obs.append(np.maximum(lo_r - Y[t], Y[t] - hi_r).reshape(-1))
            s = np.concatenate(obs)
            q = np.quantile(s, min(1.0, (1 - ALPHA) * (1 + 1 / len(s))))
        out[tgt] = {"raw-band": stage_stats(cov_raw, wid_raw),
                    "online-cp": stage_stats(cov_cp, wid_cp)}
        print(f"== {tgt} ==", flush=True)
        for k, v in out[tgt].items():
            line = " ".join(f"{st}:{d['cov']:.3f}/{d['width']}" for st, d in v.items())
            print(f"  {k:10s} {line}", flush=True)

    json.dump(out, open(f"{RES}/chronos_calib{TAG}.json", "w"), indent=2)
    print("saved", f"{RES}/chronos_calib{TAG}.json")


if __name__ == "__main__":
    main()
