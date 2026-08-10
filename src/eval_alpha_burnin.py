# -*- coding: utf-8 -*-
"""Sensitivity analyses for the online conformal stage (revision R1b/R1c).

1) Nominal-coverage sensitivity: replay the deployment at alpha in
   {0.05, 0.1, 0.2} (nominal 95/90/80%) for the static transfer, ACI, and
   online CP.  The source-calibrated correction q0 at the main setting is
   ~0 in all four cities (|q_hat| <= 0.003, see zeroshot_*_v3.json), so
   day-0 intervals use the raw quantile heads plus that correction; from
   day 1 onward each alpha is calibrated online on observed target scores.
2) Burn-in sensitivity: hold q0 for the first {1,2,3,5,7} observed days
   before switching to the online quantile (main setting switches after
   day 0, i.e. burn-in = 1 day).

Also reports the Winkler interval score at each alpha.
Reads  results/npz/preds_{tgt}_v3.npz and results/zeroshot_{tgt}_v3.json.
Writes results/alpha_burnin_v3.json.
"""
import json

import numpy as np

CITIES = ["sh", "hz", "cq", "yt"]
TAG = "_v3"
RES = "results"
NPZ = "results/npz"
ALPHAS = [0.05, 0.1, 0.2]
BURNIN_DAYS = [1, 2, 3, 5, 7]


def winkler(lo, hi, y, alpha):
    """Mean Winkler interval score."""
    w = (hi - lo) + (2.0 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    return float(w.mean())


def replay_static(P, Y, q0, alpha):
    lo = P[:, :, :, 0] - q0
    hi = P[:, :, :, -1] + q0
    cov_t = ((Y >= lo) & (Y <= hi)).reshape(P.shape[0], -1).mean(axis=1)
    wid_t = (hi - lo).reshape(P.shape[0], -1).mean(axis=1)
    return cov_t, wid_t, winkler(lo, hi, Y, alpha)


def replay_aci(P, Y, q0, alpha, gamma):
    """Per-region ACI, identical to src/aci_replay.py replay()."""
    W, N, H, _ = P.shape
    q = np.full(N, float(q0))
    cov_t, wid_t, wk = np.zeros(W), np.zeros(W), 0.0
    for t in range(W):
        lo = P[t, :, :, 0] - q[:, None]
        hi = P[t, :, :, -1] + q[:, None]
        inside = (Y[t] >= lo) & (Y[t] <= hi)
        cov_t[t] = inside.mean()
        wid_t[t] = (hi - lo).mean()
        wk += winkler(lo, hi, Y[t], alpha)
        miss = 1.0 - inside.mean(axis=1)
        q = q + gamma * (miss - alpha)
    return cov_t, wid_t, wk / W


def replay_online(P, Y, q0, alpha, burnin_days=1):
    """Online CP: full conformal quantile of all observed target scores,
    activated after `burnin_days` observed days (q0 before that)."""
    W = P.shape[0]
    cov_t, wid_t, wk = np.zeros(W), np.zeros(W), 0.0
    obs = []
    for t in range(W):
        if t >= burnin_days:
            s = np.concatenate(obs)
            q = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / len(s))))
        else:
            q = q0
        lo = P[t, :, :, 0] - q
        hi = P[t, :, :, -1] + q
        inside = (Y[t] >= lo) & (Y[t] <= hi)
        cov_t[t] = inside.mean()
        wid_t[t] = (hi - lo).mean()
        wk += winkler(lo, hi, Y[t], alpha)
        obs.append(np.maximum(P[t, :, :, 0] - Y[t], Y[t] - P[t, :, :, -1]).reshape(-1))
    return cov_t, wid_t, wk / W


def stage_stats(cov_t, wid_t):
    seg = {"day0": slice(0, 1), "day1-3": slice(1, 3), "day4-7": slice(3, 7),
           "day8+": slice(7, None), "overall": slice(None)}
    return {k: {"cov": round(float(cov_t[s].mean()), 4),
                "width": round(float(wid_t[s].mean()), 2)} for k, s in seg.items()}


def main():
    out = {}
    for tgt in CITIES:
        prd = np.load(f"{NPZ}/preds_{tgt}{TAG}.npz")
        P, Y = prd["pred_quantiles"], prd["y_true"]
        q_hat = json.load(open(f"{RES}/zeroshot_{tgt}{TAG}.json"))["interval_naive"]["q_hat"]

        res = {"q_hat_alpha0.1": q_hat, "alpha": {}, "burnin": {}}
        for a in ALPHAS:
            # source correction is only available at the main alpha; it is ~0
            # there, so other alphas start from the raw quantile heads
            q0 = q_hat if a == 0.1 else 0.0
            entry = {}
            cov, wid, wk = replay_static(P, Y, q0, a)
            entry["static"] = {**stage_stats(cov, wid), "winkler": round(wk, 2)}
            cov, wid, wk = replay_aci(P, Y, q0, a, gamma=0.05)
            entry["aci-g0.05"] = {**stage_stats(cov, wid), "winkler": round(wk, 2)}
            cov, wid, wk = replay_online(P, Y, q0, a)
            entry["online-cp"] = {**stage_stats(cov, wid), "winkler": round(wk, 2)}
            res["alpha"][f"{a:g}"] = entry

        for bd in BURNIN_DAYS:
            cov, wid, wk = replay_online(P, Y, q_hat, 0.1, burnin_days=bd)
            res["burnin"][str(bd)] = {**stage_stats(cov, wid), "winkler": round(wk, 2)}

        out[tgt] = res
        print(f"== {tgt} ==", flush=True)
        for a in ALPHAS:
            e = res["alpha"][f"{a:g}"]
            print(f"  alpha={a:<4} static ov={e['static']['overall']['cov']:.3f} "
                  f"online ov={e['online-cp']['overall']['cov']:.3f} "
                  f"d1-3={e['online-cp']['day1-3']['cov']:.3f} "
                  f"winkler={e['online-cp']['winkler']}", flush=True)
        for bd in BURNIN_DAYS:
            e = res["burnin"][str(bd)]
            print(f"  burnin={bd}d online ov={e['overall']['cov']:.3f} "
                  f"d1-3={e['day1-3']['cov']:.3f}", flush=True)

    json.dump(out, open(f"{RES}/alpha_burnin{TAG}.json", "w"), indent=2)
    print("saved", f"{RES}/alpha_burnin{TAG}.json")


if __name__ == "__main__":
    main()
