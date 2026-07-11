"""Online interval-calibration replay on the target city: ACI vs. full
online conformal recomputation, starting from a zero-shot cold start.

Setup: day 0 uses the source-calibrated correction q0; afterwards, each time a
prediction window closes and the realized demand is observed, the correction is
updated with ACI (Gibbs & Candes 2021):
    q_{t+1} = q_t + gamma * (miss_rate_t - alpha)
or, for online CP, fully recomputed from all observed target scores. Updates
use only already-observed data (no future leakage) and never retrain the model.

Outputs per-window coverage trajectories + phase summaries
(day0 / days 1-3 / days 4-7 / day 8+).
"""
import argparse
import json

import numpy as np


def replay(P, Y, q0, alpha, gamma, per_region=True):
    """P [W,N,H,Q] quantile predictions in time order, Y [W,N,H].
    Returns per-window coverage [W] and width [W]."""
    W, N, H, _ = P.shape
    q = np.full(N if per_region else 1, float(q0) if np.isscalar(q0) else 0.0)
    if not np.isscalar(q0):
        q = q0.copy().astype(float)
    cov_t, wid_t = np.zeros(W), np.zeros(W)
    for t in range(W):
        qq = q if per_region else np.full(N, q[0])
        lo = P[t, :, :, 0] - qq[:, None]
        hi = P[t, :, :, -1] + qq[:, None]
        inside = (Y[t] >= lo) & (Y[t] <= hi)          # [N,H]
        cov_t[t] = inside.mean()
        wid_t[t] = (hi - lo).mean()
        miss = 1.0 - inside.mean(axis=1)              # per-region miss rate
        if per_region:
            q = q + gamma * (miss - alpha)
        else:
            q = q + gamma * (miss.mean() - alpha)
    return cov_t, wid_t


def replay_online(P, Y, q0, alpha, burn_in=200):
    """Online target calibration: q_t = conformal quantile of all observed
    target scores; the source q0 is kept until burn_in scores accumulate."""
    W, N, H, _ = P.shape
    cov_t, wid_t = np.zeros(W), np.zeros(W)
    obs, n_obs = [], 0
    for t in range(W):
        if n_obs >= burn_in:
            s = np.concatenate(obs)
            q = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / len(s))))
        else:
            q = q0
        lo = P[t, :, :, 0] - q
        hi = P[t, :, :, -1] + q
        inside = (Y[t] >= lo) & (Y[t] <= hi)
        cov_t[t] = inside.mean()
        wid_t[t] = (hi - lo).mean()
        s_t = np.maximum(P[t, :, :, 0] - Y[t], Y[t] - P[t, :, :, -1])
        obs.append(s_t.reshape(-1))
        n_obs += s_t.size
    return cov_t, wid_t


def stage_stats(cov_t, wid_t, windows_per_day):
    d = windows_per_day
    seg = {
        "day0": slice(0, d), "day1-3": slice(d, 3 * d),
        "day4-7": slice(3 * d, 7 * d), "day8+": slice(7 * d, None),
        "overall": slice(None),
    }
    return {k: {"cov": round(float(cov_t[s].mean()), 4), "width": round(float(wid_t[s].mean()), 2)}
            for k, s in seg.items() if len(cov_t[s]) > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["sh", "hz", "cq", "yt"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.02, 0.05, 0.1])
    ap.add_argument("--out_len", type=int, default=24)
    ap.add_argument("--tag", default="", help="read tagged artifacts, e.g. _v2")
    args = ap.parse_args()

    all_res = {}
    for tgt in args.targets:
        cal = np.load(f"results/calib_{tgt}{args.tag}.npz")
        prd = np.load(f"results/preds_{tgt}{args.tag}.npz")
        P, Y = prd["pred_quantiles"], prd["y_true"]
        q_lvl = min(1.0, (1 - args.alpha) * (1 + 1 / len(cal["scores"])))
        q0 = float(np.quantile(cal["scores"], q_lvl))
        wpd = max(1, 24 // args.out_len)

        res = {}
        cov_t, wid_t = replay(P, Y, q0, args.alpha, gamma=0.0)
        res["static(q0)"] = stage_stats(cov_t, wid_t, wpd)
        for g in args.gammas:
            cov_t, wid_t = replay(P, Y, q0, args.alpha, gamma=g)
            res[f"aci-g{g}"] = stage_stats(cov_t, wid_t, wpd)
            np.savez_compressed(f"results/aci_{tgt}_g{g}{args.tag}.npz", cov=cov_t, width=wid_t)
        cov_t, wid_t = replay_online(P, Y, q0, args.alpha)
        res["online-cp"] = stage_stats(cov_t, wid_t, wpd)
        np.savez_compressed(f"results/aci_{tgt}_online{args.tag}.npz", cov=cov_t, width=wid_t)
        all_res[tgt] = res
        print(f"== target {tgt} (windows={P.shape[0]}, q0={q0:.3f}) ==", flush=True)
        for k, v in res.items():
            line = " ".join(f"{st}:{d['cov']:.3f}/{d['width']}" for st, d in v.items())
            print(f"  {k:12s} {line}", flush=True)

    json.dump(all_res, open(f"results/aci_replay{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
