"""Newsvendor capacity-provisioning replay on the zero-shot target city
(no future-information leakage).

For each cost ratio (co, cu), the critical quantile is kappa = cu/(co+cu).
Three strategies:
  point   : capacity = median forecast (common operational practice)
  raw-q   : capacity = raw model quantile at level kappa (linear interpolation
            between discrete quantile heads)
  onl-q   : raw-q + online residual correction (shift by the empirical kappa
            quantile of already-observed residuals; no correction on day 0)
Cost = co * overage + cu * underage; savings are reported relative to point.
"""
import argparse
import json

import numpy as np


def interp_quantile(P, qs, kappa):
    """P [W,N,H,Q] -> capacity [W,N,H], linear interpolation between quantiles."""
    qs = np.asarray(qs)
    if kappa <= qs[0]:
        return P[..., 0]
    if kappa >= qs[-1]:
        return P[..., -1]
    i = np.searchsorted(qs, kappa) - 1
    w = (kappa - qs[i]) / (qs[i + 1] - qs[i])
    return (1 - w) * P[..., i] + w * P[..., i + 1]


def cost(c, y, co, cu):
    return co * np.maximum(c - y, 0) + cu * np.maximum(y - c, 0)


def online_correct(c_raw, Y, kappa, per_region=True):
    """Windowed replay: c_t = c_raw_t + empirical kappa quantile of observed
    residuals (per region by default)."""
    W, N, H = c_raw.shape
    c_adj = c_raw.copy()
    if per_region:
        res = [[] for _ in range(N)]
        for t in range(W):
            for j in range(N):
                if res[j]:
                    c_adj[t, j] = c_raw[t, j] + np.quantile(np.concatenate(res[j]), kappa)
                res[j].append(Y[t, j] - c_raw[t, j])
    else:
        res = []
        for t in range(W):
            if res:
                c_adj[t] = c_raw[t] + np.quantile(np.concatenate(res), kappa)
            res.append((Y[t] - c_raw[t]).reshape(-1))
    return c_adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["sh", "hz", "cq", "yt"])
    ap.add_argument("--cost_ratios", default="1:1,1:3,3:1")
    ap.add_argument("--tag", default="", help="read preds_{tgt}{tag}.npz, e.g. _retr")
    args = ap.parse_args()
    ratios = [tuple(map(float, r.split(":"))) for r in args.cost_ratios.split(",")]

    all_res = {}
    for tgt in args.targets:
        prd = np.load(f"results/preds_{tgt}{args.tag}.npz")
        P, Y, qs = prd["pred_quantiles"], prd["y_true"], list(prd["quantiles"])
        med = P[..., qs.index(0.5)]
        res = {}
        for co, cu in ratios:
            kappa = cu / (co + cu)
            c_raw = interp_quantile(P, qs, kappa)
            c_onl = online_correct(c_raw, Y, kappa)
            cp = float(cost(med, Y, co, cu).mean())
            cr = float(cost(c_raw, Y, co, cu).mean())
            co_ = float(cost(np.maximum(c_onl, 0), Y, co, cu).mean())
            res[f"co{co:g}_cu{cu:g}"] = {
                "cost_point": round(cp, 3), "cost_rawq": round(cr, 3), "cost_onlineq": round(co_, 3),
                "save_rawq_pct": round(100 * (cp - cr) / cp, 2),
                "save_onlineq_pct": round(100 * (cp - co_) / cp, 2),
            }
        all_res[tgt] = res
        print(f"== target {tgt} ==", flush=True)
        for k, v in res.items():
            print(f"  {k:12s} point={v['cost_point']} rawq={v['cost_rawq']} ({v['save_rawq_pct']}%) "
                  f"onlineq={v['cost_onlineq']} ({v['save_onlineq_pct']}%)", flush=True)

    json.dump(all_res, open(f"results/decision_replay{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
