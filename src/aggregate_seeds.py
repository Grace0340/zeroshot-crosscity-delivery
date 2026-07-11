"""Aggregate multi-seed results: mean +/- std of point accuracy and
static/online conformal coverage."""
import argparse
import json
import os

import numpy as np


def online_replay(P, Y, q0, alpha, burn_in=200):
    W = P.shape[0]
    cov_t = np.zeros(W)
    obs, n_obs = [], 0
    for t in range(W):
        if n_obs >= burn_in:
            s = np.concatenate(obs)
            q = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / len(s))))
        else:
            q = q0
        inside = (Y[t] >= P[t, :, :, 0] - q) & (Y[t] <= P[t, :, :, -1] + q)
        cov_t[t] = inside.mean()
        s_t = np.maximum(P[t, :, :, 0] - Y[t], Y[t] - P[t, :, :, -1])
        obs.append(s_t.reshape(-1)); n_obs += s_t.size
    return cov_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["sh", "hz", "cq", "yt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tag", default="", help="read tagged artifacts, e.g. _retr")
    args = ap.parse_args()

    table = {}
    for tgt in args.targets:
        rows = []
        for sd in args.seeds:
            sfx = ("" if sd == 0 else f"_s{sd}") + args.tag
            jf = f"results/zeroshot_{tgt}{sfx}.json"
            if not os.path.exists(jf):
                print(f"missing {jf}, skip"); continue
            r = json.load(open(jf))
            prd = np.load(f"results/preds_{tgt}{sfx}.npz")
            cal = np.load(f"results/calib_{tgt}{sfx}.npz")
            q_lvl = min(1.0, (1 - args.alpha) * (1 + 1 / len(cal["scores"])))
            q0 = float(np.quantile(cal["scores"], q_lvl))
            cov_t = online_replay(prd["pred_quantiles"], prd["y_true"], q0, args.alpha)
            wpd = 1  # one 24h window per day
            rows.append({
                "seed": sd, "mae": r["point"]["MAE"], "rmse": r["point"]["RMSE"],
                "cov_static": r["interval_naive"]["coverage"],
                "cov_day0": float(cov_t[:wpd].mean()),
                "cov_day1_3": float(cov_t[wpd:3 * wpd].mean()) if len(cov_t) > wpd else None,
                "cov_overall_online": float(cov_t.mean()),
            })
        if not rows:
            continue
        agg = {}
        for k in ["mae", "rmse", "cov_static", "cov_day0", "cov_day1_3", "cov_overall_online"]:
            v = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
            agg[k] = f"{v.mean():.3f}±{v.std():.3f}"
        table[tgt] = {"per_seed": rows, "agg": agg}
        print(f"{tgt}: " + "  ".join(f"{k}={v}" for k, v in agg.items()), flush=True)

    json.dump(table, open(f"results/seeds_summary{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
