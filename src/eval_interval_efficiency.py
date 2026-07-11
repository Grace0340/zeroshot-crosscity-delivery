"""Interval efficiency comparison + point-forecast blending (offline).

Answers two questions:
1) Under the same online conformal procedure, are the model's quantile
   intervals sharper than empirical HA intervals at comparable coverage?
   - ours   : model [q05,q95] + symmetric online correction (same protocol
              as aci_replay.py)
   - ha-emp : per-hour-of-week empirical quantile intervals from pooled
              source samples + the same online correction
2) Does blending pred = a*HA + (1-a)*model_median improve point accuracy
   (grid over a)?
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_zeroshot import MAIN_CITIES, load_city, make_windows

HOW = 168


def how_arr(ts):
    return np.array([dt.datetime.fromisoformat(str(s)).weekday() * 24
                     + dt.datetime.fromisoformat(str(s)).hour for s in ts])


def online_replay(lo_raw, hi_raw, Y, alpha=0.1, burn_in=200, per_region=False):
    """Symmetric online conformal replay; returns per-window coverage / width /
    Winkler score (day 0 uses q=0, i.e. the raw interval)."""
    W, N, H = lo_raw.shape
    cov_t, wid_t, wink_t = np.zeros(W), np.zeros(W), np.zeros(W)
    if per_region:
        obs = [[] for _ in range(N)]
        q = np.zeros(N)
    else:
        obs, n_obs, q = [], 0, 0.0
    cov_r = np.zeros((W, N)); wid_r = np.zeros((W, N))
    for t in range(W):
        if per_region:
            qq = q[:, None]
        else:
            qq = q
        lo, hi = lo_raw[t] - qq, hi_raw[t] + qq
        inside = (Y[t] >= lo) & (Y[t] <= hi)
        cov_t[t] = inside.mean(); wid_t[t] = (hi - lo).mean()
        cov_r[t] = inside.mean(axis=1); wid_r[t] = (hi - lo).mean(axis=1)
        wink_t[t] = ((hi - lo) + (2 / alpha) * np.maximum(lo - Y[t], 0)
                     + (2 / alpha) * np.maximum(Y[t] - hi, 0)).mean()
        s_t = np.maximum(lo_raw[t] - Y[t], Y[t] - hi_raw[t])
        if per_region:
            for j in range(N):
                obs[j].append(s_t[j])
                s = np.concatenate(obs[j])
                if len(s) >= burn_in // 4:
                    q[j] = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / len(s))))
        else:
            obs.append(s_t.reshape(-1)); n_obs += s_t.size
            if n_obs >= burn_in:
                s = np.concatenate(obs)
                q = np.quantile(s, min(1.0, (1 - alpha) * (1 + 1 / len(s))))
    return cov_t, wid_t, wink_t, cov_r, wid_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="", help="model preds tag, e.g. _retr2")
    ap.add_argument("--alphas_blend", nargs="+", type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    in_len, out_len = cfg["data"]["input_len"], cfg["data"]["output_len"]

    data = {c: load_city(cfg["data"]["processed_dir"], cfg["model"]["llm_emb_path"], c)
            for c in MAIN_CITIES}
    hows = {c: how_arr(data[c][3]) for c in MAIN_CITIES}

    all_res = {}
    for tgt in MAIN_CITIES:
        sources = [c for c in MAIN_CITIES if c != tgt]
        # empirical HA quantile curves: pool all source (region, occurrence)
        # count samples per hour-of-week
        samples = [[] for _ in range(HOW)]
        for c in sources:
            d, how = data[c][0], hows[c]
            for hw in range(HOW):
                samples[hw].append(d[how == hw].reshape(-1))
        ha_q = np.zeros((HOW, 3))
        for hw in range(HOW):
            s = np.concatenate(samples[hw])
            ha_q[hw] = np.quantile(s, [0.05, 0.5, 0.95])

        prd = np.load(f"results/preds_{tgt}{args.tag}.npz")
        P, Y = prd["pred_quantiles"], prd["y_true"]
        qs = list(prd["quantiles"])
        med = P[..., qs.index(0.5)]
        demand_t = data[tgt][0]
        widx = make_windows(demand_t.shape[0], in_len, out_len)[::out_len]
        HW = np.stack([hows[tgt][i + in_len: i + in_len + out_len] for i in widx])   # [W,H]
        assert HW.shape[0] == Y.shape[0], f"{tgt}: window mismatch {HW.shape} vs {Y.shape}"

        N = Y.shape[1]
        ha_med = ha_q[HW, 1][:, None, :].repeat(N, axis=1)
        ha_lo = ha_q[HW, 0][:, None, :].repeat(N, axis=1)
        ha_hi = ha_q[HW, 2][:, None, :].repeat(N, axis=1)

        res = {"blend_mae": {}}
        for a in args.alphas_blend:
            blend = a * ha_med + (1 - a) * med
            res["blend_mae"][f"a{a:g}"] = round(float(np.abs(blend - Y).mean()), 4)

        for name, lo, hi in [("ours", P[..., 0], P[..., -1]), ("ha-emp", ha_lo, ha_hi)]:
            for pr in [False, True]:
                cov_t, wid_t, wink_t, cov_r, wid_r = online_replay(lo, hi, Y, per_region=pr)
                res[f"interval_{name}" + ("_pr" if pr else "")] = {
                    "cov_day1_3": round(float(cov_t[1:3].mean()), 4),
                    "cov_overall": round(float(cov_t.mean()), 4),
                    "cov_min_region": round(float(cov_r.mean(axis=0).min()), 4),
                    "width_overall": round(float(wid_t.mean()), 2),
                    "width_day8p": round(float(wid_t[7:].mean()), 2),
                    "winkler_overall": round(float(wink_t.mean()), 2),
                }
        all_res[tgt] = res
        print(f"== {tgt} ==", flush=True)
        print("  blend:", res["blend_mae"], flush=True)
        for k in ["interval_ours", "interval_ours_pr", "interval_ha-emp", "interval_ha-emp_pr"]:
            print(f"  {k}: {res[k]}", flush=True)

    json.dump(all_res, open(f"results/interval_efficiency{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
