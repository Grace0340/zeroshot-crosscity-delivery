"""Staged cold-start replay: zero-shot at day 0, rolling target history from
day 1 onward.

Compares daily MAE curves over the same non-overlapping 24h windows:
  model-staged : our model (zero-shot path on day 0, history path afterwards)
                 + online CP intervals
  ha-source    : source hour-of-week median curve (static, strongest zero-shot
                 baseline)
  ha-target    : accumulating target climatology (hour-of-week median of
                 observed data only; falls back to ha-source for unseen cells)
  ensemble     : inverse-error online ensemble of model and ha-target
Writes results/staged_summary{tag}.json (daily MAE + interval coverage/width).
"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.backbone import ZeroShotSTBackbone
from src.train_zeroshot import MAIN_CITIES, calendar_feat, load_city, make_windows

HOW = 168


def how_arr(ts):
    return np.array([dt.datetime.fromisoformat(str(s)).weekday() * 24
                     + dt.datetime.fromisoformat(str(s)).hour for s in ts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--targets", nargs="+", default=["sh", "hz", "cq", "yt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tag", default="", help="checkpoint tag, reads ckpt_{tgt}{seed}{tag}.pt")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    in_len, out_len = cfg["data"]["input_len"], cfg["data"]["output_len"]
    Q = cfg["model"]["quantiles"]
    imed, ilo, ihi = Q.index(0.5), 0, len(Q) - 1

    data = {c: load_city(cfg["data"]["processed_dir"], cfg["model"]["llm_emb_path"], c)
            for c in MAIN_CITIES}
    hows = {c: how_arr(data[c][3]) for c in MAIN_CITIES}

    all_res = {}
    for tgt in args.targets:
        sources = [c for c in MAIN_CITIES if c != tgt]
        # source HA median curve (count space)
        pool = [[] for _ in range(HOW)]
        for c in sources:
            d, how = data[c][0], hows[c]
            for hw in range(HOW):
                pool[hw].append(d[how == hw].reshape(-1))
        ha_src = np.array([np.median(np.concatenate(pool[hw])) for hw in range(HOW)])

        demand, static, emb, ts = data[tgt]
        how_t = hows[tgt]
        y_log = torch.tensor(np.log1p(demand), dtype=torch.float32, device=dev)
        static_t = torch.tensor(static, device=dev)
        emb_t = torch.tensor(emb, device=dev)
        widx = make_windows(demand.shape[0], in_len, out_len)[::out_len]
        N = demand.shape[1]

        seed_daily = {"model-staged": [], "ha-source": [], "ha-target": [], "ensemble": []}
        seed_cov, seed_wid, seed_cov_e, seed_wid_e = [], [], [], []
        for sd in args.seeds:
            sfx = ("" if sd == 0 else f"_s{sd}") + args.tag
            ck = f"results/ckpt_{tgt}{sfx}.pt"
            model = ZeroShotSTBackbone(
                static_dim_in=static.shape[1], llm_dim=cfg["model"]["llm_emb_dim"],
                hidden=cfg["model"]["hidden_dim"], out_len=out_len, quantiles=Q,
                tcn_layers=cfg["model"]["tcn_layers"], knn_k=cfg["model"]["knn_k"],
            ).to(dev)
            # strict=False: checkpoints trained without the optional retrieval
            # branch lack retr_enc keys; that branch is unused at inference here.
            model.load_state_dict(torch.load(ck, map_location=dev), strict=False)
            model.eval()

            mae_m, mae_s, mae_t, mae_e, cov_t_list, wid_t_list = [], [], [], [], [], []
            cov_e_list, wid_e_list = [], []
            obs_scores, n_obs, qhat = [], 0, 0.0
            obs_e, n_e, qhat_e = [], 0, 0.0
            ema_m, ema_h, ema_beta = None, None, 0.7   # EMA of each predictor's recent MAE
            for k, i in enumerate(widx):
                sl = slice(i + in_len, i + in_len + out_len)
                y_true = demand[sl].T                                   # [N,H]
                hw = how_t[sl]
                cal = torch.tensor(calendar_feat(ts[i + in_len])[None], device=dev)
                with torch.no_grad():
                    if k == 0:
                        pred = model(static_t, emb_t, cal, history=None)
                    else:
                        hist = y_log[i: i + in_len].T[None]             # [1,N,T]
                        pred = model(static_t, emb_t, cal, history=hist)
                p = torch.expm1(pred[0]).cpu().numpy()                  # [N,H,Q]
                mae_m.append(np.abs(p[..., imed] - y_true).mean())
                mae_s.append(np.abs(ha_src[hw][None] - y_true).mean())
                # accumulating target climatology (observations before i+in_len only)
                seen = demand[: i + in_len]
                seen_how = how_t[: i + in_len]
                ha_tgt = np.array([
                    np.median(seen[seen_how == h], axis=0) if (seen_how == h).any() else
                    np.full(N, ha_src[h]) for h in hw])                 # [H,N]
                mae_t.append(np.abs(ha_tgt.T - y_true).mean())
                # online ensemble: inverse-EMA-error blend of model and ha-target
                # (uses only already-observed errors)
                if ema_m is None:
                    blend = ha_tgt.T
                else:
                    wm, wh = 1.0 / (ema_m + 1e-6), 1.0 / (ema_h + 1e-6)
                    blend = (wm * p[..., imed] + wh * ha_tgt.T) / (wm + wh)
                mae_e.append(np.abs(blend - y_true).mean())
                # online CP intervals (symmetric, city-wide): model interval +
                # ensemble interval (model spread re-centered on the blend)
                lo, hi = p[..., ilo] - qhat, p[..., ihi] + qhat
                cov_t_list.append(((y_true >= lo) & (y_true <= hi)).mean())
                wid_t_list.append((hi - lo).mean())
                shift = blend - p[..., imed]
                lo_e, hi_e = p[..., ilo] + shift - qhat_e, p[..., ihi] + shift + qhat_e
                cov_e_list.append(((y_true >= lo_e) & (y_true <= hi_e)).mean())
                wid_e_list.append((hi_e - lo_e).mean())
                s = np.maximum(p[..., ilo] - y_true, y_true - p[..., ihi])
                obs_scores.append(s.reshape(-1)); n_obs += s.size
                if n_obs >= 200:
                    ss = np.concatenate(obs_scores)
                    qhat = np.quantile(ss, min(1.0, (1 - args.alpha) * (1 + 1 / len(ss))))
                se = np.maximum((p[..., ilo] + shift) - y_true, y_true - (p[..., ihi] + shift))
                obs_e.append(se.reshape(-1)); n_e += se.size
                if n_e >= 200:
                    ss = np.concatenate(obs_e)
                    qhat_e = np.quantile(ss, min(1.0, (1 - args.alpha) * (1 + 1 / len(ss))))
                # update EMAs with the just-observed window errors
                m_err = np.abs(p[..., imed] - y_true).mean()
                h_err = np.abs(ha_tgt.T - y_true).mean()
                ema_m = m_err if ema_m is None else ema_beta * ema_m + (1 - ema_beta) * m_err
                ema_h = h_err if ema_h is None else ema_beta * ema_h + (1 - ema_beta) * h_err
            seed_daily["model-staged"].append(mae_m)
            seed_daily["ha-source"].append(mae_s)
            seed_daily["ha-target"].append(mae_t)
            seed_daily["ensemble"].append(mae_e)
            seed_cov.append(cov_t_list); seed_wid.append(wid_t_list)
            seed_cov_e.append(cov_e_list); seed_wid_e.append(wid_e_list)

        daily = {k: np.mean(v, axis=0) for k, v in seed_daily.items()}
        cov = np.mean(seed_cov, axis=0); wid = np.mean(seed_wid, axis=0)
        cov_e = np.mean(seed_cov_e, axis=0); wid_e = np.mean(seed_wid_e, axis=0)
        seg = {"day0": slice(0, 1), "day1-3": slice(1, 3), "day4-7": slice(3, 7), "day8+": slice(7, None)}
        res = {}
        for name, arr in daily.items():
            res[name] = {sk: round(float(arr[sv].mean()), 3) for sk, sv in seg.items()}
        res["interval"] = {sk: {"cov": round(float(cov[sv].mean()), 4),
                                "width": round(float(wid[sv].mean()), 2)} for sk, sv in seg.items()}
        res["interval_ens"] = {sk: {"cov": round(float(cov_e[sv].mean()), 4),
                                    "width": round(float(wid_e[sv].mean()), 2)} for sk, sv in seg.items()}
        all_res[tgt] = res
        print(f"== {tgt} ==", flush=True)
        for k, v in res.items():
            print(f"  {k:12s} {v}", flush=True)
        np.savez_compressed(f"results/staged_{tgt}{args.tag}.npz",
                            **{k: np.array(v) for k, v in daily.items()}, cov=cov, width=wid)

    json.dump(all_res, open(f"results/staged_summary{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
