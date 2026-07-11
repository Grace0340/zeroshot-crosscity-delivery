"""Zero-shot baselines, evaluated on the same non-overlapping target windows
as train_zeroshot.py:

  ha-global : hour-of-week mean demand pooled over all source regions
              (one shared curve for every target region)
  knn-prof  : per target region, similarity-weighted average of the
              hour-of-week profiles of its top-k most LLM-similar source regions
Neither baseline uses any target-city observation. Profiles are averaged in
count space.
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


def profile(demand, how):
    """[T,N] -> [168,N] hour-of-week mean in count space."""
    p = np.zeros((HOW, demand.shape[1]), dtype=np.float64)
    for hw in range(HOW):
        m = how == hw
        p[hw] = demand[m].mean(axis=0) if m.any() else demand.mean(axis=0)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--knn_k", type=int, default=8)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    in_len, out_len = cfg["data"]["input_len"], cfg["data"]["output_len"]

    data = {c: load_city(cfg["data"]["processed_dir"], cfg["model"]["llm_emb_path"], c)
            for c in MAIN_CITIES}
    hows = {c: how_arr(data[c][3]) for c in MAIN_CITIES}

    all_res = {}
    for tgt in MAIN_CITIES:
        sources = [c for c in MAIN_CITIES if c != tgt]
        profs, embs = [], []
        for c in sources:
            profs.append(profile(data[c][0], hows[c]))     # [168,Nc]
            embs.append(data[c][2])
        bank_prof = np.concatenate(profs, axis=1)          # [168,M]
        bank_emb = np.concatenate(embs)
        bank_n = bank_emb / np.linalg.norm(bank_emb, axis=1, keepdims=True)

        demand_t, _, emb_t, ts_t = data[tgt]
        how_t = hows[tgt]
        widx = make_windows(demand_t.shape[0], in_len, out_len)[::out_len]
        Y = np.stack([demand_t[i + in_len: i + in_len + out_len].T for i in widx])   # [W,N,H]
        HW = np.stack([how_t[i + in_len: i + in_len + out_len] for i in widx])       # [W,H]

        # ha-global: mean over all source regions
        ha_curve = bank_prof.mean(axis=1)                  # [168]
        P_ha = ha_curve[HW][:, None, :].repeat(Y.shape[1], axis=1)

        # knn-prof: per-target-region retrieval
        S = (emb_t / np.linalg.norm(emb_t, axis=1, keepdims=True)) @ bank_n.T
        idx = np.argsort(-S, axis=1)[:, : args.knn_k]
        s_top = np.take_along_axis(S, idx, axis=1)
        z = (s_top - s_top.mean(axis=1, keepdims=True)) / (s_top.std(axis=1, keepdims=True) + 1e-8)
        w = np.exp(z); w /= w.sum(axis=1, keepdims=True)
        knn_prof = (bank_prof.T[idx] * w[..., None]).sum(axis=1)  # [N,168]
        P_knn = knn_prof[:, HW].transpose(1, 0, 2)          # [W,N,H]

        res = {}
        for name, P in [("ha-global", P_ha), ("knn-prof", P_knn)]:
            res[name] = {"MAE": round(float(np.abs(P - Y).mean()), 4),
                         "RMSE": round(float(np.sqrt(((P - Y) ** 2).mean())), 4)}
        all_res[tgt] = res
        print(tgt, json.dumps(res), flush=True)

    json.dump(all_res, open("results/baselines.json", "w"), indent=2)


if __name__ == "__main__":
    main()
