"""Grid evaluation of static calibration variants (offline; reads the
calib/preds artifacts written by train_zeroshot.py, no retraining).

Per target city:
  naive-abs   : absolute scores, equal weights (plain split conformal)
  naive-rel   : relative scores (divided by prediction scale), equal weights
  wz-abs-tau  : similarity-weighted (z-normalized), absolute scores, temperature grid
  wz-rel-tau  : same with relative scores
Reports overall coverage/width + worst-region coverage.

Weights are computed at the (target region x source region) level (a fast
region-level GEMM), then expanded to sample level via region_idx.
"""
import argparse
import json

import numpy as np


def eval_method(P, Y, scale_t, q_by_region, rel):
    """q_by_region [N]; if rel=True the correction acts in relative space."""
    adj = q_by_region[None, :, None] * (scale_t if rel else 1.0)
    lo, hi = P[..., 0] - adj, P[..., -1] + adj
    cover_r = ((Y >= lo) & (Y <= hi)).mean(axis=(0, 2))
    return float(cover_r.mean()), float(cover_r.min()), float((hi - lo).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["hz", "cq", "yt", "sh"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--taus", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--tag", default="", help="read tagged artifacts, e.g. _v2")
    args = ap.parse_args()

    all_res = {}
    for tgt in args.targets:
        cal = np.load(f"results/calib_{tgt}{args.tag}.npz")
        prd = np.load(f"results/preds_{tgt}{args.tag}.npz")
        s_abs = cal["scores"].astype(np.float64)
        s_rel = s_abs / cal["scale"]
        ridx, src_emb = cal["region_idx"], cal["source_emb"]
        P, Y, z_t = prd["pred_quantiles"], prd["y_true"], prd["target_emb"]
        imed = list(prd["quantiles"]).index(0.5)
        scale_t = P[..., imed] + 1.0                                  # [W,N,H]
        N, M = z_t.shape[0], len(s_abs)
        q_lvl = min(1.0, (1 - args.alpha) * (1 + 1 / M))

        # region-level similarity [N_target, N_source], z-normalized per row
        zc = src_emb / np.linalg.norm(src_emb, axis=1, keepdims=True)
        zt = z_t / np.linalg.norm(z_t, axis=1, keepdims=True)
        S = zt @ zc.T
        S = (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-8)

        res = {}
        for name, scores, rel in [("naive-abs", s_abs, False), ("naive-rel", s_rel, True)]:
            q = np.full(N, np.quantile(scores, q_lvl))
            cov, cov_min, wid = eval_method(P, Y, scale_t, q, rel)
            res[name] = {"cov": round(cov, 4), "cov_min_region": round(cov_min, 4), "width": round(wid, 2)}

        # Sort each score type once; the weighted quantile is then a cumsum
        # search over weights re-ordered by score.
        prep = {}
        for key, scores in [("abs", s_abs), ("rel", s_rel)]:
            order = np.argsort(scores)
            prep[key] = (scores[order], ridx[order])
        for tau in args.taus:
            Wr = np.exp(S / tau)                       # [N, N_src] region-level weights
            Wr = Wr / Wr.max(axis=1, keepdims=True)
            for key, rel in [("abs", False), ("rel", True)]:
                s_sorted, r_sorted = prep[key]
                w = Wr[:, r_sorted]                    # [N, M] sample-level weights (score-sorted)
                cw = np.cumsum(w, axis=1)
                cw /= cw[:, -1:] + 1
                q = np.empty(N)
                for j in range(N):
                    k = np.searchsorted(cw[j], q_lvl)
                    q[j] = s_sorted[min(k, M - 1)]
                cov, cov_min, wid = eval_method(P, Y, scale_t, q, rel)
                res[f"wz-{key}-t{tau}"] = {"cov": round(cov, 4), "cov_min_region": round(cov_min, 4),
                                           "width": round(wid, 2)}
        all_res[tgt] = res
        print(f"== target {tgt} (N={N}, M={M}) ==", flush=True)
        for k, v in res.items():
            print(f"  {k:14s} cov={v['cov']:.3f} min_region={v['cov_min_region']:.3f} width={v['width']}", flush=True)

    json.dump(all_res, open(f"results/calibration_grid{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
