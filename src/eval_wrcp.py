"""WR-CP-style conformal baseline: optimal-transport weights on region embeddings.

Approximates Wasserstein-regularized conformal prediction (Xu et al., 2025)
by computing a Sinkhorn transport plan between source calibration regions and
target regions in LLM embedding space, then applying region-specific weighted
quantiles to widen raw quantile intervals.

Requires existing calib_{tgt}{tag}.npz and preds_{tgt}{tag}.npz from
train_zeroshot.py (--calib_only if checkpoints already exist).
"""
import argparse
import json

import numpy as np

try:
    import ot

    HAS_OT = True
except ImportError:
    HAS_OT = False


def region_masses(region_idx, n_regions):
    counts = np.bincount(region_idx, minlength=n_regions).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return counts / counts.sum()


def sinkhorn_plan(src_emb, tgt_emb, src_mass, tgt_mass, reg=0.05, n_iter=300):
    """Return transport plan [Ns, Nt] from source regions to target regions."""
    src = src_emb / (np.linalg.norm(src_emb, axis=1, keepdims=True) + 1e-8)
    tgt = tgt_emb / (np.linalg.norm(tgt_emb, axis=1, keepdims=True) + 1e-8)
    C = ot.dist(src, tgt, metric="sqeuclidean")
    C = C / (C.max() + 1e-8)
    if HAS_OT:
        return ot.sinkhorn(src_mass, tgt_mass, C, reg, numItermax=n_iter)
    # Fallback: softmax kernel plan (no POT)
    K = np.exp(-C / max(reg, 1e-3))
    plan = src_mass[:, None] * K
    plan = plan * tgt_mass[None, :]
    return plan / (plan.sum() + 1e-8)


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1] + 1e-8
    k = np.searchsorted(cw, q)
    return v[min(k, len(v) - 1)]


def eval_method(P, Y, q_by_region, scale_t, rel):
    adj = q_by_region[None, :, None] * (scale_t if rel else 1.0)
    lo, hi = P[..., 0] - adj, P[..., -1] + adj
    cover_r = ((Y >= lo) & (Y <= hi)).mean(axis=(0, 2))
    return float(cover_r.mean()), float(cover_r.min()), float((hi - lo).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["sh", "hz", "cq", "yt"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tag", default="_v3")
    ap.add_argument("--reg", type=float, default=0.05, help="Sinkhorn regularization")
    ap.add_argument("--regs", nargs="+", type=float, default=None,
                    help="Grid over reg; default single --reg")
    args = ap.parse_args()
    regs = args.regs if args.regs else [args.reg]

    all_res = {}
    for tgt in args.targets:
        cal = np.load(f"results/calib_{tgt}{args.tag}.npz")
        prd = np.load(f"results/preds_{tgt}{args.tag}.npz")
        s_abs = cal["scores"].astype(np.float64)
        s_rel = s_abs / cal["scale"]
        ridx = cal["region_idx"].astype(np.int32)
        src_emb = cal["source_emb"]
        P, Y, tgt_emb = prd["pred_quantiles"], prd["y_true"], prd["target_emb"]
        imed = list(prd["quantiles"]).index(0.5)
        scale_t = P[..., imed] + 1.0
        N_t, N_s = tgt_emb.shape[0], src_emb.shape[0]
        M = len(s_abs)
        q_lvl = min(1.0, (1 - args.alpha) * (1 + 1 / M))

        src_mass = region_masses(ridx, N_s)
        tgt_mass = np.ones(N_t) / N_t
        res = {}
        for reg in regs:
            plan = sinkhorn_plan(src_emb, tgt_emb, src_mass, tgt_mass, reg=reg)
            for key, scores, rel in [("abs", s_abs, False), ("rel", s_rel, True)]:
                q_reg = np.zeros(N_t)
                for j in range(N_t):
                    # sample weight = sum_s plan[s, ridx] for each calibration point
                    w = plan[ridx, j]
                    w = w / (w.sum() + 1e-8)
                    q_reg[j] = weighted_quantile(scores, w, q_lvl)
                cov, cov_min, wid = eval_method(P, Y, q_reg, scale_t, rel)
                tag = f"wrcp-{key}-r{reg}"
                res[tag] = {"cov": round(cov, 4), "cov_min_region": round(cov_min, 4),
                            "width": round(wid, 2), "reg": reg, "has_ot": HAS_OT}
        all_res[tgt] = res
        print(f"== {tgt} (OT={HAS_OT}) ==", flush=True)
        for k, v in res.items():
            print(f"  {k:18s} cov={v['cov']:.3f} min_r={v['cov_min_region']:.3f} "
                  f"width={v['width']}", flush=True)

    out = f"results/wrcp_grid{args.tag}.json"
    json.dump(all_res, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
