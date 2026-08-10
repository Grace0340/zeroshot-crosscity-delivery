# -*- coding: utf-8 -*-
"""Statistical significance tests for the paper's key comparisons (R1a).

All tests are paired and two-sided:
  A. Staged point accuracy: Wilcoxon signed-rank on daily MAE (day 1+),
     ensemble vs. each alternative predictor, per city and pooled.
  B. Interval calibration: Wilcoxon signed-rank on daily coverage error
     |cov_t - 0.9|, online CP vs. static transfer and vs. ACI (gamma=0.1).
  C. Seed-level (n=5): one-sample t-tests of zero-shot MAE against the
     deterministic HA baseline, and of overall online-CP coverage against
     the nominal 0.9.
  D. Decision replay: Wilcoxon signed-rank on per-day newsvendor unit cost,
     online-calibrated quantile vs. point strategy, per cost ratio.

Reads results/*.json and results/npz/*.npz (v3, 5 seeds).
Writes results/significance_v3.json.
"""
import json

import numpy as np
from scipy import stats

CITIES = ["sh", "hz", "cq", "yt"]
TAG = "_v3"
RES = "results"
NPZ = "results/npz"


def wtest(a, b):
    """Paired Wilcoxon; returns mean difference (a-b) and two-sided p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if np.allclose(a, b):
        return {"mean_diff": 0.0, "p": 1.0, "n": int(len(a))}
    st, p = stats.wilcoxon(a, b)
    return {"mean_diff": round(float((a - b).mean()), 4),
            "p": float(f"{p:.3g}"), "n": int(len(a))}


# ---------------------------------------------------------------- A. staged MAE
def staged_tests():
    res = {}
    pooled = {}
    for tgt in CITIES:
        st = np.load(f"{NPZ}/staged_{tgt}{TAG}.npz")
        nv = np.load(f"{NPZ}/naive_{tgt}{TAG}.npz")
        series = {
            "ensemble": st["ensemble"], "model-staged": st["model-staged"],
            "ha-target": st["ha-target"], "ha-source": st["ha-source"],
            "persistence": nv["persistence"], "seasonal-naive": nv["seasonal_naive"],
        }
        if "ensemble3" in st:
            series["ensemble3"] = st["ensemble3"]
        pairs = [("ensemble", "ha-target"), ("ensemble", "model-staged"),
                 ("ensemble", "persistence"), ("ensemble", "seasonal-naive"),
                 ("model-staged", "ha-source")]
        if "ensemble3" in series:
            pairs += [("ensemble3", "ensemble"), ("ensemble3", "persistence"),
                      ("ensemble3", "ha-target"), ("ensemble3", "seasonal-naive")]
        res[tgt] = {}
        for a, b in pairs:
            key = f"{a}_vs_{b}"
            xa, xb = series[a][1:], series[b][1:]        # day 1+ only
            res[tgt][key] = wtest(xa, xb)
            pooled.setdefault(key, [[], []])
            pooled[key][0].extend(xa); pooled[key][1].extend(xb)
    res["pooled"] = {k: wtest(v[0], v[1]) for k, v in pooled.items()}
    return res


# ------------------------------------------------------------ B. coverage error
def coverage_tests():
    res = {}
    pooled = {}
    for tgt in CITIES:
        online = np.load(f"{NPZ}/aci_{tgt}_online{TAG}.npz")["cov"]
        aci = np.load(f"{NPZ}/aci_{tgt}_g0.1{TAG}.npz")["cov"]
        prd = np.load(f"{NPZ}/preds_{tgt}{TAG}.npz")
        P, Y = prd["pred_quantiles"], prd["y_true"]
        q0 = json.load(open(f"{RES}/zeroshot_{tgt}{TAG}.json"))["interval_naive"]["q_hat"]
        lo, hi = P[:, :, :, 0] - q0, P[:, :, :, -1] + q0
        static = ((Y >= lo) & (Y <= hi)).reshape(P.shape[0], -1).mean(axis=1)

        err = {k: np.abs(v - 0.9) for k, v in
               [("online", online), ("aci-g0.1", aci), ("static", static)]}
        res[tgt] = {}
        for b in ["static", "aci-g0.1"]:
            key = f"online_vs_{b}"
            res[tgt][key] = wtest(err["online"], err[b])
            pooled.setdefault(key, [[], []])
            pooled[key][0].extend(err["online"]); pooled[key][1].extend(err[b])
    res["pooled"] = {k: wtest(v[0], v[1]) for k, v in pooled.items()}
    return res


# -------------------------------------------------------------- C. seed level
def seed_tests():
    seeds = json.load(open(f"{RES}/seeds_summary{TAG}.json"))
    base = json.load(open(f"{RES}/baselines.json"))
    res = {}
    for tgt in CITIES:
        mae = [s["mae"] for s in seeds[tgt]["per_seed"]]
        cov = [s["cov_overall_online"] for s in seeds[tgt]["per_seed"]]
        t1, p1 = stats.ttest_1samp(mae, base[tgt]["ha-global"]["MAE"])
        t2, p2 = stats.ttest_1samp(cov, 0.9)
        res[tgt] = {
            "zeroshot_mae_vs_ha": {"mean": round(float(np.mean(mae)), 3),
                                   "ha": base[tgt]["ha-global"]["MAE"],
                                   "p": float(f"{p1:.3g}"), "n": len(mae)},
            "online_cov_vs_nominal": {"mean": round(float(np.mean(cov)), 4),
                                      "p": float(f"{p2:.3g}"), "n": len(cov)},
        }
    return res


# ------------------------------------------------------------- D. decision cost
def interp_quantile(P, qs, kappa):
    qs = np.asarray(qs)
    if kappa <= qs[0]:
        return P[..., 0]
    if kappa >= qs[-1]:
        return P[..., -1]
    i = np.searchsorted(qs, kappa) - 1
    w = (kappa - qs[i]) / (qs[i + 1] - qs[i])
    return (1 - w) * P[..., i] + w * P[..., i + 1]


def online_correct(c_raw, Y, kappa):
    W, N, H = c_raw.shape
    c_adj = c_raw.copy()
    res = [[] for _ in range(N)]
    for t in range(W):
        for j in range(N):
            if res[j]:
                c_adj[t, j] = c_raw[t, j] + np.quantile(np.concatenate(res[j]), kappa)
            res[j].append(Y[t, j] - c_raw[t, j])
    return c_adj


def decision_tests():
    res = {}
    pooled = {}
    for tgt in CITIES:
        prd = np.load(f"{NPZ}/preds_{tgt}{TAG}.npz")
        P, Y, qs = prd["pred_quantiles"], prd["y_true"], list(prd["quantiles"])
        med = P[..., qs.index(0.5)]
        res[tgt] = {}
        for co, cu in [(1.0, 3.0), (3.0, 1.0)]:
            kappa = cu / (co + cu)
            c_onl = np.maximum(online_correct(interp_quantile(P, qs, kappa), Y, kappa), 0)
            cost_pt = (co * np.maximum(med - Y, 0) + cu * np.maximum(Y - med, 0)).reshape(P.shape[0], -1).mean(axis=1)
            cost_on = (co * np.maximum(c_onl - Y, 0) + cu * np.maximum(Y - c_onl, 0)).reshape(P.shape[0], -1).mean(axis=1)
            key = f"co{co:g}_cu{cu:g}"
            res[tgt][key] = wtest(cost_on, cost_pt)
            pooled.setdefault(key, [[], []])
            pooled[key][0].extend(cost_on); pooled[key][1].extend(cost_pt)
        print(f"  decision {tgt} done", flush=True)
    res["pooled"] = {k: wtest(v[0], v[1]) for k, v in pooled.items()}
    return res


def main():
    out = {
        "staged_mae_day1+": staged_tests(),
        "coverage_error": coverage_tests(),
        "seed_level": seed_tests(),
        "decision_cost": decision_tests(),
    }
    json.dump(out, open(f"{RES}/significance{TAG}.json", "w"), indent=2)
    for block, r in out.items():
        print(f"== {block} ==", flush=True)
        for tgt, tests in r.items():
            for k, v in tests.items():
                print(f"  {tgt:6s} {k:32s} {v}", flush=True)
    print("saved", f"{RES}/significance{TAG}.json")


if __name__ == "__main__":
    main()
