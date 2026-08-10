# -*- coding: utf-8 -*-
"""Generate LaTeX tables directly from the results JSON files.

Never hand-copy numbers: rerun this script whenever results change.
Outputs to tables/*.tex at the repository root.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUTS = [ROOT / "tables"]
for _o in OUTS:
    _o.mkdir(parents=True, exist_ok=True)

CITIES = ["sh", "hz", "cq", "yt"]
CITY_NAME = {"sh": "Shanghai", "hz": "Hangzhou", "cq": "Chongqing", "yt": "Yantai"}


def jload(name):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def ms(s, nd=2):
    m, sd = s.split("\u00b1")
    return f"{float(m):.{nd}f}$\\pm${float(sd):.{nd}f}"


def seed_pm(js, city, key="mae", nd=2):
    """Mean +/- sample std (ddof=1) over per-seed values, matching tab_point."""
    import statistics
    vals = [s[key] for s in js[city]["per_seed"]]
    return f"{statistics.mean(vals):.{nd}f}$\\pm${statistics.stdev(vals):.{nd}f}"


def write(name, text):
    for out in OUTS:
        (out / name).write_text(text, encoding="utf-8")
    print("wrote", name)


# ------------------------------------------------ Table 1: point accuracy
def tab_point():
    seeds = jload("seeds_summary_v3.json")
    base = jload("baselines.json")
    staged = jload("staged_summary_v3.json")
    naive = jload("naive_baselines_v3.json")

    def seed_ms(c, key, nd=2):
        import statistics
        vals = [s[key] for s in seeds[c]["per_seed"]]
        return (f"{statistics.mean(vals):.{nd}f}$\\pm$"
                f"{statistics.stdev(vals):.{nd}f}")

    rows = []
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append("Method (MAE / RMSE) & " + " & ".join(CITY_NAME[c] for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\emph{Strict zero-shot (no target observations)}} \\")
    rows.append("HA climatology (source) & " + " & ".join(
        f"{base[c]['ha-global']['MAE']:.2f} / {base[c]['ha-global']['RMSE']:.2f}"
        for c in CITIES) + r" \\")
    rows.append("kNN profile & " + " & ".join(
        f"{base[c]['knn-prof']['MAE']:.2f} / {base[c]['knn-prof']['RMSE']:.2f}"
        for c in CITIES) + r" \\")
    rows.append("ST backbone (ours, 5 seeds) & " + " & ".join(
        f"{seed_ms(c, 'mae')} / {seed_ms(c, 'rmse')}" for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\emph{Staged deployment, day 8+ (rolling target history), MAE}} \\")
    staged_rows = [
        ("Model (rolling history)", {c: staged[c]["model-staged"]["day8+"] for c in CITIES}),
        ("Target climatology (accum.)", {c: staged[c]["ha-target"]["day8+"] for c in CITIES}),
        ("Persistence (previous day)", {c: naive[c]["persistence"]["day8+"] for c in CITIES}),
        ("Seasonal-naive (1 week)", {c: naive[c]["seasonal-naive"]["day8+"] for c in CITIES}),
        ("Ensemble (model+clim.)", {c: staged[c]["ensemble"]["day8+"] for c in CITIES}),
        ("Full ensemble (ours)", {c: staged[c]["ensemble3"]["day8+"] for c in CITIES}),
    ]
    if (RES / "chronos_summary_v3.json").exists():
        chronos = jload("chronos_summary_v3.json")
        staged_rows.insert(4, ("Chronos-Bolt (pretrained)",
                               {c: chronos[c]["day8+"] for c in CITIES}))
    best = {c: min(vals[c] for _, vals in staged_rows) for c in CITIES}
    for lab, vals in staged_rows:
        cells = [f"\\textbf{{{vals[c]:.2f}}}" if vals[c] == best[c] else f"{vals[c]:.2f}"
                 for c in CITIES]
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_point_accuracy.tex", "\n".join(rows) + "\n")


# ---------------------------- Table 1b: nominal-coverage sensitivity (alpha grid)
def tab_alpha():
    ab = jload("alpha_burnin_v3.json")

    def cw(entry):
        return f"{entry['cov'] * 100:.1f} / {entry['width']:.1f}"

    rows = []
    rows.append(r"\begin{tabular}{llcccc}")
    rows.append(r"\toprule")
    rows.append("Nominal & Method & " + " & ".join(CITY_NAME[c] for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    for a, nom in [("0.05", "95\\%"), ("0.1", "90\\%"), ("0.2", "80\\%")]:
        rows.append(f"\\multirow{{2}}{{*}}{{{nom}}} & Static transfer & " + " & ".join(
            cw(ab[c]["alpha"][a]["static"]["overall"]) for c in CITIES) + r" \\")
        rows.append(" & Online CP (ours) & " + " & ".join(
            f"\\textbf{{{cw(ab[c]['alpha'][a]['online-cp']['overall'])}}}"
            for c in CITIES) + r" \\")
        if a != "0.2":
            rows.append(r"\midrule")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_alpha_sensitivity.tex", "\n".join(rows) + "\n")


# ----------------------------------------- Table 2: interval quality
def tab_interval():
    aci = jload("aci_replay_v3.json")
    grid = jload("calibration_grid_v3.json")
    cal_cmp = jload("calibration_compare_v3.json")

    def cw(entry):
        return f"{entry['cov'] * 100:.1f} / {entry['width']:.1f}"

    rows = []
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append("Coverage (\\%) / width & " + " & ".join(CITY_NAME[c] for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    rows.append("Static SC (source calib.) & " + " & ".join(
        cw(aci[c]["static(q0)"]["overall"]) for c in CITIES) + r" \\")
    wbest = {c: max((v for k, v in grid[c].items() if k.startswith("wz")),
                    key=lambda v: v["cov"]) for c in CITIES}
    rows.append("Weighted SC (best of grid) & " + " & ".join(
        f"{wbest[c]['cov'] * 100:.1f} / {wbest[c]['width']:.1f}" for c in CITIES) + r" \\")
    rows.append("WR-CP (OT weights, best) & " + " & ".join(
        cw(cal_cmp[c]["wrcp_best"]) for c in CITIES) + r" \\")
    for g in ["0.02", "0.05", "0.1"]:
        rows.append(f"ACI ($\\gamma$={g}) & " + " & ".join(
            cw(aci[c][f"aci-g{g}"]["overall"]) for c in CITIES) + r" \\")
    rows.append(r"Online CP, day 1--3 (ours) & " + " & ".join(
        cw(aci[c]["online-cp"]["day1-3"]) for c in CITIES) + r" \\")
    rows.append(r"Online CP, overall (ours) & " + " & ".join(
        f"\\textbf{{{cw(aci[c]['online-cp']['overall'])}}}" for c in CITIES) + r" \\")
    if (RES / "chronos_calib_v3.json").exists():
        cc = jload("chronos_calib_v3.json")
        rows.append(r"\midrule")
        rows.append(r"Chronos-Bolt raw band (day 1+) & " + " & ".join(
            cw(cc[c]["raw-band"]["overall_d1+"]) for c in CITIES) + r" \\")
        rows.append(r"Chronos-Bolt + online CP (day 1+) & " + " & ".join(
            cw(cc[c]["online-cp"]["overall_d1+"]) for c in CITIES) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_interval_quality.tex", "\n".join(rows) + "\n")


# ------------------------------------------ Table 3: decision costs
def tab_decision():
    dec = jload("decision_replay_v3.json")
    ratios = [("co1_cu1", "1:1"), ("co1_cu3", "1:3"), ("co3_cu1", "3:1")]

    rows = []
    rows.append(r"\begin{tabular}{llcccc}")
    rows.append(r"\toprule")
    rows.append("$c_o$:$c_u$ & Strategy & " + " & ".join(CITY_NAME[c] for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    for rk, rl in ratios:
        rows.append(f"\\multirow{{3}}{{*}}{{{rl}}} & Point (median) & " + " & ".join(
            f"{dec[c][rk]['cost_point']:.2f}" for c in CITIES) + r" \\")
        rows.append(" & Raw quantile & " + " & ".join(
            f"{dec[c][rk]['cost_rawq']:.2f}" for c in CITIES) + r" \\")
        rows.append(" & Online-calibrated (ours) & " + " & ".join(
            f"\\textbf{{{dec[c][rk]['cost_onlineq']:.2f}}} ({dec[c][rk]['save_onlineq_pct']:+.1f}\\%)"
            for c in CITIES) + r" \\")
        if rk != "co3_cu1":
            rows.append(r"\midrule")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_decision_cost.tex", "\n".join(rows) + "\n")


# ------------------------------------------------ Table 4: ablation
def tab_ablation():
    files = [("seeds_summary_v3.json", "Backbone (final)"),
             ("seeds_summary_retr2.json", "+ shape retrieval"),
             ("seeds_summary_anchor.json", "+ HA anchor-residual")]
    rows = []
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append("Variant (MAE) & " + " & ".join(CITY_NAME[c] for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    for fn, lab in files:
        js = jload(fn)
        rows.append(f"{lab} & " + " & ".join(seed_pm(js, c) for c in CITIES) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_ablation.tex", "\n".join(rows) + "\n")


# -------------------------------- Table 5: protocol upper bound (IMPEL vs strict zero-shot)
def tab_protocol():
    seeds = jload("seeds_summary_v3.json")
    impel = jload("impel_loco.json")

    rows = []
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append("Setting (MAE) & " + " & ".join(CITY_NAME[c] for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    rows.append("Strict zero-shot (ours) & " + " & ".join(
        seed_pm(seeds, c) for c in CITIES) + r" \\")
    rows.append(r"IMPEL (mean over sources) & " + " & ".join(
        f"{impel[c]['mean_mae']:.2f}" for c in CITIES) + r" \\")
    rows.append(r"IMPEL (best source) & " + " & ".join(
        f"{impel[c]['best_mae']:.2f}" for c in CITIES) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_protocol_compare.tex", "\n".join(rows) + "\n")


if __name__ == "__main__":
    tab_point()
    tab_alpha()
    tab_interval()
    tab_decision()
    tab_ablation()
    tab_protocol()
