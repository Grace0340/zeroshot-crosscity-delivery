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


def write(name, text):
    for out in OUTS:
        (out / name).write_text(text, encoding="utf-8")
    print("wrote", name)


# ------------------------------------------------ Table 1: point accuracy
def tab_point():
    seeds = jload("seeds_summary_v3.json")
    base = jload("baselines.json")
    eff = jload("interval_efficiency_v3.json")
    staged = jload("staged_summary_v3.json")

    rows = []
    rows.append(r"\begin{tabular}{lcccc}")
    rows.append(r"\toprule")
    rows.append("Method & " + " & ".join(CITY_NAME[c] for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\emph{Strict zero-shot (no target observations)}} \\")
    rows.append("HA climatology (source) & " + " & ".join(
        f"{base[c]['ha-global']['MAE']:.2f}" for c in CITIES) + r" \\")
    rows.append("kNN profile & " + " & ".join(
        f"{base[c]['knn-prof']['MAE']:.2f}" for c in CITIES) + r" \\")
    rows.append("ST backbone (ours, 5 seeds) & " + " & ".join(
        ms(seeds[c]["agg"]["mae"]) for c in CITIES) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\emph{Staged deployment, day 8+ (rolling target history)}} \\")
    rows.append("Model (rolling history) & " + " & ".join(
        f"{staged[c]['model-staged']['day8+']:.2f}" for c in CITIES) + r" \\")
    rows.append("Target climatology (accum.) & " + " & ".join(
        f"{staged[c]['ha-target']['day8+']:.2f}" for c in CITIES) + r" \\")
    rows.append(r"Online ensemble (ours) & " + " & ".join(
        f"\\textbf{{{staged[c]['ensemble']['day8+']:.2f}}}" for c in CITIES) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_point_accuracy.tex", "\n".join(rows) + "\n")


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
        rows.append(f"{lab} & " + " & ".join(ms(js[c]["agg"]["mae"]) for c in CITIES) + r" \\")
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
        ms(seeds[c]["agg"]["mae"]) for c in CITIES) + r" \\")
    rows.append(r"IMPEL (mean over sources) & " + " & ".join(
        f"{impel[c]['mean_mae']:.2f}" for c in CITIES) + r" \\")
    rows.append(r"IMPEL (best source) & " + " & ".join(
        f"{impel[c]['best_mae']:.2f}" for c in CITIES) + r" \\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    write("tab_protocol_compare.tex", "\n".join(rows) + "\n")


if __name__ == "__main__":
    tab_point()
    tab_interval()
    tab_decision()
    tab_ablation()
    tab_protocol()
