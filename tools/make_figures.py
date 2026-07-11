# -*- coding: utf-8 -*-
"""Publication figures for the zero-shot cross-city delivery demand study.

All figures are generated from results/*.json and results/npz/*.npz produced
by the experiment pipeline (adopted recipe, 5 seeds). Vector PDF + PNG preview.
"""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NPZ = RES / "npz"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

CITIES = ["sh", "hz", "cq", "yt"]
CITY_NAME = {"sh": "Shanghai", "hz": "Hangzhou", "cq": "Chongqing", "yt": "Yantai"}

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GRAY = "#7F7F7F"
C_YELLOW = "#F0E442"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 120,
})

SINGLE_W = 3.5   # IEEE single column (in)
DOUBLE_W = 7.16  # IEEE double column (in)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def jload(name):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def parse_ms(s):
    """'9.065±1.553' -> (9.065, 1.553)"""
    m, sd = s.split("\u00b1")
    return float(m), float(sd)


def rolling(x, w=7):
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.nanmean(x[lo:i + 1])
    return out


# ---------------------------------------------------------------- Fig 1: motivation
def fig_motivation():
    """Claim: naive cross-city transfer of conformal calibration under-covers in
    every target city on day 0 (56-74% vs nominal 90%)."""
    aci = jload("aci_replay_v3.json")
    day0 = [aci[c]["static(q0)"]["day0"]["cov"] for c in CITIES]
    overall = [aci[c]["static(q0)"]["overall"]["cov"] for c in CITIES]

    fig, ax = plt.subplots(figsize=(SINGLE_W, 2.35))
    x = np.arange(len(CITIES))
    w = 0.38
    ax.bar(x - w / 2, day0, w, color=C_RED, label="Day 0")
    ax.bar(x + w / 2, overall, w, color=C_ORANGE, label="Whole horizon")
    ax.axhline(0.9, ls="--", lw=1, color=C_GRAY)
    ax.text(len(CITIES) - 0.45, 0.915, "nominal 90%", color=C_GRAY, fontsize=7, ha="right")
    for xi, v in zip(x - w / 2, day0):
        ax.text(xi, v + 0.015, f"{v * 100:.0f}", ha="center", fontsize=7, color=C_RED)
    ax.set_xticks(x, [CITY_NAME[c] for c in CITIES])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Empirical coverage")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncols=2)
    save(fig, "fig_motivation_day0")


# --------------------------------------------------- Fig 2: online CP convergence
def fig_convergence():
    """Claim: full-recompute online CP reaches nominal coverage within 1-3 days
    in all four cities, while ACI converges far more slowly."""
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_W, 1.9), sharey=True)
    for ax, c in zip(axes, CITIES):
        on = np.load(NPZ / f"aci_{c}_online_v3.npz")
        ac = np.load(NPZ / f"aci_{c}_g0.05_v3.npz")
        days = np.arange(len(on["cov"]))
        ax.plot(days, rolling(on["cov"]), color=C_BLUE, lw=1.4, label="Online CP (ours)")
        ax.plot(days, rolling(ac["cov"]), color=C_ORANGE, lw=1.2, label=r"ACI ($\gamma$=0.05)")
        ax.axhline(0.9, ls="--", lw=0.8, color=C_GRAY)
        ax.scatter([0], [on["cov"][0]], color=C_RED, s=12, zorder=5,
                   label="Day 0 (source calib.)")
        ax.set_title(CITY_NAME[c])
        ax.set_xlabel("Days since deployment")
        ax.set_ylim(0.45, 1.0)
    axes[0].set_ylabel("Coverage (7-day rolling)")
    axes[0].legend(loc="lower right", fontsize=6.3)
    fig.tight_layout()
    save(fig, "fig_online_cp_convergence")


# ------------------------------------------- Fig 3: coverage-width trade-off
def fig_cov_width():
    """Claim: only online CP sits at nominal coverage; static/weighted/ACI
    under-cover, HA empirical intervals over-cover."""
    aci = jload("aci_replay_v3.json")
    grid = jload("calibration_grid_v3.json")
    eff = jload("interval_efficiency_v3.json")

    methods = [
        ("Static SC (source)", C_RED, "o"),
        ("Weighted SC (best)", C_PURPLE, "s"),
        (r"ACI ($\gamma$=0.05)", C_ORANGE, "^"),
        ("Online CP (ours)", C_BLUE, "*"),
        ("HA-emp + online CP", C_GREEN, "D"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_W, 1.95), sharey=True)
    for ax, c in zip(axes, CITIES):
        wz = max((v for k, v in grid[c].items() if k.startswith("wz")),
                 key=lambda v: v["cov"])
        pts = [
            (aci[c]["static(q0)"]["overall"]["width"], aci[c]["static(q0)"]["overall"]["cov"]),
            (wz["width"], wz["cov"]),
            (aci[c]["aci-g0.05"]["overall"]["width"], aci[c]["aci-g0.05"]["overall"]["cov"]),
            (aci[c]["online-cp"]["overall"]["width"], aci[c]["online-cp"]["overall"]["cov"]),
            (eff[c]["interval_ha-emp"]["width_overall"], eff[c]["interval_ha-emp"]["cov_overall"]),
        ]
        for (wd, cv), (lab, col, mk) in zip(pts, methods):
            ax.scatter(wd, cv, color=col, marker=mk, s=48 if mk == "*" else 26,
                       label=lab, zorder=5)
        ax.axhline(0.9, ls="--", lw=0.8, color=C_GRAY)
        ax.set_title(CITY_NAME[c])
        ax.set_xlabel("Mean interval width")
        ax.set_ylim(0.55, 1.02)
    axes[0].set_ylabel("Empirical coverage")
    axes[-1].legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    save(fig, "fig_coverage_width")


# ------------------------------------------------- Fig 4: staged cold-start MAE
def fig_staged():
    """Claim: with rolling history the model overtakes climatology in the most
    complex city (SH); the online ensemble is uniformly safe and often best."""
    labels = [("model-staged", "Model (rolling history)", C_BLUE),
              ("ha-source", "Source climatology", C_GRAY),
              ("ha-target", "Target climatology (accum.)", C_ORANGE),
              ("ensemble", "Online ensemble (ours)", C_GREEN)]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_W, 2.0), sharex=False)
    for ax, c in zip(axes, CITIES):
        d = np.load(NPZ / f"staged_{c}_v3.npz")
        days = np.arange(len(d["model-staged"]))
        for key, lab, col in labels:
            ax.plot(days, rolling(d[key]), lw=1.2, color=col, label=lab)
        ax.set_title(CITY_NAME[c])
        ax.set_xlabel("Days since deployment")
    axes[0].set_ylabel("Daily MAE (7-day rolling)")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncols=4, fontsize=7,
               bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout()
    save(fig, "fig_staged_mae")


# ------------------------------------------------------ Fig 5: decision costs
def fig_decision():
    """Claim: under asymmetric costs, calibrated quantile decisions cut
    newsvendor cost by up to 65% vs point forecasts."""
    dec = jload("decision_replay_v3.json")
    ratios = [("co1_cu1", "1:1"), ("co1_cu3", "1:3"), ("co3_cu1", "3:1")]
    strategies = [("cost_point", "Point (median)", C_GRAY),
                  ("cost_rawq", "Raw quantile", C_ORANGE),
                  ("cost_onlineq", "Online-calibrated quantile (ours)", C_BLUE)]
    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_W, 2.1), sharex=True)
    x = np.arange(len(ratios))
    w = 0.26
    for ax, c in zip(axes, CITIES):
        for i, (key, lab, col) in enumerate(strategies):
            vals = [dec[c][r][key] for r, _ in ratios]
            ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab)
        for xi, (r, _) in zip(x, ratios):
            sv = dec[c][r]["save_onlineq_pct"]
            if abs(sv) > 1:
                ymax = max(dec[c][r][k] for k, _, _ in strategies)
                ax.text(xi, ymax * 1.03, f"{sv:+.0f}%", ha="center", fontsize=6.5,
                        color=C_GREEN if sv > 0 else C_RED)
        ax.set_xticks(x, [lab for _, lab in ratios])
        ax.set_xlabel("Overage : underage cost")
        ax.set_title(CITY_NAME[c])
    axes[0].set_ylabel("Mean unit cost")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncols=3, fontsize=7,
               bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout()
    save(fig, "fig_decision_cost")


# ---------------------------------------------------------- Fig 6: ablations
def fig_ablation():
    """Claim (honest negatives): retrieval/anchoring do not beat the plain
    backbone; semantic weighting improves coverage by <=2.6 pp."""
    variants = [("seeds_summary_v3.json", "Backbone (final)", C_BLUE),
                ("seeds_summary_retr2.json", "+ Shape retrieval", C_SKY),
                ("seeds_summary_anchor.json", "+ HA anchor-residual", C_PURPLE)]
    base = jload("baselines.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.1),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(CITIES))
    w = 0.2
    for i, (fn, lab, col) in enumerate(variants):
        js = jload(fn)
        mm = [parse_ms(js[c]["agg"]["mae"]) for c in CITIES]
        ax1.bar(x + (i - 1.5) * w, [m for m, _ in mm], w,
                yerr=[s for _, s in mm], color=col, label=lab,
                error_kw={"lw": 0.7, "capsize": 1.5})
    ax1.bar(x + 1.5 * w, [base[c]["ha-global"]["MAE"] for c in CITIES], w,
            color=C_GRAY, label="HA climatology")
    ax1.set_xticks(x, [CITY_NAME[c] for c in CITIES], fontsize=7)
    ax1.set_ylabel("Zero-shot MAE")
    ax1.set_title("(a) Point-accuracy ablation", loc="left")
    # legend in a row above the panel title so it covers neither bars nor title
    ax1.legend(fontsize=6.3, ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.14), columnspacing=0.8,
               handlelength=1.2, borderaxespad=0.0)

    grid = jload("calibration_grid_v3.json")
    naive = [grid[c]["naive-abs"]["cov"] for c in CITIES]
    wbest = [max(v["cov"] for k, v in grid[c].items() if k.startswith("wz"))
             for c in CITIES]
    ax2.bar(x - 0.19, naive, 0.38, color=C_RED, label="Naive SC")
    ax2.bar(x + 0.19, wbest, 0.38, color=C_PURPLE, label="Weighted SC (best)")
    ax2.axhline(0.9, ls="--", lw=0.8, color=C_GRAY)
    ax2.set_xticks(x, [CITY_NAME[c] for c in CITIES], fontsize=7)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Coverage")
    ax2.set_title("(b) Semantic weighting (negative result)", loc="left")
    ax2.legend(fontsize=6.3, ncols=2, loc="lower center",
               bbox_to_anchor=(0.5, 1.14), columnspacing=0.8,
               handlelength=1.2, borderaxespad=0.0)
    fig.tight_layout()
    save(fig, "fig_ablation")


# -------------------------------------------------------- Fig 7: case study
def fig_case_study(city="hz", days=slice(60, 63)):
    """Claim: interval quality is heterogeneous across regions - honest view of
    a well-covered and a poorly-covered region."""
    d = np.load(NPZ / f"preds_{city}_v3.npz")
    q = d["pred_quantiles"]      # (D, R, 24, 5)
    y = d["y_true"]              # (D, R, 24)
    med = q[..., 2]
    mae_r = np.abs(med - y).mean(axis=(0, 2))
    good, bad = int(np.argmin(mae_r)), int(np.argmax(mae_r))

    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_W, 3.2), sharex=True)
    for ax, r, tag in [(axes[0], good, "well-predicted region"),
                       (axes[1], bad, "hard region")]:
        yy = y[days, r].reshape(-1)
        lo = q[days, r, :, 0].reshape(-1)
        hi = q[days, r, :, 4].reshape(-1)
        md = q[days, r, :, 2].reshape(-1)
        t = np.arange(len(yy))
        ax.fill_between(t, lo, hi, color=C_BLUE, alpha=0.18, lw=0,
                        label="90% predictive interval")
        ax.plot(t, md, color=C_BLUE, lw=1.1, label="Median forecast")
        ax.plot(t, yy, color="k", lw=0.9, ls="-", label="Observed demand")
        cov = float(((yy >= lo) & (yy <= hi)).mean())
        ax.set_title(f"{CITY_NAME[city]} - {tag} (coverage {cov * 100:.0f}%)",
                     loc="left")
        ax.set_ylabel("Parcels / h")
        for k in range(24, len(yy), 24):
            ax.axvline(k, color=C_GRAY, lw=0.5, ls=":")
    axes[1].set_xlabel("Hour (3 consecutive days)")
    # legend above the top panel so it cannot cover the interval band/curves
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, fontsize=6, ncols=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.985), columnspacing=0.8,
               handlelength=1.2)
    fig.tight_layout()
    save(fig, "fig_case_study")


# ------------------------------------------------- Fig 8: framework overview
def fig_framework():
    """Graphical abstract: zero-to-calibrated staged cold-start framework."""
    fig, ax = plt.subplots(figsize=(DOUBLE_W, 3.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="none", fs=7.2, tc="k", bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, color=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=color, lw=1.1))

    # inputs
    box(0.15, 3.5, 1.9, 1.1, "Source cities (LaDe)\ndemand + regions", "#E8F0F8")
    box(0.15, 2.0, 1.9, 1.1, "LLM geographic\nembeddings (offline)", "#E8F0F8")
    box(0.15, 0.5, 1.9, 1.1, "Target city\nzero historical data", "#FDECEC")

    # model
    box(2.7, 2.1, 2.1, 2.3, "Inductive ST backbone\nkNN graph + TCN\nquantile heads\n$q_{0.05}\\ldots q_{0.95}$",
        "#DCEBF7", bold=False)
    arrow(2.05, 4.0, 2.7, 3.6)
    arrow(2.05, 2.55, 2.7, 2.9)
    arrow(2.05, 1.05, 2.7, 2.3)

    # stage panel
    box(5.35, 3.4, 2.0, 1.15, "Day 0\nsource-calibrated\nsplit conformal", "#FBE8D8")
    box(5.35, 1.85, 2.0, 1.15, "Day 1+\nonline conformal\n(full recompute)", "#DFF2E9")
    box(5.35, 0.3, 2.0, 1.15, "Day 1+\nonline ensemble\npoint forecast", "#DFF2E9")
    arrow(4.8, 3.3, 5.35, 3.85)
    arrow(4.8, 3.0, 5.35, 2.45)
    arrow(4.8, 2.7, 5.35, 0.95)

    # outcomes
    box(8.0, 3.4, 1.85, 1.15, "Calibrated 90%\nintervals in 1-3 days", "#D6E9DC", bold=True)
    box(8.0, 1.85, 1.85, 1.15, "Newsvendor capacity\ncost \u221225% to \u221265%", "#D6E9DC", bold=True)
    box(8.0, 0.3, 1.85, 1.15, "Point MAE \u2264 best\nsingle predictor", "#D6E9DC")
    arrow(7.35, 3.97, 8.0, 3.97)
    arrow(7.35, 2.42, 8.0, 2.42)
    arrow(7.35, 0.87, 8.0, 0.87)

    ax.text(0.15, 4.85, "Zero-to-calibrated: staged cold-start forecasting for a new city",
            fontsize=9, fontweight="bold")
    save(fig, "fig_framework")


if __name__ == "__main__":
    fig_motivation()
    fig_convergence()
    fig_cov_width()
    fig_staged()
    fig_decision()
    fig_ablation()
    fig_case_study()
    fig_framework()
    print("all figures written to", OUT)
