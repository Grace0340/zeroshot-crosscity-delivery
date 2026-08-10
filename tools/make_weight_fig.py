# -*- coding: utf-8 -*-
"""Ensemble weight trajectories, reconstructed from the staged replay.

Applies the ensemble's inverse-EMA weighting rule (Eq. (4) of the paper,
EMA decay 0.7) to the seed-averaged daily member MAE series stored in
results/npz/staged_<city>_v3.npz. The reconstruction replicates the update
order of eval_staged.py: the weights used for day k are computed from EMAs
updated through day k-1; day 0 falls back to pure climatology, and
persistence joins the member set once its first daily error is observed.
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "code" / "results" / "npz"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

CITIES = ["sh", "hz", "cq", "yt"]
CITY_NAME = {"sh": "Shanghai", "hz": "Hangzhou", "cq": "Chongqing", "yt": "Yantai"}

C_BLUE = "#0072B2"    # model
C_ORANGE = "#E69F00"  # accumulating target climatology
C_GREEN = "#009E73"   # persistence

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 120,
})

SINGLE_W = 3.5
EMA_BETA = 0.7


def weight_trajectories(city):
    d = np.load(NPZ / f"staged_{city}_v3.npz")
    m, h, p = d["model-staged"], d["ha-target"], d["persistence"]
    T = len(m)
    w = np.full((T, 3), np.nan)  # columns: model, climatology, persistence
    ema_m = ema_h = ema_p = None
    for k in range(T):
        # weights in effect for day k (EMAs observed through day k-1)
        if ema_m is None:
            w[k] = [0.0, 1.0, 0.0]              # day 0: climatology fallback
        elif ema_p is None:
            wm, wh = 1.0 / (ema_m + 1e-6), 1.0 / (ema_h + 1e-6)
            w[k] = [wm / (wm + wh), wh / (wm + wh), 0.0]
        else:
            wm, wh = 1.0 / (ema_m + 1e-6), 1.0 / (ema_h + 1e-6)
            wp = 1.0 / (ema_p + 1e-6)
            s = wm + wh + wp
            w[k] = [wm / s, wh / s, wp / s]
        # update EMAs with day k's observed errors
        ema_m = m[k] if ema_m is None else EMA_BETA * ema_m + (1 - EMA_BETA) * m[k]
        ema_h = h[k] if ema_h is None else EMA_BETA * ema_h + (1 - EMA_BETA) * h[k]
        if not np.isnan(p[k]):
            ema_p = p[k] if ema_p is None else EMA_BETA * ema_p + (1 - EMA_BETA) * p[k]
    return w


def main():
    fig, axes = plt.subplots(2, 2, figsize=(SINGLE_W, 2.7), sharex=False, sharey=True)
    for ax, city in zip(axes.ravel(), CITIES):
        w = weight_trajectories(city)
        # plot from day 1: on day 0 the ensemble is the pure climatology
        # fallback (weight 1), which would render as a distracting spike
        x = np.arange(1, len(w))
        ax.plot(x, w[1:, 0], color=C_BLUE, lw=1.2, label="Model")
        ax.plot(x, w[1:, 1], color=C_ORANGE, lw=1.2, label="Target climatology")
        ax.plot(x, w[1:, 2], color=C_GREEN, lw=1.2, label="Persistence")
        ax.set_title(CITY_NAME[city], pad=2)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(1, len(w) - 1)
        ax.tick_params(length=2)
        print(f"{city}: final weights model={w[-1,0]:.2f} "
              f"clim={w[-1,1]:.2f} pers={w[-1,2]:.2f}")
    for ax in axes[1]:
        ax.set_xlabel("Deployment day")
    for ax in axes[:, 0]:
        ax.set_ylabel("Weight")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(pad=0.4, h_pad=0.8)
    fig.savefig(OUT / "fig_ensemble_weights.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_ensemble_weights.png", dpi=300, bbox_inches="tight")
    print("saved fig_ensemble_weights")


if __name__ == "__main__":
    main()
