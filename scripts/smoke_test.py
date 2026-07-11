"""Smoke test for the backbone, calibration, and decision modules
(synthetic data, no training)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.backbone import ZeroShotSTBackbone
from src.calibration.weighted_conformal import calibrate_region, apply_interval, coverage_width
from src.decision.newsvendor import evaluate_policies

# ---- backbone forward (history mode + zero-shot mode) ----
N, T, B = 30, 24, 4
model = ZeroShotSTBackbone(static_dim_in=16, llm_dim=4096, hidden=64, out_len=24,
                           quantiles=[0.05, 0.5, 0.95])
static = torch.randn(N, 16)
llm = torch.randn(N, 4096)
cal = torch.zeros(B, 31); cal[:, 8] = 1; cal[:, 24 + 2] = 1
out_hist = model(static, llm, cal, history=torch.rand(B, N, T))
out_zero = model(static, llm, cal, history=None)
assert out_hist.shape == (B, N, 24, 3) and out_zero.shape == (B, N, 24, 3)
print("backbone OK:", tuple(out_hist.shape), "zero-shot mode OK:", tuple(out_zero.shape))

# ---- weighted conformal ----
rng = np.random.default_rng(0)
M = 500
y_cal = rng.gamma(2, 2, M)
qlo, qhi = y_cal - rng.uniform(1, 2, M), y_cal + rng.uniform(1, 2, M)
z_cal, z_t = rng.normal(size=(M, 64)), rng.normal(size=64)
qh = calibrate_region(y_cal, qlo, qhi, z_cal, z_t, alpha=0.1, tau=0.1)
lo, hi = apply_interval(y_cal - 1.5, y_cal + 1.5, qh)
cov, wid = coverage_width(y_cal, lo, hi)
print(f"conformal OK: q_hat={qh:.3f} coverage={cov:.3f} width={wid:.2f}")

# ---- newsvendor ----
y = rng.gamma(2, 2, (100, N))
rows = evaluate_policies(y, y * 0.9, y * 1.3, [[1, 1], [1, 3]])
print("newsvendor OK:", [f"co={r['co']},cu={r['cu']},saving={r['saving_pct']:.1f}%" for r in rows])
print("ALL SMOKE TESTS PASSED")
