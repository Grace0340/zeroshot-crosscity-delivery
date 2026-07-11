"""Similarity-weighted split conformal prediction.

- Calibration points come from source-city (region, window) samples; the
  nonconformity score follows CQR: s = max(q_lo - y, y - q_hi).
- For a target region j, source sample i receives weight
  w_i ~ exp(cos(z_i, z_j) / tau), where z are LLM geographic embeddings
  (a similarity-kernel instance of weighted conformal prediction,
  Tibshirani et al. 2019).
- The weighted empirical quantile of the scores expands/shrinks the raw
  interval [q_lo, q_hi].

Unweighted split conformal is the tau -> inf special case of this module.
"""
import numpy as np


def cqr_score(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    return np.maximum(q_lo - y, y - q_hi)


def weighted_quantile(scores, weights, q):
    """Weighted empirical quantile (conservative +inf point mass correction)."""
    idx = np.argsort(scores)
    s, w = scores[idx], weights[idx]
    cw = np.cumsum(w) / (w.sum() + 1)  # +1 accounts for the test point's own mass
    k = np.searchsorted(cw, q)
    return s[min(k, len(s) - 1)] if k < len(s) else np.inf


def geo_weights(z_calib: np.ndarray, z_target: np.ndarray, tau: float, min_w: float = 1e-3):
    """z_calib [M,D] source calibration-region embeddings; z_target [D]."""
    zc = z_calib / np.linalg.norm(z_calib, axis=1, keepdims=True)
    zt = z_target / np.linalg.norm(z_target)
    sim = zc @ zt
    w = np.exp(sim / tau)
    return np.maximum(w / w.max(), min_w)


def calibrate_region(y_cal, qlo_cal, qhi_cal, z_cal, z_tgt, alpha=0.1, tau=0.1):
    """Return the quantile correction q_hat for one target region
    (>= 0 widens the interval, < 0 shrinks it)."""
    scores = cqr_score(y_cal, qlo_cal, qhi_cal)
    w = geo_weights(z_cal, z_tgt, tau)
    return weighted_quantile(scores, w, 1 - alpha)


def apply_interval(q_lo, q_hi, q_hat):
    return q_lo - q_hat, q_hi + q_hat


def coverage_width(y, lo, hi):
    cover = ((y >= lo) & (y <= hi)).mean()
    width = (hi - lo).mean()
    return float(cover), float(width)
