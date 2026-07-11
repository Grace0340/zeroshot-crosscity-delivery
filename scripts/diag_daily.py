"""Diagnostic: daily mean demand per city (checks for low-coverage days at the
start or end of the collection period)."""
import numpy as np
import yaml

cfg = yaml.safe_load(open("configs/default.yaml"))

for c in ["sh", "hz", "cq", "yt"]:
    d = np.load(f"{cfg['data']['processed_dir']}/{c}.npz")["demand"]
    T, N = d.shape
    days = T // 24
    m = d[: days * 24].reshape(days, 24, N).mean(axis=(1, 2))
    print(f"{c}: T={T} N={N} days={days} overall_daily_mean={m.mean():.2f}")
    print("  daily means:", np.round(m, 1).tolist())
