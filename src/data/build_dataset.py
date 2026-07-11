"""LaDe pickup CSVs -> hourly region-level demand tensor + static region features.

Conventions match IMPEL exactly (so its pre-generated Llama-3 region embeddings
align row-by-row):
- source: LaDe pickup_{city}.csv ("delivery demand" in the IMPEL paper refers
  to pickup order counts)
- demand: hourly counts by pickup_time
- region order: first-appearance order of region_id after sorting by
  pickup_time (df.region_id.unique())

Output (one npz per city):
  demand [T,N] / static [N,F] / regions [N] / timestamps [T]
Under the zero-shot protocol, the target city may only use static (incl. LLM
embeddings); demand is used for evaluation only.
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml


def build_city(csv_path: str, year: int, freq: str):
    df = pd.read_csv(csv_path, usecols=["region_id", "lng", "lat", "aoi_type", "pickup_time"])
    df["ts"] = pd.to_datetime(f"{year}-" + df["pickup_time"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")

    regions = list(df["region_id"].unique())  # IMPEL: first-appearance order
    r2i = {r: i for i, r in enumerate(regions)}

    hours = pd.date_range(df["ts"].min().floor("h"), df["ts"].max().floor("h"), freq=freq)
    h2i = {h: i for i, h in enumerate(hours)}
    demand = np.zeros((len(hours), len(regions)), dtype=np.float32)
    for (h, r), cnt in df.groupby([df["ts"].dt.floor("h"), "region_id"]).size().items():
        demand[h2i[h], r2i[r]] = cnt

    cent = df.groupby("region_id")[["lng", "lat"]].mean().loc[regions]
    aoi = (
        df.groupby(["region_id", "aoi_type"]).size().unstack(fill_value=0)
        .reindex(regions, fill_value=0).reindex(columns=range(1, 15), fill_value=0)
    )
    aoi_frac = aoi.div(aoi.sum(axis=1).replace(0, 1), axis=0)
    static = np.concatenate([cent.values, aoi_frac.values], axis=1).astype(np.float32)

    # Trim the low-coverage warm-up period at the start (e.g. LaDe Shanghai's
    # first ~29 days have near-zero daily means -- a collection ramp-up artifact).
    days = len(hours) // 24
    dm = demand[: days * 24].reshape(days, 24, -1).mean(axis=(1, 2))
    thr = 0.3 * np.median(dm[dm > 0])
    start_day = int(np.argmax(dm >= thr))
    if start_day > 0:
        print(f"  trim warm-up: drop first {start_day} days (daily mean < {thr:.2f})")
        demand = demand[start_day * 24:]
        hours = hours[start_day * 24:]
    return demand, static, np.array(regions), hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["data"]

    os.makedirs(cfg["processed_dir"], exist_ok=True)
    for c in cfg["cities"]:
        demand, static, regions, hours = build_city(
            os.path.join(cfg["lade_dir"], f"pickup_{c}.csv"), cfg["year"], cfg["freq"]
        )
        out = os.path.join(cfg["processed_dir"], f"{c}.npz")
        np.savez_compressed(out, demand=demand, static=static, regions=regions,
                            timestamps=np.array([str(h) for h in hours]))
        print(f"{c.upper()}: demand {demand.shape}, static {static.shape} -> {out}")


if __name__ == "__main__":
    main()
