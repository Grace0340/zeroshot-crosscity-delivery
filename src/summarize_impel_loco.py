"""Parse IMPEL transfer_partial logs into LOCO summary JSON.

Expects logs named impel_loco_{SRC}_{TGT}.log under logs/.
"""
import argparse
import json
import re
from pathlib import Path

MAIN = ["SH", "HZ", "CQ", "YT"]


def parse_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Average Test MAE:\s*([\d.]+),\s*Test RMSE:\s*([\d.]+)", text)
    if not m:
        return None
    return {"mae": float(m.group(1)), "rmse": float(m.group(2))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="../logs")
    ap.add_argument("--out", default="results/impel_loco.json")
    args = ap.parse_args()
    log_dir = Path(args.log_dir)

    pairs = {}
    for tgt in MAIN:
        pairs[tgt.lower()] = {}
        for src in MAIN:
            if src == tgt:
                continue
            log = log_dir / f"impel_loco_{src}_{tgt}.log"
            if log.exists():
                pairs[tgt.lower()][src.lower()] = parse_log(log)

    summary = {}
    for tgt, srcs in pairs.items():
        vals = [v for v in srcs.values() if v]
        if not vals:
            continue
        summary[tgt] = {
            "per_source": srcs,
            "mean_mae": round(sum(v["mae"] for v in vals) / len(vals), 4),
            "mean_rmse": round(sum(v["rmse"] for v in vals) / len(vals), 4),
            "best_mae": round(min(v["mae"] for v in vals), 4),
            "best_source": min(srcs, key=lambda s: srcs[s]["mae"] if srcs[s] else 1e9),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
