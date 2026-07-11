"""Leave-one-city-out zero-shot experiment: train on three source cities, then
predict the fully unobserved target city and export calibration artifacts.

Usage:
  python src/train_zeroshot.py --config configs/default.yaml --target hz
Outputs:
  results/zeroshot_{target}.json   point accuracy / interval quality metrics
  results/preds_{target}.npz       predicted quantiles + ground truth (for replays)
  results/calib_{target}.npz       source calibration scores + embeddings
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.backbone import ZeroShotSTBackbone
from src.calibration.weighted_conformal import cqr_score

MAIN_CITIES = ["sh", "hz", "cq", "yt"]  # jl has only 4 regions; excluded from the main study


def load_city(processed_dir, impel_dir, c):
    d = np.load(os.path.join(processed_dir, f"{c}.npz"), allow_pickle=True)
    emb = np.load(os.path.join(impel_dir, f"llmvec_llama3_Delivery_{c.upper()}.npy"))
    demand = d["demand"]          # [T, N]
    static = d["static"]          # [N, 16]
    ts = d["timestamps"]
    assert emb.shape[0] == demand.shape[1], f"{c}: emb/region mismatch"
    return demand, static, emb.astype(np.float32), ts


def calendar_feat(ts_str):
    """hour one-hot(24) + day-of-week one-hot(7)"""
    import datetime as dt
    t = dt.datetime.fromisoformat(str(ts_str))
    v = np.zeros(31, dtype=np.float32)
    v[t.hour] = 1.0
    v[24 + t.weekday()] = 1.0
    return v


def make_windows(T, in_len, out_len):
    return np.arange(0, T - in_len - out_len + 1)


def how_index(ts_str):
    """hour-of-week in [0,168)"""
    import datetime as dt
    t = dt.datetime.fromisoformat(str(ts_str))
    return t.weekday() * 24 + t.hour


def pinball_loss(pred, y, quantiles):
    """pred [B,N,H,Q], y [B,N,H]"""
    loss = 0.0
    for i, q in enumerate(quantiles):
        e = y - pred[..., i]
        loss = loss + torch.maximum(q * e, (q - 1) * e).mean()
    return loss / len(quantiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--target", required=True, choices=MAIN_CITIES)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--calib_only", action="store_true",
                    help="skip training; load results/ckpt_{target}.pt and re-export artifacts")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed; non-zero seeds add an _s{seed} suffix to outputs")
    ap.add_argument("--retrieval", action="store_true", help="enable source-memory retrieval augmentation")
    ap.add_argument("--retr_shape", action="store_true", help="mean-center retrieved profiles (shape only)")
    ap.add_argument("--anchor", action="store_true",
                    help="HA anchor-residual learning: prediction = source hour-of-week anchor + model residual")
    ap.add_argument("--lr", type=float, default=None, help="override config learning rate")
    ap.add_argument("--clip", type=float, default=5.0, help="gradient-norm clipping threshold")
    ap.add_argument("--tag", default="", help="suffix appended to output filenames, e.g. _retr")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = cfg["experiment"]["seed"] if args.seed is None else args.seed
    torch.manual_seed(seed); np.random.seed(seed)
    sfx = ("" if seed == 0 else f"_s{seed}") + args.tag

    in_len, out_len = cfg["data"]["input_len"], cfg["data"]["output_len"]
    Q = cfg["model"]["quantiles"]
    sources = [c for c in MAIN_CITIES if c != args.target]
    print(f"target={args.target} sources={sources} device={dev}")

    data = {c: load_city(cfg["data"]["processed_dir"], cfg["model"]["llm_emb_path"], c)
            for c in MAIN_CITIES}

    # ------- source-city sample indices: chronological 70/10/20 train/val/calib -------
    splits = {}
    for c in sources:
        T = data[c][0].shape[0]
        w = make_windows(T, in_len, out_len)
        n = len(w)
        splits[c] = {"train": w[: int(0.7 * n)], "val": w[int(0.7 * n): int(0.8 * n)],
                     "calib": w[int(0.8 * n):]}

    model = ZeroShotSTBackbone(
        static_dim_in=data[sources[0]][1].shape[1], llm_dim=cfg["model"]["llm_emb_dim"],
        hidden=cfg["model"]["hidden_dim"], out_len=out_len, quantiles=Q,
        tcn_layers=cfg["model"]["tcn_layers"], knn_k=cfg["model"]["knn_k"],
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or cfg["train"]["lr"])

    tensors = {}
    for c in MAIN_CITIES:
        demand, static, emb, ts = data[c]
        tensors[c] = {
            "y_log": torch.tensor(np.log1p(demand), dtype=torch.float32, device=dev),
            "static": torch.tensor(static, device=dev),
            "emb": torch.tensor(emb, device=dev),
            "ts": ts,
            "how": np.array([how_index(s) for s in ts]),
        }

    # ---------- retrieval augmentation: source hour-of-week profile memory ----------
    # During training, same-city retrieval is disabled (to mimic the zero-shot
    # condition); profiles use only each source city's train segment (no leakage).
    # Shape mode injects mean-centered profiles (shape without scale) to avoid
    # importing cross-city demand scale.
    HOW = 168

    def city_profile(c):
        """Train-segment hour-of-week profile (log1p space) [N,168]."""
        y = tensors[c]["y_log"].cpu().numpy()
        end = int(splits[c]["train"][-1]) + in_len + out_len
        how = tensors[c]["how"][:end]
        p = np.zeros((HOW, y.shape[1]), dtype=np.float32)
        for hw in range(HOW):
            m = how == hw
            p[hw] = y[:end][m].mean(axis=0) if m.any() else y[:end].mean(axis=0)
        return p.T

    src_profiles = {c: city_profile(c) for c in sources} if (args.retrieval or args.anchor) else {}

    # HA anchor: mean curve over the remaining source cities (leave-one-out, no leakage)
    anchor_prof = {}
    if args.anchor:
        for c in MAIN_CITIES:
            others = [s for s in sources if s != c] or sources
            curve = np.mean([src_profiles[s].mean(axis=0) for s in others], axis=0)
            anchor_prof[c] = torch.tensor(curve, dtype=torch.float32, device=dev)   # [168]

    retr_prof = {}
    if args.retrieval:
        prof = {}
        for c in sources:
            pc = src_profiles[c].copy()
            if args.retr_shape:
                pc = pc - pc.mean(axis=1, keepdims=True)
            prof[c] = pc
        bank_emb = np.concatenate([data[c][2] for c in sources])
        bank_prof = np.concatenate([prof[c] for c in sources])
        bank_city = np.concatenate([np.full(data[c][2].shape[0], c) for c in sources])
        bank_n = bank_emb / np.linalg.norm(bank_emb, axis=1, keepdims=True)
        K = cfg["model"]["knn_k"]
        for c in MAIN_CITIES:
            q = data[c][2]
            S = (q / np.linalg.norm(q, axis=1, keepdims=True)) @ bank_n.T
            if c in sources:
                S[:, bank_city == c] = -np.inf
            idx = np.argsort(-S, axis=1)[:, :K]
            s_top = np.take_along_axis(S, idx, axis=1)
            z = (s_top - s_top.mean(axis=1, keepdims=True)) / (s_top.std(axis=1, keepdims=True) + 1e-8)
            w = np.exp(z); w /= w.sum(axis=1, keepdims=True)
            retr_prof[c] = torch.tensor((bank_prof[idx] * w[..., None]).sum(axis=1), device=dev)

    def batch_forward(c, widx, zero_shot: bool):
        t = tensors[c]
        hist = torch.stack([t["y_log"][i: i + in_len].T for i in widx])          # [B,N,T]
        y = torch.stack([t["y_log"][i + in_len: i + in_len + out_len].T for i in widx])
        cal = torch.tensor(np.stack([calendar_feat(t["ts"][i + in_len]) for i in widx]), device=dev)
        retr, hw = None, None
        if args.retrieval or args.anchor:
            hw = torch.tensor(np.stack([t["how"][i + in_len: i + in_len + out_len] for i in widx]),
                              device=dev, dtype=torch.long)                       # [B,H]
        if args.retrieval:
            retr = retr_prof[c][:, hw].permute(1, 0, 2)                           # [B,N,H]
        pred = model(t["static"], t["emb"], cal, history=None if zero_shot else hist, retr=retr)
        if args.anchor:
            pred = pred + anchor_prof[c][hw][:, None, :, None]                    # residual learning
        return pred, y

    # ------------------------------ training ------------------------------
    bs = cfg["train"]["batch_size"]
    max_epochs = 0 if args.calib_only else (args.max_epochs or cfg["train"]["max_epochs"])
    patience, best_val, bad = cfg["train"]["patience"], float("inf"), 0
    t0 = time.time()
    for ep in range(max_epochs):
        model.train()
        pool = [(c, i) for c in sources for i in splits[c]["train"]]
        np.random.shuffle(pool)
        ep_loss, steps = 0.0, 0
        for k in range(0, len(pool), bs):
            chunk = pool[k: k + bs]
            for c in set(x[0] for x in chunk):
                widx = [i for cc, i in chunk if cc == c]
                # Route 50% of batches through the zero-shot path so the
                # generative mode is a trained capability.
                pred, y = batch_forward(c, widx, zero_shot=(np.random.rand() < 0.5))
                loss = pinball_loss(pred, y, Q)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # guards against divergent seeds
                opt.step()
                ep_loss += loss.item(); steps += 1
        model.eval()
        with torch.no_grad():
            vl = np.mean([
                pinball_loss(*batch_forward(c, splits[c]["val"][k: k + bs], True), Q).item()
                for c in sources for k in range(0, len(splits[c]["val"]), bs)
            ])
        if vl < best_val - 1e-4:
            best_val, bad = vl, 0
            torch.save(model.state_dict(), f"/tmp/best_{args.target}{sfx}.pt")
        else:
            bad += 1
        print(f"ep{ep} train={ep_loss/max(steps,1):.4f} val_zs={vl:.4f} best={best_val:.4f} bad={bad}", flush=True)
        if bad >= patience:
            break
    ckpt = f"results/ckpt_{args.target}{sfx}.pt" if args.calib_only else f"/tmp/best_{args.target}{sfx}.pt"
    model.load_state_dict(torch.load(ckpt))
    train_min = (time.time() - t0) / 60

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), f"results/ckpt_{args.target}{sfx}.pt")  # persist for calibration-only reruns

    # ------------------------- source calibration scores -------------------------
    # Per-point (window x region x horizon) CQR scores + scale proxy + region index;
    # embeddings are stored once per region.
    model.eval()
    ilo, ihi, imed = 0, len(Q) - 1, Q.index(0.5)
    sc_abs, sc_scale, sc_reg, src_embs = [], [], [], []
    reg_offset = 0
    with torch.no_grad():
        for c in sources:
            t = tensors[c]
            N = t["emb"].shape[0]
            src_embs.append(t["emb"].cpu().numpy())
            for k in range(0, len(splits[c]["calib"]), bs):
                widx = splits[c]["calib"][k: k + bs]
                pred, y = batch_forward(c, widx, zero_shot=True)
                p = torch.expm1(pred).cpu().numpy(); yy = torch.expm1(y).cpu().numpy()
                s = cqr_score(yy, p[..., ilo], p[..., ihi])          # [B,N,H]
                sc_abs.append(s.reshape(-1))
                sc_scale.append((p[..., imed] + 1.0).reshape(-1))
                sc_reg.append(np.broadcast_to(
                    np.arange(N)[None, :, None] + reg_offset, s.shape).reshape(-1))
            reg_offset += N
    calib_scores = np.concatenate(sc_abs)
    calib_scale = np.concatenate(sc_scale)
    calib_region = np.concatenate(sc_reg)
    if len(calib_scores) > 500_000:
        sel = np.random.choice(len(calib_scores), 500_000, replace=False)
        calib_scores, calib_scale, calib_region = (
            calib_scores[sel], calib_scale[sel], calib_region[sel])
    np.savez_compressed(
        f"results/calib_{args.target}{sfx}.npz",
        scores=calib_scores.astype(np.float32), scale=calib_scale.astype(np.float32),
        region_idx=calib_region.astype(np.int32),
        source_emb=np.concatenate(src_embs).astype(np.float32),
    )

    # --------------------------- target-city zero-shot evaluation ---------------------------
    t = tensors[args.target]
    Ttot = t["y_log"].shape[0]
    widx_all = make_windows(Ttot, in_len, out_len)[:: out_len]  # non-overlapping windows
    preds, ys = [], []
    with torch.no_grad():
        for k in range(0, len(widx_all), bs):
            pred, y = batch_forward(args.target, widx_all[k: k + bs], zero_shot=True)
            preds.append(torch.expm1(pred).cpu().numpy()); ys.append(torch.expm1(y).cpu().numpy())
    P = np.concatenate(preds)   # [W,N,H,Q]
    Y = np.concatenate(ys)      # [W,N,H]
    mae = float(np.abs(P[..., imed] - Y).mean())
    rmse = float(np.sqrt(((P[..., imed] - Y) ** 2).mean()))

    # naive split-conformal baseline (absolute scores); the full grid of
    # calibration variants is evaluated in analyze_calibration.py
    alpha = cfg["conformal"]["alpha"]
    q_naive = float(np.quantile(calib_scores, min(1.0, (1 - alpha) * (1 + 1 / len(calib_scores)))))
    lo_n, hi_n = P[..., ilo] - q_naive, P[..., ihi] + q_naive
    cov_n = float(((Y >= lo_n) & (Y <= hi_n)).mean()); wid_n = float((hi_n - lo_n).mean())

    res = {
        "target": args.target, "sources": sources, "train_minutes": round(train_min, 1),
        "point": {"MAE": round(mae, 4), "RMSE": round(rmse, 4)},
        "interval_naive": {"coverage": round(cov_n, 4), "width": round(wid_n, 3), "q_hat": round(q_naive, 3)},
        "n_calib": int(len(calib_scores)), "alpha": alpha,
    }
    json.dump(res, open(f"results/zeroshot_{args.target}{sfx}.json", "w"), indent=2)
    np.savez_compressed(
        f"results/preds_{args.target}{sfx}.npz",
        pred_quantiles=P.astype(np.float32), y_true=Y.astype(np.float32),
        target_emb=t["emb"].cpu().numpy().astype(np.float32),
        quantiles=np.array(Q),
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
