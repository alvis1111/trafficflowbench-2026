"""Task 2 shockwave: test upstream-dilation of the queue front on train windows.

The queue front propagates upstream at the backward wave speed. We approximate
this by dilating the T-5 queue state upstream by K links per 5-minute cell and
measure IoU vs persistence.
"""
import glob
import os
import numpy as np
import pandas as pd

PANEL = "D12_I5_N"
ROOT = "C:/Users/GEM07/OneDrive/Desktop/kaggle_public"

links = pd.read_csv(f"{ROOT}/corridors/{PANEL}/network/links.csv", dtype={"link_id": str})
v_cut = dict(zip(links.link_id, links.free_speed_kmh * 0.60))
topo = pd.read_csv(f"{ROOT}/corridors/{PANEL}/network/lwr_mainline_topology.csv", dtype=str)
order = {r["mainline_link_id"]: int(r["order_index"]) for _, r in topo.iterrows()}
max_order = max(order.values())

widx = pd.read_csv(f"{ROOT}/task2/{PANEL}/train/window_index.csv")
hist = pd.read_parquet(f"{ROOT}/task2/{PANEL}/train/window_history.parquet")
truth_files = {}
for p in glob.glob(f"{ROOT}/corridors/{PANEL}/train/mainline_states/year_month=*/*.parquet"):
    truth_files[os.path.basename(p).replace("synthetic_mainline_", "").replace(".parquet", "")] = p


def queue_truth(date, fstart, fend):
    day = pd.read_parquet(truth_files[date], columns=["timestamp", "link_id", "speed_kmh"])
    day["link_id"] = day.link_id.astype(str); day["ts"] = pd.to_datetime(day.timestamp, utc=True)
    h = day[(day.ts >= fstart) & (day.ts <= fend)].copy()
    h["q"] = (h.speed_kmh <= h.link_id.map(v_cut)).astype(int)
    return h


def dilate(q0: dict, k: int) -> dict:
    """Dilate the queue upstream (higher order_index) by k links."""
    out = dict(q0)
    if k <= 0:
        return out
    for link, q in q0.items():
        if q and link in order:
            o = order[link]
            for target, to in order.items():
                if o < to <= o + k:
                    out[target] = 1
    return out


def eval_K(K):
    rows = []
    for _, w in widx.iterrows():
        fstart = pd.Timestamp(w.forecast_start); fend = pd.Timestamp(w.forecast_end)
        truth = queue_truth(w.date.replace("-", "_"), fstart, fend)
        h = hist[hist.window_id == w.window_id].copy()
        h["ts"] = pd.to_datetime(h.timestamp, utc=True)
        org = h[h.ts == h.ts.max()][["link_id", "speed_kmh"]].copy()
        org["q0"] = (org.speed_kmh <= org.link_id.map(v_cut)).astype(int)
        q0 = dict(zip(org.link_id, org.q0))
        # for each horizon cell, dilate by K * (time index + 1)
        preds = []
        for i, (_, r) in enumerate(truth.groupby("ts")):
            d = dilate(q0, K * (i + 1)) if w.condition == "queue_ongoing" else dict(q0)
            p = r.link_id.map(d).fillna(0).astype(int)
            preds.append(p)
        pred = np.concatenate(preds)
        a = pred.to_numpy() if hasattr(pred, "to_numpy") else np.asarray(pred)
        b = truth.q.to_numpy()
        inter = int((a & b).sum()); union = int((a | b).sum())
        iou = 1.0 if union == 0 else inter / union
        rows.append({"window": w.window_id, "condition": w.condition, "iou": iou})
    res = pd.DataFrame(rows)
    return res.iou.mean(), res[res.condition == "queue_onset"].iou.mean(), res[res.condition == "queue_ongoing"].iou.mean()


print(f"persistence baseline: overall=0.2812 onset=0.0 ongoing=0.5624")
for K in [0, 1, 2, 3, 5, 8]:
    ov, on, og = eval_K(K)
    print(f"K={K} (links/5min): overall={ov:.4f}  onset={on:.4f}  ongoing={og:.4f}")
